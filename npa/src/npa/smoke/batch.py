"""Batch runner for container golden evals."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.manifest import container, load_manifest


@dataclass(frozen=True)
class ContainerRunResult:
    name: str
    mode: str
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    exit_code: int | None = None
    status: str = ""
    gpu: str = ""
    command: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    results: list[ContainerRunResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok or r.skipped for r in self.results)

    @property
    def ran(self) -> list[ContainerRunResult]:
        return [r for r in self.results if not r.skipped]

    def to_json(self) -> str:
        payload = {
            "ok": self.ok,
            "total": len(self.results),
            "ran": len(self.ran),
            "passed": sum(1 for r in self.ran if r.ok),
            "failed": sum(1 for r in self.ran if not r.ok),
            "skipped": sum(1 for r in self.results if r.skipped),
            "results": [asdict(r) for r in self.results],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def iter_containers(
    *,
    include_blocked: bool = False,
    include_foundation: bool = True,
    tools_only: bool = False,
) -> list[str]:
    specs = load_manifest()
    names: list[str] = []
    for name in sorted(specs):
        if tools_only and name not in CONTAINER_IMAGE_NAMES:
            continue
        if not include_foundation and specs[name].foundation:
            continue
        if not include_blocked and specs[name].golden_eval.status == "blocked-on-upstream":
            continue
        names.append(name)
    return names


def run_container_eval(
    name: str,
    *,
    serverless: bool = False,
    execute: bool = False,
    gpu: str = "",
    timeout: str = "40m",
    on_state_change: Callable[[object], None] | None = None,
) -> ContainerRunResult:
    spec = container(name)
    ge = spec.golden_eval
    mode = "dry-run"
    if serverless:
        mode = "serverless"
    elif execute:
        mode = "execute"

    base = ContainerRunResult(
        name=name,
        mode=mode,
        ok=True,
        status=ge.status,
        gpu=ge.gpu,
        command=ge.command,
    )

    if serverless:
        from npa.clients.serverless import ServerlessClientError
        from npa.smoke.serverless_runner import submit_golden_eval

        try:
            detail = submit_golden_eval(
                name,
                gpu_type=gpu or None,
                timeout=timeout,
                on_state_change=on_state_change,
            )
        except ServerlessClientError as exc:
            return ContainerRunResult(
                name=name,
                mode=mode,
                ok=False,
                exit_code=1,
                status=ge.status,
                gpu=ge.gpu,
                command=ge.command,
                detail={"error": "ServerlessClientError", "message": str(exc)},
            )
        except (RuntimeError, TimeoutError) as exc:
            return ContainerRunResult(
                name=name,
                mode=mode,
                ok=False,
                exit_code=1,
                status=ge.status,
                gpu=ge.gpu,
                command=ge.command,
                detail={"error": type(exc).__name__, "message": str(exc)},
            )
        return ContainerRunResult(
            name=name,
            mode=mode,
            ok=bool(detail.get("ok")),
            exit_code=0 if detail.get("ok") else 1,
            status=ge.status,
            gpu=ge.gpu,
            command=ge.command,
            detail=detail,
        )

    if not execute:
        return base

    try:
        command_parts = shlex.split(ge.command)
        if command_parts and command_parts[0] == "python":
            command_parts[0] = sys.executable
        completed = subprocess.run(
            command_parts,
            timeout=ge.timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return ContainerRunResult(
            name=name,
            mode=mode,
            ok=False,
            exit_code=2,
            status=ge.status,
            gpu=ge.gpu,
            command=ge.command,
            detail={"error": str(exc)},
        )
    except subprocess.TimeoutExpired:
        return ContainerRunResult(
            name=name,
            mode=mode,
            ok=False,
            exit_code=124,
            status=ge.status,
            gpu=ge.gpu,
            command=ge.command,
            detail={"error": f"timed out after {ge.timeout_seconds}s"},
        )

    return ContainerRunResult(
        name=name,
        mode=mode,
        ok=completed.returncode == 0,
        exit_code=completed.returncode,
        status=ge.status,
        gpu=ge.gpu,
        command=ge.command,
    )


def run_all(
    names: list[str],
    *,
    serverless: bool = False,
    execute: bool = False,
    gpu: str = "",
    timeout: str = "40m",
    parallel: int = 1,
    on_progress: Callable[[ContainerRunResult], None] | None = None,
) -> BatchResult:
    if parallel < 1:
        raise ValueError("parallel must be >= 1")
    if not names:
        return BatchResult()

    batch = BatchResult()

    def _run_one(name: str) -> ContainerRunResult:
        return run_container_eval(
            name,
            serverless=serverless,
            execute=execute,
            gpu=gpu,
            timeout=timeout,
        )

    if parallel == 1 or len(names) == 1:
        for name in names:
            result = _run_one(name)
            batch.results.append(result)
            if on_progress:
                on_progress(result)
        return batch

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_run_one, name): name for name in names}
        for future in as_completed(futures):
            result = future.result()
            batch.results.append(result)
            if on_progress:
                on_progress(result)

    batch.results.sort(key=lambda r: r.name)
    return batch
