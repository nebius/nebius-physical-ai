"""Shared invocation runtime — the single execution path.

Every surface (CLI, SDK, YAML, API) is a thin waiter: it translates the
user's request into (descriptor_id, command, inputs) and calls
``runtime.invoke``. All validation, argv rendering, backend dispatch,
artifact collection, and success evaluation happen here, exactly once.

The runtime translates validated requests into existing execution
machinery (backends); it does not reimplement launching or storage.
There are no per-tool branches here — adding a tool means adding a
descriptor file, never touching this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .catalog import Catalog
from .schema import CommandSpec


class Backend(Protocol):
    name: str

    def run(
        self,
        image_pinned: str,
        argv: list[str],
        gpu: int,
        outputs: list[tuple[str, str]],  # (output name, container path)
        payload: dict[str, str] | None = None,  # container path -> file content
    ) -> "BackendResult": ...


@dataclass
class BackendResult:
    exit_code: int
    logs: str
    artifacts_raw: dict[str, str]  # output name -> file contents
    elapsed_s: float
    stdout: str = ""


@dataclass
class InvocationResult:
    descriptor_id: str
    image_digest: str
    command: str
    surface: str
    backend: str
    inputs: dict[str, Any]
    argv: list[str]
    exit_code: int
    artifacts: dict[str, Any]
    checks: list[dict[str, Any]]
    success: bool
    elapsed_s: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def verification_record(self) -> dict[str, Any]:
        """Evidence of this run, stored separately from author declarations."""
        return {
            "descriptor_id": self.descriptor_id,
            "image_digest": self.image_digest,
            "command": self.command,
            "surface": self.surface,
            "backend": self.backend,
            "inputs": self.inputs,
            "argv": self.argv,
            "exit_code": self.exit_code,
            "artifacts": self.artifacts,
            "checks": self.checks,
            "success": self.success,
            "elapsed_s": round(self.elapsed_s, 3),
            "timestamp": self.timestamp,
        }


class Runtime:
    def __init__(
        self,
        catalog: Catalog,
        backends: dict[str, Backend],
        records_dir: str | Path = "records",
    ):
        self.catalog = catalog
        self.backends = backends
        self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def invoke(
        self,
        tool: str,
        version: str,
        command: str,
        inputs: dict[str, Any] | None = None,
        surface: str = "sdk",
        backend: str = "local",
    ) -> InvocationResult:
        """The one execution path. Every surface calls this."""
        t0 = time.time()
        descriptor = self.catalog.get(tool, version)
        if command not in descriptor.commands:
            raise KeyError(f"{descriptor.id}: unknown command {command!r}")
        spec: CommandSpec = descriptor.commands[command]

        resolved = spec.resolve_inputs(inputs or {})
        argv = spec.render_argv(resolved)

        if backend not in self.backends:
            raise KeyError(f"unknown backend {backend!r}")
        be = self.backends[backend]
        outputs = [(o.name, o.path) for o in spec.outputs.values()]
        payload: dict[str, str] = {}
        for cpath, hpath in descriptor.payload_files:
            p = Path(hpath)
            if not p.is_absolute():
                # Relative to the descriptor's catalog dir parent (project root).
                p = self.catalog.descriptor_dir.parent / hpath
            payload[cpath] = p.read_text()
        bres = be.run(
            image_pinned=descriptor.image.pinned(),
            argv=argv,
            gpu=descriptor.resources.gpu,
            outputs=outputs,
            payload=payload,
        )

        artifacts: dict[str, Any] = {}
        for oname, ospec in spec.outputs.items():
            if ospec.source == "stdout":
                raw: str | None = bres.stdout or None
            else:
                raw = bres.artifacts_raw.get(oname)
            if raw is None:
                continue
            artifacts[oname] = json.loads(raw) if ospec.format == "json" else raw

        check_results: list[dict[str, Any]] = []
        all_ok = bres.exit_code == 0
        for check in descriptor.success_checks:
            ok, msg = check.evaluate(artifacts)
            check_results.append(
                {
                    "check": f"{check.artifact}.{check.json_path}",
                    "ok": ok,
                    "detail": msg,
                }
            )
            all_ok = all_ok and ok
        for oname, ospec in spec.outputs.items():
            if ospec.required and oname not in artifacts:
                all_ok = False
                check_results.append(
                    {
                        "check": f"artifact:{oname}",
                        "ok": False,
                        "detail": "required output missing",
                    }
                )

        result = InvocationResult(
            descriptor_id=descriptor.id,
            image_digest=descriptor.image.digest,
            command=command,
            surface=surface,
            backend=be.name,
            inputs=resolved,
            argv=argv,
            exit_code=bres.exit_code,
            artifacts=artifacts,
            checks=check_results,
            success=all_ok,
            elapsed_s=time.time() - t0,
        )
        # Verification evidence, stored separately from the author's descriptor.
        rec_path = self.records_dir / f"{descriptor.name}-{int(t0)}-{surface}.json"
        rec_path.write_text(json.dumps(result.verification_record(), indent=2))
        return result
