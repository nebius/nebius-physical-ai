#!/usr/bin/env python3
"""Audit the built, restricted NVIDIA Content Agents image.

This scanner proves the deliberate boundary on actual image bytes: reviewed
Content Agents and OVRTX must be present, while optional proprietary components,
sample/customer data, model weights, credentials, and source-control metadata
must be absent. It never decides that the image is publicly redistributable;
success requires the restricted label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence


class ImageAuditError(RuntimeError):
    """Raised when the built image violates its packaging contract."""


Runner = Callable[[Sequence[str]], str]


def _run(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    if completed.returncode:
        raise ImageAuditError(
            f"{Path(argv[0]).name} failed with exit code {completed.returncode}"
        )
    return completed.stdout


def _container_json(image: str, script: str, runner: Runner) -> dict[str, Any]:
    raw = runner(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/venv/bin/python",
            image,
            "-c",
            script,
        ]
    )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ImageAuditError("container audit did not return a JSON object")
    return payload


def audit_image(
    image: str,
    *,
    expected_npa_source_sha: str = "",
    runner: Runner = _run,
) -> dict[str, Any]:
    inspected = json.loads(runner(["docker", "image", "inspect", image]))
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise ImageAuditError("docker inspect did not resolve exactly one image")
    record = inspected[0]
    config = record.get("Config") or {}
    labels = config.get("Labels") or {}
    expected_labels = {
        "npa.tool": "content-agents",
        "npa.redistribution": "restricted",
        "npa.driver_provisioning": "gpu-operator-host-mounted",
        "npa.driver_capabilities": "compute,utility,graphics,display",
        "org.opencontainers.image.revision": (
            "36dbf3f274f8e256637230a05a085853f65cc175"
        ),
        "org.opencontainers.image.version": "0.5.2",
    }
    for key, expected in expected_labels.items():
        if labels.get(key) != expected:
            raise ImageAuditError(f"image label {key} is not {expected!r}")
    source_sha = str(labels.get("npa.source_revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ImageAuditError("image label npa.source_revision is not a full Git SHA")
    if expected_npa_source_sha and source_sha != expected_npa_source_sha:
        raise ImageAuditError(
            "image label npa.source_revision does not match the requested checkpoint"
        )
    licenses = str(labels.get("org.opencontainers.image.licenses") or "")
    if "Apache-2.0" not in licenses or "NVIDIA-Proprietary-OVRTX" not in licenses:
        raise ImageAuditError(
            "image license label omits a required source/runtime class"
        )
    if str(config.get("User") or "") not in {"ubuntu", "1000", "1000:1000"}:
        raise ImageAuditError("final image user is not the reviewed non-root identity")

    runtime = _container_json(
        image,
        """
import json
from npa.workflows.content_agents import inspect_runtime
print(json.dumps(inspect_runtime(), sort_keys=True))
""",
        runner,
    )
    if runtime.get("status") != "ready":
        raise ImageAuditError("runtime self-inspection did not report ready")

    cli_contract = _container_json(
        image,
        """
import json
import shutil
import subprocess

executable = shutil.which("npa")
completed = subprocess.run(
    [executable, "--version"] if executable else ["/bin/false"],
    check=False,
    capture_output=True,
    text=True,
)
print(json.dumps({
    "cli_contract": {
        "executable_present": executable is not None,
        "version_exit_code": completed.returncode,
        "version_output_present": bool(completed.stdout.strip()),
    }
}, sort_keys=True))
""",
        runner,
    )
    cli_result = cli_contract.get("cli_contract") or {}
    if (
        not cli_result.get("executable_present")
        or cli_result.get("version_exit_code") != 0
        or not cli_result.get("version_output_present")
    ):
        raise ImageAuditError("npa console entry point is not runnable")

    config_parse = _container_json(
        image,
        """
import json
from pathlib import Path
import subprocess
import tempfile

import yaml

from npa.workflows.content_agents import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    _generate_fixture,
    _material_library,
    material_config,
    physics_config,
)

with tempfile.TemporaryDirectory(prefix="npa-content-agents-config-") as raw:
    work = Path(raw)
    source = work / "source.usda"
    _generate_fixture(source)
    _material_library(work)
    configs = {
        "material": material_config(
            input_usd=source,
            output_usd=work / "material.usda",
            work_dir_name=".material",
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
        ),
        "physics": physics_config(
            input_usd=source,
            output_usd=work / "physics.usda",
            work_dir_name=".physics",
            model=DEFAULT_MODEL,
            base_url=DEFAULT_BASE_URL,
        ),
    }
    results = {}
    for name, config in configs.items():
        path = work / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        completed = subprocess.run(
            [f"{name}-agent", "run", str(path), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )
        results[name] = {
            "dry_run_exit_code": completed.returncode,
            "plan_rendered": "Pipeline Execution Plan" in completed.stdout,
        }
    print(json.dumps({"config_parse": results}, sort_keys=True))
""",
        runner,
    )
    config_results = config_parse.get("config_parse") or {}
    for name in ("material", "physics"):
        result = config_results.get(name) or {}
        if result.get("dry_run_exit_code") != 0 or not result.get("plan_rendered"):
            raise ImageAuditError(
                f"{name}-agent rejected the generated config during dry-run"
            )

    inventory = _container_json(
        image,
        """
import json
from pathlib import Path

root = Path('/opt/content-agents')
forbidden_exact = [
    root / '.git',
    root / '.build-resources' / 'scene_optimizer_core',
    root / 'apps' / 'material_agent' / 'data',
    root / 'apps' / 'physics_agent' / 'data',
]
forbidden_dirs = []
for name in ('tests', 'examples'):
    forbidden_dirs.extend(str(path) for path in root.glob(f'apps/*/{name}'))
weight_suffixes = {'.pt', '.ckpt', '.safetensors', '.onnx', '.gguf'}
weight_files = [
    str(path) for path in root.rglob('*')
    if path.is_file() and path.suffix.lower() in weight_suffixes
]
# Python reserves direct ``site-packages/*.pth`` files for import-path hooks
# (for example ``_virtualenv.pth``). A checkpoint beneath a package/model
# directory remains forbidden, as does every other reviewed weight suffix.
weight_files.extend(
    str(path) for path in root.rglob('*.pth')
    if path.is_file() and path.parent.name != 'site-packages'
)
print(json.dumps({
    'forbidden_exact': [str(path) for path in forbidden_exact if path.exists()],
    'forbidden_dirs': forbidden_dirs,
    'weight_files': weight_files,
}, sort_keys=True))
""",
        runner,
    )
    findings = {
        key: values
        for key, values in inventory.items()
        if isinstance(values, list) and values
    }
    if findings:
        raise ImageAuditError(f"forbidden payload found in built image: {findings}")

    image_id = str(record.get("Id") or "")
    if not image_id.startswith("sha256:"):
        raise ImageAuditError("docker inspect returned no immutable image ID")
    repo_digests = sorted(str(value) for value in (record.get("RepoDigests") or []))
    return {
        "schema": "npa.content_agents.image_audit.v1",
        "status": "passed",
        "image_id": image_id,
        "repo_digests": repo_digests,
        "labels": expected_labels,
        "npa_source_revision": source_sha,
        "licenses": licenses,
        "runtime": runtime,
        "cli_contract": cli_contract,
        "config_parse": config_parse,
        "inventory": inventory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", help="Local image tag or immutable private-registry ref"
    )
    parser.add_argument(
        "--output", type=Path, help="Write the private JSON audit record"
    )
    parser.add_argument(
        "--expected-npa-source-sha",
        default="",
        help="Require the image's npa.source_revision label to equal this commit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_npa_source_sha and not re.fullmatch(
        r"[0-9a-f]{40}", args.expected_npa_source_sha
    ):
        raise SystemExit("--expected-npa-source-sha must be a full lowercase Git SHA")
    result = audit_image(
        args.image, expected_npa_source_sha=args.expected_npa_source_sha
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
