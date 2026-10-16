"""Reload and attest one exact native SmolVLA interchange checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from npa.workbench.lerobot.policy_container import (
    INTERCHANGE_MANIFEST_FILENAME,
    LeRobotPolicyRuntime,
)
from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
    read_json,
    storage,
    write_json,
)


SCHEMA = "npa.sim2real.smolvla_reload_validation.v1"
HF_SOURCE_RE = re.compile(r"^hf://(?P<repo>[^@\s]+)@(?P<revision>[0-9a-f]{40})$")


class SmolVLAReloadError(RuntimeError):
    """Raised when exact checkpoint materialization or binding fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_checkpoint(source: str, destination: Path) -> Path:
    """Materialize only immutable private-S3 or exact-revision HF sources."""

    if destination.exists():
        raise SmolVLAReloadError("checkpoint destination must not already exist")
    if source.startswith("s3://"):
        storage().download_directory(source.rstrip("/") + "/", str(destination))
        if not destination.is_dir():
            raise SmolVLAReloadError("private checkpoint download is absent")
        return destination
    match = HF_SOURCE_RE.fullmatch(source)
    if not match:
        raise SmolVLAReloadError(
            "checkpoint must be a private s3:// prefix or hf://repo@40-hex-revision"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SmolVLAReloadError("huggingface_hub is required") from exc
    snapshot_download(
        repo_id=match.group("repo"),
        revision=match.group("revision"),
        local_dir=destination,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return destination


def build_reload_receipt(
    *,
    runtime: LeRobotPolicyRuntime,
    preparation: dict[str, Any],
    checkpoint_source: str,
    hardware_topology: str,
) -> dict[str, Any]:
    """Bind the loaded native objects back to the pre-submit contract."""
    manifest = runtime.manifest
    expected = _expected_binding(preparation)
    actual = {key: manifest.get(key) for key in expected}
    if not _stable_policy_binding(manifest, actual, expected, preparation):
        raise SmolVLAReloadError(
            "loaded SmolVLA checkpoint differs from the prepared policy contract"
        )
    manifest_path = runtime.checkpoint_path / INTERCHANGE_MANIFEST_FILENAME
    return {
        "schema": SCHEMA,
        "status": "native_checkpoint_reloaded",
        "checkpoint_source_sha256": hashlib.sha256(
            checkpoint_source.encode("utf-8")
        ).hexdigest(),
        "hardware_topology": hardware_topology,
        "device": runtime.device,
        "native_policy_class": type(runtime.policy).__qualname__,
        "native_preprocessor_class": type(runtime.preprocessor).__qualname__,
        "native_postprocessor_class": type(runtime.postprocessor).__qualname__,
        "checkpoint_validation": runtime.validation.to_dict(),
        "interchange": {
            **actual,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "normalization_identity": manifest["normalization_identity"],
            "manifest_sha256": _sha256(manifest_path),
            "normalization_files": manifest["normalization_files"],
            "action": manifest["action"],
            "measured_state_names": manifest["measured_state_names"],
            "cameras": manifest["cameras"],
        },
        "inference_or_task_success_measured": False,
    }


def _expected_binding(preparation: dict[str, Any]) -> dict[str, Any]:
    """Extract immutable policy identities from preparation."""

    compatibility = preparation.get("policy_compatibility") or {}
    return {
        "backend": compatibility.get("backend"),
        "policy_identity": compatibility.get("policy_identity"),
        "robot_identity": compatibility.get("robot_identity"),
        "task_identity": compatibility.get("task_identity"),
    }


def _stable_policy_binding(
    manifest: dict[str, Any],
    actual: dict[str, Any],
    expected: dict[str, Any],
    preparation: dict[str, Any],
) -> bool:
    """Compare native processors and action metadata to preparation."""

    policy = preparation.get("policy") or {}
    return bool(
        actual == expected
        and manifest.get("action") == policy.get("action")
        and manifest.get("measured_state_names") == policy.get("measured_state_names")
        and [row.get("key") for row in manifest.get("cameras") or []]
        == policy.get("camera_keys")
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Reload the exact native checkpoint and validate held-out actions."""

    with tempfile.TemporaryDirectory(prefix="npa-smolvla-reload-") as directory:
        root = Path(directory)
        preparation = read_json(args.preparation_uri, directory=root / "preparation")
        checkpoint = materialize_checkpoint(args.checkpoint_uri, root / "checkpoint")
        runtime = LeRobotPolicyRuntime(checkpoint)
        result = build_reload_receipt(
            runtime=runtime,
            preparation=preparation,
            checkpoint_source=args.checkpoint_uri,
            hardware_topology=args.hardware_topology,
        )
        write_json(args.output_uri, result, directory=root / "output")
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=9,
        name="smolvla_native_checkpoint_reload",
        tier="WORKS",
        require_gpu=True,
        evidence=(
            "Loaded exact native SmolVLA weights and checkpoint-local processors; "
            "closed-loop task outcome remains a separate stage."
        ),
        artifacts={"reload": args.output_uri, "status": result["status"]},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the native SmolVLA reload-stage argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--preparation-uri", required=True)
    parser.add_argument("--hardware-topology", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute checkpoint reload validation and publish its receipt."""

    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
