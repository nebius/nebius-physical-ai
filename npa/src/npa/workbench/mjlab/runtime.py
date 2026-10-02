"""Stage S3 artifacts and isolate MJLab execution in a simulator subprocess."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from npa.clients.storage import StorageClient
from npa.clients.credentials import load_credentials
from npa.workbench.storage_scope import authorize_uri

from .schemas import EvalRequest, ExportRequest, MJLAB_VERSION, MjlabError, TrainRequest


def train(request: TrainRequest, *, dry_run: bool = False) -> dict:
    """Train a policy and publish checkpoints, configuration and TensorBoard logs.

    Args:
        request: Validated training settings.
        dry_run: Validate and describe execution without I/O or GPU work.
    Returns:
        Training manifest or a non-executed plan.
    Raises:
        MjlabError: Simulator or artifact validation failed.
    """
    return _execute("train", request, dry_run=dry_run)


def evaluate(request: EvalRequest, *, dry_run: bool = False) -> dict:
    """Publish measured episode returns, lengths and survival fraction.

    Args:
        request: Checkpoint and evaluation settings.
        dry_run: Return a plan without metrics, storage I/O or simulation.
    Returns:
        Measured evaluation report or a non-executed plan.
    Raises:
        MjlabError: Execution failed or metrics are invalid.
    """
    return _execute("eval", request, dry_run=dry_run)


def export(request: ExportRequest, *, dry_run: bool = False) -> dict:
    """Publish an ONNX policy exported by the upstream runner.

    Args:
        request: Native checkpoint and task settings.
        dry_run: Return a plan without execution.
    Returns:
        Export manifest containing artifact hashes.
    Raises:
        MjlabError: Export or validation failed.
    """
    return _execute("export", request, dry_run=dry_run)


def system_info() -> dict:
    """Report installed dependency versions without importing GPU libraries.

    Args:
        None.
    Returns:
        Version inventory and installation readiness (not GPU health).
    Raises:
        None.
    """
    versions = {}
    for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "backend": "mjlab",
        "versions": versions,
        "required_version": MJLAB_VERSION,
        "installed": versions["mjlab"] == MJLAB_VERSION,
    }


def list_tasks() -> dict:
    """Query the installed upstream task registry in an isolated process.

    Args:
        None.
    Returns:
        Real registered task IDs.
    Raises:
        MjlabError: MJLab is absent or registry discovery failed.
    """
    with tempfile.TemporaryDirectory(prefix="npa-mjlab-list-") as directory:
        root = Path(directory)
        _worker("list", {}, root)
        return json.loads((root / "result.json").read_text())


def _execute(operation, request, *, dry_run):
    values = request.model_dump()
    for field in ("output_path", "input_path", "checkpoint"):
        if values.get(field):
            authorize_uri(values[field], operation=f"mjlab {operation} {field}")
    if dry_run:
        return {
            "status": "planned",
            "executed": False,
            "operation": operation,
            "request": values,
            "required_version": MJLAB_VERSION,
        }
    if not system_info()["installed"]:
        raise MjlabError(
            f"MJLab {MJLAB_VERSION} is required; use the MJLab image or npa[mjlab]"
        )
    client = _storage_client()
    with tempfile.TemporaryDirectory(prefix="npa-mjlab-") as directory:
        root = Path(directory)
        inputs = _stage_inputs(client, values, root)
        _worker(operation, {"request": values, "inputs": inputs}, root)
        report = json.loads((root / "result.json").read_text())
        return _publish(client, operation, request, root, report, inputs)


def _stage_inputs(client, values, root):
    inputs = {}
    for field, filename in (
        ("input_path", "motion.npz"),
        ("checkpoint", "checkpoint.pt"),
    ):
        if values.get(field):
            path = root / "inputs" / filename
            path.parent.mkdir(exist_ok=True)
            client.download_file(values[field], str(path))
            inputs[field] = {"path": str(path), "sha256": _sha256(path)}
    return inputs


def _worker(operation, payload, root):
    recipe = root / "request.json"
    recipe.write_text(json.dumps(payload))
    env = dict(os.environ, TORCH_FORCE_WEIGHTS_ONLY_LOAD="1", WANDB_MODE="disabled")
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    if sys.platform == "linux":
        env.setdefault("MUJOCO_GL", "egl")
    command = [
        sys.executable,
        "-m",
        "npa.workbench.mjlab.worker",
        operation,
        str(recipe),
    ]
    with (root / "worker.log").open("w") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        tail = (root / "worker.log").read_text(errors="replace").splitlines()[-1:]
        raise MjlabError(
            f"MJLab {operation} failed (exit {result.returncode}): {' '.join(tail)}"
        )


def _publish(client, operation, request, root, report, inputs):
    outputs = root / "outputs"
    files = sorted(path for path in outputs.rglob("*") if path.is_file())
    if not files:
        raise MjlabError("MJLab produced no artifacts")
    artifacts = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(outputs.resolve()):
            raise MjlabError("MJLab output escapes its staging directory")
        name = path.relative_to(outputs).as_posix()
        uri = request.output_path.rstrip("/") + "/" + name
        artifacts[name] = {
            "uri": uri,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        client.upload_file(str(path), uri)
    report.update(
        backend="mjlab",
        operation=operation,
        executed=True,
        task=request.task,
        versions=system_info()["versions"],
        artifacts=artifacts,
        input_sha256={key: value["sha256"] for key, value in inputs.items()},
    )
    report["schema_version"] = f"npa.mjlab.{operation}.v1"
    report["result_uri"] = result_uri_for(request.output_path, operation=operation)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    client.upload_file(str(manifest), report["result_uri"])
    return report


def _sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def result_uri_for(output_path: str, *, operation: str = "eval") -> str:
    """Resolve the manifest written last after artifact publication.

    Args:
        output_path: S3 output prefix.
        operation: train, eval or export.
    Returns:
        The manifest S3 URI.
    Raises:
        ValueError: Unknown operation.
    """
    if operation not in {"train", "eval", "export"}:
        raise ValueError("Unknown MJLab operation")
    return output_path.rstrip("/") + f"/mjlab_{operation}.json"


def _storage_client():
    credentials = load_credentials()
    return StorageClient.from_environment(
        endpoint_url=credentials.s3_endpoint,
        aws_access_key_id=credentials.s3_access_key_id,
        aws_secret_access_key=credentials.s3_secret_access_key,
    )
