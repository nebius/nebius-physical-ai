"""Execute prebuilt BYOF capabilities inside an allocated workflow worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.orchestration.npa_workflow.run_resolution import validate_run_id


@dataclass(frozen=True)
class WorkerSmoke:
    """Describe a capability to execute in the current immutable image.

    Args:
        run_id: Workflow run identifier.
        image: Exact digest selected by the workflow scheduler.
        repo_url: Expected source repository, retained in image metadata.
        repo_ref: Expected source revision, retained in image metadata.
        output_root: S3 root under which the run writes its artifacts.
        command: Real capability shell command supplied by the workflow.
        solution: Registered solution label.
        capability: Named capability exercised by the command.
        artifact_name: Required nonempty output file, relative to the output directory.
        repo_root: Image's baked source directory.

    Returns:
        None.
    Raises:
        None.
    """

    run_id: str
    image: str
    repo_url: str
    repo_ref: str
    output_root: str
    command: str
    solution: str
    capability: str
    artifact_name: str
    repo_root: Path = Path("/opt/byof")


def in_workflow_worker() -> bool:
    """Identify the context injected by the standard workflow renderer.

    Args:
        None.
    Returns:
        Whether any workflow execution marker is present.
    Raises:
        None.
    """
    return any(
        os.environ.get(name) for name in ("NPA_WORKFLOW_RUN_ID", "NPA_WORKFLOW_STATE")
    )


def _validate_request(request: WorkerSmoke) -> str:
    validate_run_id(request.run_id)
    if not os.environ.get("NPA_WORKFLOW_RUN_ID"):
        raise ValueError("BYOF worker requires the workflow run marker")
    if not os.environ.get("NPA_WORKFLOW_STATE"):
        raise ValueError("BYOF worker requires the workflow state marker")
    image = request.image.removeprefix("docker:")
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("BYOF worker requires a digest-pinned prebuilt image")
    if os.environ.get("NPA_TASK_IMAGE", "").removeprefix("docker:") != image:
        raise ValueError(
            "BYOF prebuilt image does not match the allocated worker image"
        )
    if not request.command.strip() or not request.capability.strip():
        raise ValueError("BYOF worker requires a smoke command and named capability")
    artifact = PurePosixPath(request.artifact_name)
    if (
        not artifact.parts
        or artifact.is_absolute()
        or artifact.as_posix() != request.artifact_name
        or any(part in {".", ".."} for part in artifact.parts)
        or "\\" in request.artifact_name
    ):
        raise ValueError("BYOF smoke artifact must be a relative output file")
    output = urlparse(request.output_root)
    if (
        output.scheme != "s3"
        or not output.netloc
        or output.query
        or output.fragment
        or output.username
        or output.password
    ):
        raise ValueError("BYOF worker requires an s3:// output root")
    return request.output_root.rstrip("/") + f"/{request.run_id}/"


def _source_metadata(request: WorkerSmoke) -> dict[str, Any]:
    path = request.repo_root / "npa_source_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("BYOF source metadata must be a JSON object")
    if metadata.get("source") == "private-byof":
        expected = {
            "repository_sha256": hashlib.sha256(request.repo_url.encode()).hexdigest(),
            "ref_sha256": hashlib.sha256(request.repo_ref.encode()).hexdigest(),
        }
    else:
        expected = {"repo": request.repo_url, "ref": request.repo_ref}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "BYOF baked source metadata does not match the requested repository/ref"
        )
    return metadata


def _smoke_environment(request: WorkerSmoke, root: Path, prefix: str) -> dict[str, str]:
    return {
        **os.environ,
        "NPA_BYOF_RUN_ID": request.run_id,
        "NPA_SMOKE_OUTPUT_DIR": str(root),
        "BYOF_REPO_ROOT": str(request.repo_root),
        "BYOF_IMAGE": request.image,
        "BYOF_SMOKE_COMMAND": request.command,
        "BYOF_SOLUTION_NAME": request.solution,
        "BYOF_CAPABILITY_NAME": request.capability,
        "BYOF_SMOKE_ARTIFACT_NAME": request.artifact_name,
        "S3_OUTPUT_PREFIX": prefix,
    }


def _execute(request: WorkerSmoke, root: Path, prefix: str) -> int:
    with (root / "solution_smoke_stdout.log").open("w") as stdout:
        with (root / "solution_smoke_stderr.log").open("w") as stderr:
            process = subprocess.run(
                ["/bin/bash", "-lc", request.command],
                cwd=request.repo_root,
                env=_smoke_environment(request, root, prefix),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    return process.returncode


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("BYOF output artifacts must not contain symbolic links")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return artifacts


def _summary(
    request: WorkerSmoke, metadata: dict[str, Any], root: Path, exit_code: int
) -> dict[str, Any]:
    artifact = root / request.artifact_name
    present = artifact.is_file() and artifact.stat().st_size > 0
    return {
        "schema": "npa.byof.worker-smoke.v1",
        "status": "success" if exit_code == 0 and present else "failed",
        "tool": "byof",
        "workload": "solution-smoke",
        "execution": "workflow-worker",
        "run_id": request.run_id,
        "workflow_worker_run_id": os.environ["NPA_WORKFLOW_RUN_ID"],
        "workflow_state": os.environ["NPA_WORKFLOW_STATE"],
        "image": request.image.removeprefix("docker:"),
        "metadata": metadata,
        "solution_name": request.solution,
        "capability_name": request.capability,
        "smoke_command": request.command,
        "smoke_artifact_name": request.artifact_name,
        "smoke_artifact_present": present,
        "smoke_exit_code": exit_code,
        "artifacts": _artifact_inventory(root),
    }


def _publish_artifacts(storage: StorageClient, root: Path, prefix: str) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: (
        path.name == "npa_byof_summary.json", path.relative_to(root).as_posix()
    ))
    for path in paths:
        if path.is_symlink():
            raise ValueError("BYOF output artifacts must not contain symbolic links")
        if not path.is_file():
            continue
        uri = prefix + path.relative_to(root).as_posix()
        payload = path.read_bytes()
        try:
            storage.put_bytes_conditional(payload, uri, if_none_match=True)
        except StoragePreconditionFailed:
            existing = storage.read_bytes_with_etag(uri)
            if existing is None or existing[0] != payload:
                raise RuntimeError(
                    "BYOF artifact already exists with different bytes; use a new run id"
                ) from None


def run_prebuilt_smoke(request: WorkerSmoke) -> dict[str, Any]:
    """Execute and upload a capability without starting another scheduler.

    Args:
        request: Validated workflow capability and its required output contract.
    Returns:
        The successful smoke summary, including immutable artifact hashes.
    Raises:
        ValueError: Worker identity, source metadata, or output paths are invalid.
        RuntimeError: The command failed or did not produce its required artifact.
        OSError: Local evidence could not be read or written.
        StorageError: Object storage configuration or publication failed.
    """
    prefix = _validate_request(request)
    metadata = _source_metadata(request)
    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-byof-smoke-") as temporary:
        root = Path(temporary)
        shutil.copyfile(
            request.repo_root / "npa_source_metadata.json",
            root / "npa_source_metadata.json",
        )
        exit_code = _execute(request, root, prefix)
        summary = _summary(request, metadata, root, exit_code)
        (root / "npa_byof_summary.json").write_text(json.dumps(summary, indent=2))
        _publish_artifacts(storage, root, prefix)
    if exit_code != 0:
        raise RuntimeError(
            f"BYOF smoke command failed with exit code {exit_code}; diagnostics uploaded"
        )
    if not summary["smoke_artifact_present"]:
        raise RuntimeError(
            "BYOF smoke command did not create its required artifact; diagnostics uploaded"
        )
    return summary
