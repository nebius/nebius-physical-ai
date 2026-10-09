"""Share stateless Marble operations between the CLI, SDK, and native workflow stages."""

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri

from .acquisition import acquire_bundle
from .api import MarbleError


def _paths(request):
    validate_write_path(request.output_path, tool="marble", required=True)
    if hasattr(request, "input_path"):
        validate_read_path(request.input_path, tool="marble", allow_hf=False)


def _materialize(uri, manifest_name, root):
    payload = read_bytes_uri(uri_join(uri, manifest_name))
    manifest = json.loads(payload)
    (root / manifest_name).write_bytes(payload)
    for name, record in manifest["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise MarbleError("Manifest asset path escapes the bundle")
        data = read_bytes_uri(uri_join(uri, name))
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise MarbleError("Input artifact SHA-256 mismatch")
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def _publish(root, uri):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            target = uri_join(uri, path.relative_to(root).as_posix())
            payload = path.read_bytes()
            write_bytes_uri(target, payload)
            if (
                hashlib.sha256(read_bytes_uri(target)).digest()
                != hashlib.sha256(payload).digest()
            ):
                raise MarbleError("Output artifact read-after-write hash mismatch")


def acquire(request):
    """Acquire a world and its assets through the common Workbench S3 contract.

    Args: AcquireRequest selecting generation or an attributed example.
    Returns: The resulting world manifest.
    Raises: MarbleError or ValueError on provider, storage, or path failure.
    """
    _paths(request)
    return acquire_bundle(request)


def _process(request, kind):
    from .gpu import process_world

    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-marble-") as directory:
        root = Path(directory)
        _materialize(request.input_path, "world.json", root)
        report = process_world(root, request, kind)
        _publish(root, request.output_path)
    return {
        "kind": kind,
        "run_id": request.run_id,
        "frames": report["frames"],
        "gpu": report["gpu"],
    }


def capture(request):
    """Render Gaussian splats into an explicit camera dataset on CUDA.

    Args: RunRequest pointing to a world bundle.
    Returns: A compact measured execution summary.
    Raises: MarbleError when CUDA, source assets, or output verification fail.
    """
    return _process(request, "capture")


def scan(request):
    """Raycast the world's actual collision geometry on CUDA.

    Args: RunRequest pointing to a world bundle.
    Returns: A compact measured execution summary.
    Raises: MarbleError when CUDA, source assets, or output verification fail.
    """
    return _process(request, "scan")


def report(request):
    """Publish a portable interactive HTML report only from verified GPU artifacts.

    Args: RunRequest pointing to a completed GPU result bundle.
    Returns: Report artifact names and run identity.
    Raises: MarbleError for invalid or unverifiable GPU evidence.
    """
    from .demo import write_demo

    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-marble-report-") as directory:
        root = Path(directory)
        result = _materialize(request.input_path, "result.json", root)
        validate_result(result, request.run_id)
        if result.get("kind") == "rover":
            from .rover_report import write_rover_report

            write_rover_report(root, result)
        else:
            write_demo(root, result)
        _publish(root, request.output_path)
    return {"run_id": request.run_id, "report": "index.html", "verified": True}


def validate_result(result, run_id):
    """Reject incomplete GPU evidence or mismatched run identities.

    Args: Decoded GPU result and expected run identity.
    Returns: None.
    Raises: MarbleError when provenance, frames, or timings are incomplete.
    """
    import math

    timings = result.get("metrics", {}).get("cuda_frame_ms", [])
    count = result.get("frames", 0)
    if result.get("run_id") != run_id or not result.get("gpu", {}).get("name"):
        raise MarbleError("Missing GPU device or mismatched run identity")
    if (
        count <= 0
        or len(timings) != count
        or any(not math.isfinite(t) or t <= 0 for t in timings)
    ):
        raise MarbleError("Every frame must have a positive measured CUDA duration")
    if any(f"frames/{i:04d}.jpg" not in result.get("files", {}) for i in range(count)):
        raise MarbleError("GPU report is missing rendered frames")
