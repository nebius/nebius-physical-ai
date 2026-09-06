"""Verified S3 augmentation client and recovery without repeated GPU requests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import httpx

from . import nano_video as video
from . import nano_video_augment as core


class AugmentationClientError(video.NanoVideoError):
    """An augmentation cannot be submitted, verified or safely recovered."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _digest(path: Path) -> dict[str, Any]:
    return video.artifact(path)


def _s3(value: str) -> tuple[str, str]:
    parsed = urlsplit(value)
    if (parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/")
            or parsed.query or parsed.fragment or parsed.username or parsed.port):
        raise ValueError("handoff paths must name a bucket and nonempty S3 object or prefix")
    return parsed.netloc, parsed.path.strip("/")


def _connection(endpoint: str, token_env: str) -> tuple[str, str]:
    endpoint = (endpoint or os.environ.get("NPA_COSMOS3_VIDEO_ENDPOINT", "")).rstrip("/")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("a serving endpoint without embedded credentials is required")
    token = os.environ.get(token_env, "")
    if re.fullmatch(r"[!-~]+", token) is None:
        raise ValueError("a nonempty serving API token is required")
    return endpoint, token


def _http(token: str) -> httpx.Client:
    return httpx.Client(timeout=None, trust_env=False, follow_redirects=False,
                        headers={"Authorization": f"Bearer {token}"})


def _private_root(output_path: str, *, fresh: bool) -> Path:
    recovery = Path(os.environ.get("NPA_COSMOS3_VIDEO_RECOVERY_DIR", tempfile.gettempdir()))
    recovery.mkdir(parents=True, exist_ok=True)
    root = recovery / ("npa-nano-augment-" + hashlib.sha256(output_path.encode()).hexdigest()[:24])
    if root.is_symlink():
        raise AugmentationClientError("recovery directory must not be a symlink")
    root.mkdir(mode=0o700, exist_ok=not fresh)
    return root


def _safe_file(root: Path, name: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
        raise AugmentationClientError("unsafe artifact path")
    target = root / name
    if root.is_symlink() or target.is_symlink() or target.resolve().parent != root.resolve():
        raise AugmentationClientError("artifact path is not contained in recovery storage")
    return target


def _matches(path: Path, item: dict[str, Any]) -> bool:
    observed = _digest(path)
    return all(observed[key] == item[key] for key in ("bytes", "sha256"))


def _read_object(storage: Any, bucket: str, key: str) -> bytes:
    response = storage.s3.get_object(Bucket=bucket, Key=key)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


def _put_verified(storage: Any, bucket: str, key: str, data: bytes) -> dict[str, Any]:
    """Conditional write; accept an existing object only after exact readback."""
    try:
        storage.s3.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch="*")
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}:
            raise
    actual = _read_object(storage, bucket, key)
    digest = hashlib.sha256(data).hexdigest()
    if len(actual) != len(data) or hashlib.sha256(actual).hexdigest() != digest:
        raise AugmentationClientError("immutable object differs from expected bytes")
    return {"bytes": len(data), "sha256": digest}


def _probe(path: Path, *, width: int = 832, expected_frames: int | None = None) -> dict[str, Any]:
    """Run the shared strict decoder locally, independently of service claims."""
    try:
        result = core.validate_media(path, frames=expected_frames, width=width)
        if expected_frames is None and result["decoded_frames"] < 6:
            raise video.NanoVideoError("source must contain at least six frames")
        return result
    except video.NanoVideoError as exc:
        raise AugmentationClientError("video dimensions, frames, fps, timestamps or decode differ") from exc


def _download_result(client: httpx.Client, endpoint: str, root: Path,
                     report: dict[str, Any], request: dict[str, Any]) -> None:
    core.validate_report(report, request)
    root.mkdir(mode=0o700, exist_ok=True)
    for name, item in core.artifact_manifest(report).items():
        path = _safe_file(root, name)
        if path.exists():
            if not _matches(path, item):
                raise AugmentationClientError("retained artifact conflicts with service manifest")
            continue
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".download-", delete=False) as stream:
                temporary = Path(stream.name)
                digest, count = hashlib.sha256(), 0
                with client.stream("GET", f"{endpoint}/artifacts/{request['request_id']}/{name}") as response:
                    response.raise_for_status()
                    for block in response.iter_bytes():
                        stream.write(block)
                        digest.update(block)
                        count += len(block)
                stream.flush()
                os.fsync(stream.fileno())
            if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise AugmentationClientError("downloaded artifact hash or length differs")
            # Link is exclusive: a concurrent recovery cannot overwrite an artifact.
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or not _matches(path, item):
                    raise AugmentationClientError("concurrent artifact download conflict") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    validation = _probe(root / "input.mp4", expected_frames=report["source"]["video"]["decoded_frames"])
    frames = validation["decoded_frames"]
    _probe(root / "augmented.mp4", expected_frames=frames)
    _probe(root / "comparison.mp4", width=1664, expected_frames=frames)
    if _digest(root / "input.mp4")["sha256"] != request["source_sha256"]:
        raise AugmentationClientError("downloaded input is not the submitted source")
    if _digest(root / "augmented.mp4")["sha256"] == request["source_sha256"]:
        raise AugmentationClientError("augmentation output is byte-identical to source")
    for chunk in report["chunks"]:
        _probe(root / chunk["output_path"], expected_frames=chunk["source_frames"])
        _probe(root / chunk["control_path"], expected_frames=chunk["source_frames"])
        if chunk.get("reference_path"):
            _probe(root / chunk["reference_path"], expected_frames=5)
    video.write_json(root.parent / "client-validation.json", {
        "technical_validation_passed": True, "source": validation,
        "source_sha256": request["source_sha256"],
        "output_sha256": _digest(root / "augmented.mp4")["sha256"],
        "request_sha256": core.request_sha256(request),
        "report_sha256": hashlib.sha256(_json_bytes(report)).hexdigest(),
        "quality_review_status": "pending", "meaningful_augmentation_proven_by_hash": False,
    })


def _summary(report: dict[str, Any], *, publication_verified: bool) -> dict[str, Any]:
    return {"status": "succeeded" if publication_verified else "pending",
            "generation_status": report["status"], "request_id": report["request_id"],
            "technical_validation_passed": True, "publication_verified": publication_verified,
            "source_sha256": report["source"]["sha256"],
            "output_sha256": report["output"]["sha256"],
            "video": report["output"]["video"], "chunks": len(report["chunks"]),
            "total_wall_seconds": report["total_wall_seconds"],
            "device_peak_used_mib": report["device_peak_used_mib"],
            "quality_review_status": "pending"}


def _publish(storage: Any, output_path: str, root: Path,
             report: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    core.validate_report(report, request)
    bucket, prefix = _s3(output_path)
    objects = []
    try:
        validation = json.loads(_safe_file(root, "client-validation.json").read_text())
        required = {"technical_validation_passed": True, "source_sha256": request["source_sha256"],
                    "output_sha256": report["output"]["sha256"],
                    "request_sha256": core.request_sha256(request),
                    "report_sha256": hashlib.sha256(_json_bytes(report)).hexdigest()}
        if any(validation.get(key) != value for key, value in required.items()):
            raise AugmentationClientError("local decode proof does not match publication artifacts")
        for name, item in core.artifact_manifest(report).items():
            path = _safe_file(root / "artifacts", name)
            if not path.is_file() or not _matches(path, item):
                raise AugmentationClientError("publication requires every verified artifact")
            proof = _put_verified(storage, bucket, f"{prefix}/{name}", path.read_bytes())
            objects.append({"path": name, **proof})
        for name in ("result.json", "client-validation.json"):
            data = _safe_file(root, name).read_bytes()
            objects.append({"path": name, **_put_verified(storage, bucket, f"{prefix}/{name}", data)})
        manifest = {"schema_version": "npa.cosmos3.nano-video.augmentation-publication.v1",
                    "request_id": request["request_id"], "request_sha256": core.request_sha256(request),
                    "verified": True, "objects": sorted(objects, key=lambda item: item["path"])}
        proof = _put_verified(storage, bucket, f"{prefix}/publication.json", _json_bytes(manifest))
        video.write_json(root / "publication.json", {**manifest, "readback": proof})
        return _summary(report, publication_verified=True)
    except Exception as exc:
        video.write_json(root / "publication-pending.json", {
            "status": "pending", "verified": False, "error_type": type(exc).__name__,
            "verified_objects": objects, "generation_will_not_be_repeated": True,
        })
        raise


def _recover_report(client: httpx.Client, endpoint: str, request: dict[str, Any]) -> dict[str, Any]:
    response = client.get(f"{endpoint}/result", params={"request_id": request["request_id"]})
    if response.status_code in {202, 404}:
        return {"status": "pending", "generation_status": "unknown" if response.status_code == 404 else "running",
                "request_id": request["request_id"], "publication_verified": False}
    response.raise_for_status()
    report = response.json()
    if (not isinstance(report, dict) or report.get("request_id") != request["request_id"]
            or report.get("request_sha256") != core.request_sha256(request)):
        raise AugmentationClientError("recovery response does not match request identity")
    if report.get("status") != "succeeded":
        if report.get("status") != "failed":
            raise AugmentationClientError("unexpected terminal recovery status")
        error_type = report.get("error_type", "GenerationFailed")
        if not isinstance(error_type, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", error_type) is None:
            error_type = "GenerationFailed"
        return {"status": "failed", "generation_status": "failed",
                "request_id": request["request_id"], "publication_verified": False,
                "error_type": error_type}
    core.validate_report(report, request)
    return report


def run_augmentation(*, endpoint: str, output_dir: Path, input_video: Path,
                     request: dict[str, Any], token: str) -> dict[str, Any]:
    """Submit exactly once; transport recovery only reads the immutable request."""
    request = core.validate_request(request)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if input_video.is_symlink() or not _matches(input_video, {
            "sha256": request["source_sha256"], "bytes": request["source_bytes"]}):
        raise AugmentationClientError("input video does not match request")
    _probe(input_video)
    state = _safe_file(output_dir, "submission-attempted.json")
    if state.exists():
        raise AugmentationClientError("submission already attempted; recover without resubmitting")
    # O_EXCL protects against a concurrent local caller before any HTTP POST.
    with state.open("xb") as stream:
        os.chmod(state, 0o600)
        stream.write(_json_bytes({"request_id": request["request_id"],
                                  "request_sha256": core.request_sha256(request),
                                  "started_at": video.utc_now()}))
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    with _http(token) as client:
        try:
            with input_video.open("rb") as source:
                response = client.post(endpoint + "/run", files={
                    "request": (None, json.dumps(request, allow_nan=False), "application/json"),
                    "input_reference": ("input.mp4", source, "video/mp4"),
                })
            response.raise_for_status()
            report = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            video.write_json(output_dir / "transport-failure.json", {
                "error_type": type(exc).__name__, "request_id": request["request_id"],
                "generation_will_not_be_repeated": True,
            })
            report = _recover_report(client, endpoint, request)
        if report.get("status") != "succeeded":
            video.write_json(output_dir / "generation-state.json", report)
            return report
        core.validate_report(report, request)
        video.write_json(output_dir / "result.json", report)
        _download_result(client, endpoint, output_dir / "artifacts", report, request)
    video.write_json(output_dir / "client-timing.json", {"total_wall_seconds": time.monotonic() - started})
    return report


def submit_augmentation(*, input_path: str, output_path: str, prompt: str, seed: int = 0,
                        negative_prompt: str = "", system_prompt: str = "",
                        num_inference_steps: int = 35, guidance_scale: float = 3.0,
                        flow_shift: float = 10.0, control_guidance: float = 1.5,
                        edge_threshold: str = "medium", chunk_frames: int = 121,
                        max_sequence_length: int = 4096, endpoint: str = "",
                        token_env: str = "NPA_COSMOS3_VIDEO_TOKEN", storage_client: Any = None) -> dict[str, Any]:
    """Upload a complete S3 source, structurally augment it and verify publication."""
    _s3(input_path)
    bucket, prefix = _s3(output_path)
    endpoint, token = _connection(endpoint, token_env)
    request = {"mode": "augmentation", "request_id": "augment-" + uuid.uuid4().hex,
               "prompt": prompt, "seed": seed, "source_sha256": "0" * 64, "source_bytes": 1,
               "negative_prompt": negative_prompt, "num_inference_steps": num_inference_steps,
               "guidance_scale": guidance_scale, "flow_shift": flow_shift,
               "control_guidance": control_guidance, "edge_threshold": edge_threshold,
               "chunk_frames": chunk_frames, "max_sequence_length": max_sequence_length}
    if system_prompt:
        request["system_prompt"] = system_prompt
    request = core.validate_request(request)
    from npa.clients.storage import StorageClient

    storage = storage_client or StorageClient.from_environment()
    existing = storage.s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1)
    if existing.get("KeyCount", 0) or existing.get("Contents"):
        raise AugmentationClientError("output prefix exists; recover rather than generate again")
    root = _private_root(output_path, fresh=True)
    source = root / "input.mp4"
    storage.download_file(input_path, str(source))
    source.chmod(0o400)
    metadata = _probe(source)
    digest = _digest(source)
    request.update(source_sha256=digest["sha256"], source_bytes=digest["bytes"])
    request = core.validate_request(request)
    reservation = {"schema_version": "npa.cosmos3.nano-video.augmentation-reservation.v1",
                   "request": request, "request_sha256": core.request_sha256(request),
                   "source_video": metadata}
    video.write_json(root / "reservation.json", reservation)
    # Reservation must be exclusively new. A competing reservation is never a retry.
    data = _json_bytes(reservation)
    storage.s3.put_object(Bucket=bucket, Key=f"{prefix}/reservation.json", Body=data, IfNoneMatch="*")
    if _read_object(storage, bucket, f"{prefix}/reservation.json") != data:
        raise AugmentationClientError("reservation readback failed before generation")
    _put_verified(storage, bucket, f"{prefix}/input.mp4", source.read_bytes())
    report = run_augmentation(endpoint=endpoint, output_dir=root, input_video=source,
                              request=request, token=token)
    if report.get("status") != "succeeded":
        return report
    return _publish(storage, output_path, root, report, request)


def recover_augmentation(*, output_path: str, endpoint: str = "",
                         token_env: str = "NPA_COSMOS3_VIDEO_TOKEN", storage_client: Any = None) -> dict[str, Any]:
    """Recover an accepted request or retry S3 publication; never submit generation."""
    bucket, prefix = _s3(output_path)
    from npa.clients.storage import StorageClient

    storage = storage_client or StorageClient.from_environment()
    root = _private_root(output_path, fresh=False)
    reservation = json.loads(_read_object(storage, bucket, f"{prefix}/reservation.json"))
    request = core.validate_request(reservation["request"])
    if core.request_sha256(request) != reservation.get("request_sha256"):
        raise AugmentationClientError("reservation request digest differs")
    video.write_json(root / "reservation.json", reservation)
    result = _safe_file(root, "result.json")
    if result.is_file() and (root / "client-validation.json").is_file():
        report = json.loads(result.read_text())
        core.validate_report(report, request)
        return _publish(storage, output_path, root, report, request)
    endpoint, token = _connection(endpoint, token_env)
    with _http(token) as client:
        report = json.loads(result.read_text()) if result.is_file() else _recover_report(client, endpoint, request)
        if report.get("status") != "succeeded":
            video.write_json(root / "generation-state.json", report)
            return report
        core.validate_report(report, request)
        video.write_json(result, report)
        _download_result(client, endpoint, root / "artifacts", report, request)
    return _publish(storage, output_path, root, report, request)
