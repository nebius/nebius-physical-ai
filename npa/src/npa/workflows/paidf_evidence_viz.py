"""Record independently established PAIDF image evidence without copying raw reports.

The caller must derive facts from real run reports, establish the executed image
digest and revisions, verify upstream licensing, and review media for privacy.
This converter checks the normalized contract and exact media bytes; it cannot
establish the truth of a caller's workload or artifact-discovery assertions.
It neither launches workloads nor publishes artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import ipaddress
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

from npa.guardrails.confidentiality import compile_builtin_nebius_infra

APPLICATION_ID = "npa-paidf-image-evidence"
EVIDENCE_SCHEMA = "npa.paidf.image-evidence.v1"
IMAGE_NAMES = frozenset(
    f"npa-paidf-{name}-sky"
    for name in (
        "detection", "captioning", "visual-qa", "attribute-search",
        "image-edit", "event-video", "anomalygen",
    )
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PRIVATE = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|www\.|localhost|"
    r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b|"
    r"\b[a-z][a-z0-9-]*\.(?:[a-z][a-z0-9-]*\.)*[a-z]{2,}\b|"
    r"\b(?:hf_|ghp_|gho_|github_pat_|AKIA|ASIA)[a-z0-9_]+|"
    r"\bbearer\s+\S+|"
    r"\b(?:bearer|password|secret|token|credential|authorization)\s*[:=]\s*\S+|"
    r"\b(?:tenant|project|bucket|registry|cluster|hostname|node|pod|network)"
    r"(?:[_ -]?(?:id|name))?\s*[:=]\s*\S+|"
    r"\beyJ[a-z0-9_-]+\.[a-z0-9_-]+\.[a-z0-9_-]+)"
)
_INFRA = compile_builtin_nebius_infra()
_STAGE_STATUS = {"passed", "failed", "completed", "succeeded", "skipped"}


class PaidfEvidenceError(ValueError):
    """The evidence is incomplete, unsafe, inconsistent, or cannot be decoded."""


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or not required <= value.keys():
        raise PaidfEvidenceError("evidence object is missing required fields")
    if value.keys() - required - (optional or set()):
        raise PaidfEvidenceError("evidence contains unsupported fields; normalize raw reports first")


def _safe_text(value: Any, *, name: bool = False, run_id: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise PaidfEvidenceError("evidence text must be a nonempty sanitized string")
    if _PRIVATE.search(value) or _INFRA.search(value) or _contains_ipv6(value):
        raise PaidfEvidenceError("evidence contains private or secret-shaped provenance")
    if name or run_id:
        if not (_RUN_ID if run_id else _NAME).fullmatch(value):
            raise PaidfEvidenceError("evidence identifier is not a safe entity name")
    elif not re.fullmatch(r"[A-Za-z0-9 .,;:_+()'/-]+", value):
        raise PaidfEvidenceError("evidence text contains unsupported characters")
    return value


def _contains_ipv6(value: str) -> bool:
    # Parse candidate literals so factual times/decimals are not mistaken for IPs.
    for candidate in re.findall(r"(?<![A-Za-z0-9])[0-9A-Fa-f:.]+", value):
        if ":" not in candidate:
            continue
        try:
            ipaddress.IPv6Address(candidate.rstrip("."))
        except ValueError:
            continue
        return True
    return False


def _finite(value: Any, *, nonnegative: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PaidfEvidenceError("evidence scalars must be finite numbers")
    if nonnegative and value < 0:
        raise PaidfEvidenceError("evidence duration must be nonnegative")


def _revision(value: Any) -> None:
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise PaidfEvidenceError("source revisions must be full lowercase Git SHAs")


def _validate(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _keys(evidence, {
        "schema", "image_name", "image_digest", "run_id", "source_revisions",
        "image_build_source_revision", "runtime_source_revisions",
        "upstream_sources", "validation", "stages", "source_artifacts",
    }, {"gpu", "limitations"})
    if evidence["schema"] != EVIDENCE_SCHEMA or evidence["image_name"] not in IMAGE_NAMES:
        raise PaidfEvidenceError("unsupported PAIDF evidence schema or image")
    digest = evidence["image_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise PaidfEvidenceError("image_digest must be an immutable sha256 digest")
    _safe_text(evidence["run_id"], run_id=True)
    revisions = evidence["source_revisions"]
    if not isinstance(revisions, list) or not revisions:
        raise PaidfEvidenceError("source_revisions must be nonempty")
    for revision in revisions:
        _revision(revision)
    _revision(evidence["image_build_source_revision"])
    runtime_revisions = evidence["runtime_source_revisions"]
    if not isinstance(runtime_revisions, list) or not runtime_revisions:
        raise PaidfEvidenceError("runtime_source_revisions must be nonempty")
    for revision in runtime_revisions:
        _revision(revision)
    if set(revisions) != {evidence["image_build_source_revision"], *runtime_revisions}:
        raise PaidfEvidenceError("source_revisions must be the exact build and runtime revision union")
    if len(set(revisions)) != len(revisions) or len(set(runtime_revisions)) != len(runtime_revisions):
        raise PaidfEvidenceError("source revision lists must not contain duplicates")
    sources = evidence["upstream_sources"]
    if not isinstance(sources, list) or not sources:
        raise PaidfEvidenceError("upstream attribution is required")
    for source in sources:
        _keys(source, {"repository", "revision", "license", "adaptation"})
        repository = source["repository"]
        if not isinstance(repository, str):
            raise PaidfEvidenceError("upstream repository must be a public repository URL")
        parts = urlsplit(repository)
        if (parts.scheme != "https" or parts.netloc != "github.com"
                or parts.query or parts.fragment
                or not re.fullmatch(r"/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", parts.path)
                or _INFRA.search(parts.path)
                or _PRIVATE.search(parts.path)):
            raise PaidfEvidenceError("upstream repository must be a public GitHub repository URL")
        _revision(source["revision"])
        _safe_text(source["license"])
        _safe_text(source["adaptation"])
    if not any(urlsplit(source["repository"]).path.split("/")[1].lower()
               in {"nvidia", "nvidia-ai-blueprints", "nvlabs", "nvidia-cosmos"}
               for source in sources):
        raise PaidfEvidenceError("NVIDIA upstream attribution is required")
    validation = evidence["validation"]
    _keys(validation, {"status", "checks"})
    if validation["status"] not in {"passed", "failed"}:
        raise PaidfEvidenceError("validation must state passed or failed")
    if not isinstance(validation["checks"], list) or not validation["checks"]:
        raise PaidfEvidenceError("validation checks must be nonempty")
    for check in validation["checks"]:
        _keys(check, {"name", "status"})
        _safe_text(check["name"], name=True)
        if check["status"] not in {"passed", "failed"}:
            raise PaidfEvidenceError("validation checks must state passed or failed")
    if (validation["status"] == "passed") != all(
        check["status"] == "passed" for check in validation["checks"]
    ):
        raise PaidfEvidenceError("validation result contradicts its checks")
    stages = evidence["stages"]
    if not isinstance(stages, list) or not stages:
        raise PaidfEvidenceError("factual stages are required")
    if not any(stage.get("status") != "skipped" for stage in stages if isinstance(stage, dict)):
        raise PaidfEvidenceError("at least one executed factual stage is required")
    for stage in stages:
        _keys(stage, {"state", "source_revision", "status", "duration_seconds", "metrics"})
        _safe_text(stage["state"], name=True)
        _revision(stage["source_revision"])
        if stage["source_revision"] not in runtime_revisions:
            raise PaidfEvidenceError("stage revision is absent from runtime_source_revisions")
        if stage["status"] not in _STAGE_STATUS:
            raise PaidfEvidenceError("unsupported stage status")
        _finite(stage["duration_seconds"], nonnegative=True)
        if not isinstance(stage["metrics"], dict):
            raise PaidfEvidenceError("stage metrics must be a scalar mapping")
        if "duration_seconds" in stage["metrics"]:
            raise PaidfEvidenceError("metrics must not override the measured stage duration")
        for name, value in stage["metrics"].items():
            _safe_text(name, name=True)
            _finite(value)
    if "gpu" in evidence:
        _keys(evidence["gpu"], {"model", "count"})
        _safe_text(evidence["gpu"]["model"])
        if type(evidence["gpu"]["count"]) is not int or evidence["gpu"]["count"] < 0:
            raise PaidfEvidenceError("GPU count must be a nonnegative integer")
    if not isinstance(evidence.get("limitations", []), list):
        raise PaidfEvidenceError("limitations must be a list")
    for limitation in evidence.get("limitations", []):
        _safe_text(limitation)
    if not isinstance(evidence["source_artifacts"], list) or not evidence["source_artifacts"]:
        raise PaidfEvidenceError("hashed source artifacts are required")
    artifacts = {}
    for artifact in evidence["source_artifacts"]:
        _keys(artifact, {"role", "sha256", "size_bytes"}, {"media_type"})
        role = _safe_text(artifact["role"], name=True)
        if role in artifacts:
            raise PaidfEvidenceError("source artifact roles must be unique")
        if not isinstance(artifact["sha256"], str) or not _HEX64.fullmatch(artifact["sha256"]):
            raise PaidfEvidenceError("source artifacts require full sha256 hashes")
        if type(artifact["size_bytes"]) is not int or artifact["size_bytes"] <= 0:
            raise PaidfEvidenceError("source artifacts must be nonempty")
        if "media_type" in artifact and artifact["media_type"] not in {"image", "video"}:
            raise PaidfEvidenceError("media_type must be image or video")
        artifacts[role] = artifact
    return artifacts


def _frames(contents: bytes, media_type: str):
    """Decode all source frames; reencoding pixels removes source file metadata."""
    from PIL import Image, ImageSequence

    if media_type == "image":
        with Image.open(BytesIO(contents)) as source:
            for frame in ImageSequence.Iterator(source):
                yield frame.convert("RGB")
    else:
        import av

        with av.open(BytesIO(contents)) as source:
            if not source.streams.video:
                raise PaidfEvidenceError("source video has no video stream")
            for frame in source.decode(video=0):
                yield frame.to_image().convert("RGB")


def _recording_id(evidence: dict[str, Any]) -> str:
    """Keep independent image evidence and revisions separate within a workflow."""
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{evidence['run_id']}:image-evidence:{hashlib.sha256(canonical).hexdigest()}"


def _inspect(path: Path, evidence: dict[str, Any], expected_media: dict) -> dict[str, Any]:
    """Independently decode the closed RRD and compare its factual rows/pixels."""
    from PIL import Image
    from rerun.recording import load_recording

    recording = load_recording(path)
    if recording.recording_id() != _recording_id(evidence) or recording.application_id() != APPLICATION_ID:
        raise PaidfEvidenceError("decoded recording identity differs from the run and image digest")
    batches: dict[str, list[Any]] = {}
    timelines = set()
    for chunk in recording.chunks():
        batch = chunk.to_record_batch()
        batches.setdefault(str(chunk.entity_path), []).append(batch)
        timelines.update(name for name in ("stage_index", "source_frame") if name in batch.schema.names)

    def rows(entity: str, column: str, timeline: str | None = None):
        found = []
        for batch in batches.get("/" + entity, []):
            if column not in batch.schema.names:
                continue
            values = batch.column(column).to_pylist()
            indices = batch.column(timeline).to_pylist() if timeline else [None] * len(values)
            found.extend((index, value[0]) for index, value in zip(indices, values) if value)
        return sorted(found, key=lambda row: -1 if row[0] is None else row[0])

    provenance = rows("provenance/run", "TextDocument:text")
    if len(provenance) != 1 or json.loads(provenance[0][1]) != evidence:
        raise PaidfEvidenceError("decoded provenance differs from normalized evidence")
    expected_scalars: dict[str, list[tuple[int, float]]] = {}
    expected_events = []
    for index, stage in enumerate(evidence["stages"]):
        expected_events.append((index, json.dumps({k: v for k, v in stage.items() if k != "metrics"}, sort_keys=True)))
        for name, value in {"duration_seconds": stage["duration_seconds"], **stage["metrics"]}.items():
            entity = f"stages/{stage['state']}/metrics/{name}"
            expected_scalars.setdefault(entity, []).append((index, float(value)))
    if rows("stages/events", "TextLog:text", "stage_index") != expected_events:
        raise PaidfEvidenceError("decoded stage events differ from source evidence")
    for entity, expected in expected_scalars.items():
        if rows(entity, "Scalars:scalars", "stage_index") != expected:
            raise PaidfEvidenceError("decoded scalar sequence differs from source evidence")
    for role, expected in expected_media.items():
        decoded = []
        for index, blob in rows(f"media/{role}", "EncodedImage:blob", "source_frame"):
            with Image.open(BytesIO(bytes(blob))) as source:
                rgb = source.convert("RGB")
                decoded.append({"index": index, "width": rgb.width, "height": rgb.height,
                                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        if decoded != expected["frames"] or not decoded:
            raise PaidfEvidenceError("decoded media pixels or frame sequence differ from the source")
    if timelines != {"stage_index", "source_frame"}:
        raise PaidfEvidenceError("recording is missing factual timelines")
    return {"recording_id": recording.recording_id(), "application_id": recording.application_id(),
            "timelines": sorted(timelines), "entity_paths": sorted(batches),
            "stage_count": len(evidence["stages"]), "scalar_series_count": len(expected_scalars),
            "media": expected_media}


def build_image_evidence_rrd(
    evidence: dict[str, Any], output_path: Path, media_paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Build a private RRD from the strict ``npa.paidf.image-evidence.v1`` contract.

    Required fields are ``image_name``, ``image_digest``, ``run_id``, full Git
    ``image_build_source_revision``, nonempty ``runtime_source_revisions``, their
    exact union ``source_revisions``, ``upstream_sources`` (repository/revision/license/
    adaptation), ``validation`` (status and named checks), ``stages`` (state,
    source_revision, status, duration_seconds, scalar metrics), and hashed
    ``source_artifacts`` (role/sha256/size_bytes, optionally media_type image or
    video). Optional fields are aggregate ``gpu`` (model/count) and factual
    ``limitations``. Unknown fields are rejected. Each declared media role must
    have exactly one supplied file, with matching hash and size. Real generation
    outputs or service-result context must be selected by the caller.

    ``stage_index`` means evidence-list order, not elapsed wall time. The separate
    ``source_frame`` timeline is each media file's zero-based decoded frame index;
    different files are not asserted to share capture times. Every decoded frame
    is embedded losslessly after removing ancillary metadata. No source path is
    logged or returned. The output is atomically created with mode 0600 and never
    overwrites existing evidence. The returned manifest describes readback, not
    an independently established workload-validation result.
    """
    import rerun as rr

    # Copy only JSON values so another thread cannot change provenance mid-write.
    try:
        evidence = json.loads(json.dumps(evidence, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PaidfEvidenceError("evidence must be finite JSON data") from exc
    artifacts = _validate(evidence)
    required_media = {role for role, artifact in artifacts.items() if "media_type" in artifact}
    if not required_media or set(media_paths) != required_media:
        raise PaidfEvidenceError("every recording requires exact declared media roles")
    output_path = Path(output_path)
    if output_path.suffix != ".rrd":
        raise PaidfEvidenceError("output_path must have an .rrd suffix")
    if output_path.exists():
        raise PaidfEvidenceError("existing evidence cannot be overwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".paidf-evidence-", suffix=".rrd", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    recording = rr.RecordingStream(APPLICATION_ID, recording_id=_recording_id(evidence))
    media = {}
    try:
        recording.save(temporary_path)
        recording.log("provenance/run", rr.TextDocument(json.dumps(evidence, sort_keys=True)), static=True)
        recording.log("provenance/timelines", rr.TextDocument(
            "NVIDIA Physical AI Data Factory; NPA evidence adaptation. stage_index is "
            "the supplied evidence order. source_frame is each source file's decoded "
            "frame index; files are not asserted to be synchronized. Media privacy "
            "and licensing and original run bindings are established by the caller."
        ), static=True)
        for index, stage in enumerate(evidence["stages"]):
            recording.set_time("stage_index", sequence=index)
            event = {k: v for k, v in stage.items() if k != "metrics"}
            recording.log("stages/events", rr.TextLog(json.dumps(event, sort_keys=True)))
            for name, value in {"duration_seconds": stage["duration_seconds"], **stage["metrics"]}.items():
                recording.log(f"stages/{stage['state']}/metrics/{name}", rr.Scalars(value))
        recording.reset_time()
        for role in sorted(media_paths):
            try:
                contents = Path(media_paths[role]).read_bytes()
                artifact = artifacts[role]
                if len(contents) != artifact["size_bytes"] or hashlib.sha256(contents).hexdigest() != artifact["sha256"]:
                    raise PaidfEvidenceError("media hash or size differs from its source artifact")
                frames = []
                for index, frame in enumerate(_frames(contents, artifact["media_type"])):
                    frames.append({"index": index, "width": frame.width, "height": frame.height,
                                   "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
                    buffer = BytesIO()
                    frame.info.clear()
                    frame.save(buffer, format="PNG")
                    recording.set_time("source_frame", sequence=index)
                    recording.log(f"media/{role}", rr.EncodedImage(contents=buffer.getvalue(), media_type="image/png"))
                if not frames:
                    raise PaidfEvidenceError("source media decoded no frames")
                media[role] = {"source_sha256": artifact["sha256"], "source_size_bytes": artifact["size_bytes"],
                               "frame_count": len(frames), "frames": frames}
            except PaidfEvidenceError:
                raise
            except Exception as exc:
                raise PaidfEvidenceError("source media could not be decoded") from exc
        recording.flush()
        recording.disconnect()
        decoded = _inspect(temporary_path, evidence, media)
        contents = temporary_path.read_bytes()
        result = {"schema": "npa.paidf.image-evidence-rrd.v1", "status": "validated",
                  "run_id": evidence["run_id"], "image_name": evidence["image_name"],
                  "image_digest": evidence["image_digest"], "source_revisions": evidence["source_revisions"],
                  "image_build_source_revision": evidence["image_build_source_revision"],
                  "runtime_source_revisions": evidence["runtime_source_revisions"],
                  "sha256": hashlib.sha256(contents).hexdigest(), "size_bytes": len(contents),
                  "validation": evidence["validation"], "decoded": decoded}
        # Link rather than replace: a racing writer must never overwrite evidence.
        os.link(temporary_path, output_path)
        return result
    finally:
        recording.disconnect()
        temporary_path.unlink(missing_ok=True)
