"""Prepare one immutable, input-conditioned PAIDF starter clip before GPU work.

The default is a small, pinned real robot capture fetched by the operator-side
CLI.  Geometric frames remain available only through the explicit fixture flag.
All video paths converge on the same canonical run prefix and provenance schema.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path
import re
import shutil
from string import Formatter
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class PaidfInputError(RuntimeError):
    """An input cannot be selected, verified, normalized, or staged safely."""


PROVENANCE_SCHEMA = "npa.paidf.input-provenance.v1"
CONDITIONING_FRAMES = 8
CONDITIONING_FRAME_COUNT = 93
CONDITIONING_FPS = 16
SUPPORTED_CODECS = frozenset({"h264"})
MAX_LEROBOT_INFO_BYTES = 1_000_000
MAX_LEROBOT_EPISODE_METADATA_BYTES = 64_000_000
MAX_LEROBOT_VIDEO_PATH_BYTES = 1_024


@dataclass(frozen=True)
class PreparedPaidfInput:
    """Non-secret result injected into workflow config and manifests."""

    selection: str
    provenance: dict[str, Any]
    reused: bool
    cache_status: str = ""

    def config_overrides(self) -> dict[str, str]:
        source = self.provenance
        customer_dataset = source.get("source_kind") == "lerobot_dataset"
        return {
            "seed_fixture": "true"
            if self.selection == "synthetic_fixture"
            else "false",
            # Compatibility with the shipped stage parameter during this schema
            # transition. Both names are explicit opt-ins; neither defaults true.
            "seed_default_input": (
                "true" if self.selection == "synthetic_fixture" else "false"
            ),
            "input_provenance_uri": str(source.get("provenance_uri") or ""),
            "input_source_kind": str(source.get("source_kind") or ""),
            "input_origin": str(source.get("input_origin") or ""),
            "input_origin_label": str(source.get("input_origin_label") or ""),
            # LeRobot content hashes are used only for in-memory immutable-byte
            # checks; they are not expanded into scheduler commands or persisted.
            "input_sha256": "" if customer_dataset else str(source.get("sha256") or ""),
            "input_license": str(source.get("asset_license") or ""),
            "input_staged_uri": str(source.get("staged_canonical_s3_uri") or ""),
            "input_source_format": str(source.get("source_format") or "video"),
            "input_authoritative_url": str(
                source.get("authoritative_upstream_url")
                or source.get("source_ref")
                or ""
            ),
            "input_immutable_revision": str(source.get("immutable_revision") or ""),
            "input_attribution": str(source.get("asset_attribution") or ""),
        }


def load_starter_contract() -> dict[str, Any]:
    """Load and minimally validate the packaged machine-readable asset contract."""

    path = resources.files("npa.assets").joinpath("paidf_starter_video.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "npa.paidf.starter-video.v1":
        raise PaidfInputError(
            "the packaged PAIDF starter-video contract has an unknown schema"
        )
    digest = str((payload.get("integrity") or {}).get("sha256") or "")
    if len(digest) != 64:
        raise PaidfInputError(
            "the packaged PAIDF starter-video contract has no SHA-256"
        )
    return payload


def select_paidf_input(
    *,
    input_video: Path | None = None,
    input_uri: str = "",
    lerobot_uri: str = "",
    seed_fixture: bool = False,
) -> str:
    """Resolve mutually exclusive CLI selectors without performing I/O."""

    selected = (
        int(input_video is not None)
        + int(bool(input_uri.strip()))
        + int(bool(lerobot_uri.strip()))
        + int(seed_fixture)
    )
    if selected > 1:
        raise PaidfInputError(
            "choose exactly one PAIDF input: --input-video, --input-uri, "
            "--lerobot-uri, or --seed-fixture; the options conflict"
        )
    if seed_fixture:
        return "synthetic_fixture"
    if input_video is not None:
        if input_video.suffix.lower() != ".mp4":
            raise PaidfInputError(
                "--input-video must be an MP4 path ending in .mp4; transcode the "
                "source before submit"
            )
        return "local_video"
    if input_uri.strip():
        parsed = urlparse(input_uri.strip())
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise PaidfInputError(
                "--input-uri must name one S3 object, for example "
                "s3://bucket/path/input.mp4"
            )
        if not parsed.path.lower().endswith(".mp4"):
            raise PaidfInputError("--input-uri must name one MP4 object ending in .mp4")
        return "object_uri"
    if lerobot_uri.strip():
        parsed = urlparse(lerobot_uri.strip())
        if parsed.scheme != "s3" or not parsed.netloc:
            raise PaidfInputError(
                "--lerobot-uri must name an S3 LeRobotDataset prefix, for example "
                "s3://bucket/datasets/robot-run/"
            )
        return "lerobot_dataset"
    return "starter"


def plan_paidf_input(
    *,
    run_id: str,
    bucket: str,
    input_video: Path | None = None,
    input_uri: str = "",
    lerobot_uri: str = "",
    lerobot_camera: str = "",
    lerobot_episode: int = 0,
    require_explicit_lerobot_selection: bool = False,
    lerobot_episode_was_explicit: bool = False,
    seed_fixture: bool = False,
) -> PreparedPaidfInput:
    """Describe selection for ``--plan-only`` without filesystem, S3, or network I/O."""

    selection = select_paidf_input(
        input_video=input_video,
        input_uri=input_uri,
        lerobot_uri=lerobot_uri,
        seed_fixture=seed_fixture,
    )
    validate_lerobot_selector(
        selection=selection,
        camera=lerobot_camera,
        episode=lerobot_episode,
        require_explicit_selection=require_explicit_lerobot_selection,
        episode_was_explicit=lerobot_episode_was_explicit,
    )
    base_uri = (
        f"s3://{bucket}/physical-ai-data-factory/{run_id}/input/"
        if bucket and bucket != "example-bucket"
        else ""
    )
    if selection == "synthetic_fixture":
        provenance = _fixture_provenance(run_id, base_uri)
    elif selection == "starter":
        contract = load_starter_contract()
        provenance = _build_provenance(
            run_id=run_id,
            base_uri=base_uri,
            source={
                **_starter_source_metadata(contract),
                "sha256": contract["integrity"]["sha256"],
                "byte_size": contract["integrity"]["byte_size"],
                "media": contract["media"],
            },
            derivation={
                "kind": "normalized_conditioning_clip",
                "staged_uri": f"{base_uri}conditioning.mp4" if base_uri else "",
            },
        )
    else:
        source_ref = (
            lerobot_uri.strip()
            if selection == "lerobot_dataset"
            else str(input_video)
            if input_video is not None
            else input_uri.strip()
        )
        source = (
            _lerobot_source_metadata(
                camera=lerobot_camera,
                explicit_selection=require_explicit_lerobot_selection,
            )
            if selection == "lerobot_dataset"
            else _user_source_metadata(
                source_ref=source_ref,
                transport="local_file" if input_video is not None else "s3_object",
            )
        )
        provenance = _build_provenance(
            run_id=run_id,
            base_uri=base_uri,
            source=source,
            derivation={
                "kind": "normalized_conditioning_clip",
                "staged_uri": f"{base_uri}conditioning.mp4" if base_uri else "",
            },
        )
    return PreparedPaidfInput(selection, provenance, reused=False)


def prepare_paidf_input(
    *,
    run_id: str,
    bucket: str,
    input_video: Path | None = None,
    input_uri: str = "",
    lerobot_uri: str = "",
    lerobot_camera: str = "",
    lerobot_episode: int = 0,
    require_explicit_lerobot_selection: bool = False,
    lerobot_episode_was_explicit: bool = False,
    seed_fixture: bool = False,
    endpoint_url: str = "",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    cache_dir: Path | None = None,
    offline: bool | None = None,
    storage_client: Any | None = None,
    reporter: Callable[[str], None] | None = None,
) -> PreparedPaidfInput:
    """Verify, normalize, and stage the selected input under the canonical prefix.

    ``provenance.json`` is the commit marker. A retry with no explicit selector
    reuses any valid committed input (including a prior user input) instead of
    replacing it with the starter. Explicit selectors must identify the same
    source once a run is committed.
    """

    selection = select_paidf_input(
        input_video=input_video,
        input_uri=input_uri,
        lerobot_uri=lerobot_uri,
        seed_fixture=seed_fixture,
    )
    validate_lerobot_selector(
        selection=selection,
        camera=lerobot_camera,
        episode=lerobot_episode,
        require_explicit_selection=require_explicit_lerobot_selection,
        episode_was_explicit=lerobot_episode_was_explicit,
    )
    clean_run_id = str(run_id or "").strip()
    clean_bucket = str(bucket or "").strip()
    if not clean_run_id:
        raise PaidfInputError("PAIDF input preparation requires a non-empty run id")
    if not clean_bucket or clean_bucket == "example-bucket":
        raise PaidfInputError(
            "PAIDF input preparation requires the real object-storage bucket; "
            "pass --var bucket=<bucket>"
        )
    base_uri = f"s3://{clean_bucket}/physical-ai-data-factory/{clean_run_id}/input/"

    if storage_client is None:
        from npa.clients.storage import StorageClient

        storage_client = StorageClient.from_environment(
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    report = reporter or (lambda _message: None)
    existing = _read_provenance(storage_client, base_uri)
    if selection == "synthetic_fixture":
        if existing:
            if existing.get("source_kind") != "synthetic_fixture":
                raise PaidfInputError(
                    "run input is immutable: --seed-fixture cannot replace the "
                    f"committed {existing.get('input_origin_label', 'input')} at "
                    f"{base_uri}. Use a new --run-id."
                )
            report(
                f"PAIDF input: reusing committed Synthetic seeded fixture at {base_uri}"
            )
            return PreparedPaidfInput(
                selection, dict(existing), reused=True, cache_status=""
            )
        legacy = _legacy_staged_video(storage_client, base_uri)
        if legacy:
            raise PaidfInputError(
                "run input is immutable: --seed-fixture cannot replace the existing "
                f"uncommitted video {legacy}. Use a new --run-id."
            )
        provenance = _fixture_provenance(clean_run_id, base_uri)
        return PreparedPaidfInput(selection, provenance, reused=False)

    if (
        existing
        and selection == "starter"
        and existing.get("source_kind") == "synthetic_fixture"
    ):
        report(f"PAIDF input: reusing committed Synthetic seeded fixture at {base_uri}")
        return PreparedPaidfInput(
            "synthetic_fixture", dict(existing), reused=True, cache_status=""
        )

    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise PaidfInputError(
            "ffprobe and ffmpeg are required to validate and prepare PAIDF video input; "
            "install FFmpeg before submit"
        )

    with tempfile.TemporaryDirectory(prefix="npa-paidf-input-") as tmp_name:
        tmp = Path(tmp_name)
        cache_status = ""
        requested_source: Path
        requested_meta: dict[str, Any]

        if existing and selection == "starter":
            # An implicit default must never replace a committed user source.
            if existing.get("source_kind") == "lerobot_dataset":
                report(
                    "PAIDF input: reusing committed operator-supplied "
                    "LeRobotDataset selection (source identifiers redacted)"
                )
            else:
                report(
                    "PAIDF input: reusing committed "
                    f"{existing.get('input_origin_label', 'run input')} at {base_uri}"
                )
            requested_source = _download_staged_source(storage_client, base_uri, tmp)
            # Confidential LeRobot provenance intentionally omits content hashes.
            # The immutable stage check below compares the bytes in memory without
            # serializing their digest into reports or object metadata.
            if existing.get("source_kind") != "lerobot_dataset":
                _verify_digest(
                    requested_source,
                    str(existing.get("sha256") or ""),
                    context="committed staged source",
                )
            requested_meta = dict(existing)
            selection = str(existing.get("source_kind") or "user_supplied")
        elif selection == "local_video":
            assert input_video is not None
            requested_source = input_video.expanduser().resolve()
            if not requested_source.is_file():
                raise PaidfInputError(
                    f"--input-video does not exist or is not a file: {requested_source}"
                )
            requested_meta = _user_source_metadata(
                # Provenance is portable and may be shared with workers/reviewers;
                # never persist the operator's absolute local directory.
                source_ref=requested_source.name,
                transport="local_file",
            )
        elif selection == "object_uri":
            requested_source = tmp / "requested-source.mp4"
            try:
                downloaded = Path(
                    storage_client.download_path(
                        input_uri.strip(), str(requested_source)
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise PaidfInputError(
                    f"could not fetch --input-uri {input_uri.strip()}: {exc}. "
                    "Check the object path and storage credentials."
                ) from exc
            if downloaded.is_file() and downloaded != requested_source:
                requested_source = downloaded
            if not requested_source.is_file():
                raise PaidfInputError(
                    f"--input-uri did not resolve to one readable object: {input_uri.strip()}"
                )
            requested_meta = _user_source_metadata(
                source_ref=input_uri.strip(), transport="s3_object"
            )
        elif selection == "lerobot_dataset":
            requested_source = tmp / "lerobot-source.mp4"
            requested_meta = _materialize_lerobot_episode(
                storage_client,
                lerobot_uri=lerobot_uri.strip(),
                camera=lerobot_camera,
                episode=lerobot_episode,
                explicit_selection=require_explicit_lerobot_selection,
                destination=requested_source,
            )
        else:
            legacy = _legacy_staged_video(storage_client, base_uri)
            if legacy:
                requested_source = tmp / "legacy-source.mp4"
                downloaded = Path(
                    storage_client.download_path(legacy, str(requested_source))
                )
                if downloaded.is_file() and downloaded != requested_source:
                    requested_source = downloaded
                requested_meta = _user_source_metadata(
                    source_ref=legacy, transport="pre_staged_s3_object"
                )
                selection = "user_supplied"
                report(
                    f"PAIDF input: adopting pre-staged user video {legacy}; the starter is not used"
                )
            else:
                contract = load_starter_contract()
                requested_source, cache_status = _fetch_starter(
                    contract,
                    cache_dir=cache_dir,
                    offline=offline,
                    reporter=report,
                )
                requested_meta = _starter_source_metadata(contract)

        media = probe_video(requested_source)
        source_sha = _sha256(requested_source)
        source_size = requested_source.stat().st_size
        requested_meta.update(
            {
                "sha256": source_sha,
                "byte_size": source_size,
                "media": media,
            }
        )
        _assert_existing_matches(
            existing, requested_meta, explicit=selection != "starter"
        )

        conditioning = tmp / "conditioning.mp4"
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        ffmpeg_contract = _derive_conditioning(requested_source, conditioning) or {
            "name": "ffmpeg",
            "version": "unreported",
            "arguments": _conditioning_arguments(),
        }
        conditioning_media = probe_video(conditioning)
        frames = _extract_conditioning_frames(conditioning, frames_dir)
        derivation = {
            "kind": "normalized_conditioning_clip",
            "derived_from_sha256": source_sha,
            "operations": [
                "repeat source as needed to exactly 93 frames",
                "sample at 16 fps",
                "preserve aspect ratio and letterbox to 1280x720",
                "encode H.264/yuv420p without audio",
            ],
            "sha256": _sha256(conditioning),
            "byte_size": conditioning.stat().st_size,
            "media": conditioning_media,
            "tool": ffmpeg_contract,
            "staged_uri": f"{base_uri}conditioning.mp4",
            "frame_derivation": {
                "kind": "derived_conditioning_frames",
                "count": len(frames),
                "staged_uri_pattern": f"{base_uri}conditioning-frame-%04d.png",
                "tool": {
                    "name": "ffmpeg",
                    "version": str(ffmpeg_contract.get("version") or "unreported"),
                    "arguments": _frame_extraction_arguments(),
                },
                "items": [
                    {"name": frame.name, "sha256": _sha256(frame)} for frame in frames
                ],
            },
        }
        provenance = _build_provenance(
            run_id=clean_run_id,
            base_uri=base_uri,
            source=requested_meta,
            derivation=derivation,
        )
        confidential_input = provenance.get("source_kind") == "lerobot_dataset"

        if existing:
            # Different ffmpeg/libx264 builds can produce different bytes from
            # the same source. Never stage those bytes beneath frozen provenance.
            _assert_existing_derivation_matches(existing, derivation)
            provenance = dict(existing)
        _stage_file(
            storage_client,
            requested_source,
            f"{base_uri}source.mp4",
            source_sha,
            immutable_source=True,
            persist_digest_metadata=not confidential_input,
        )
        _stage_file(
            storage_client,
            conditioning,
            f"{base_uri}conditioning.mp4",
            _sha256(conditioning),
            persist_digest_metadata=not confidential_input,
        )
        for frame in frames:
            _stage_file(
                storage_client,
                frame,
                f"{base_uri}{frame.name}",
                _sha256(frame),
                persist_digest_metadata=not confidential_input,
            )
        if not existing:
            _upload_json(
                storage_client,
                provenance,
                f"{base_uri}provenance.json",
                tmp,
                persist_digest_metadata=not confidential_input,
            )
        if provenance.get("source_kind") == "lerobot_dataset":
            report(
                "PAIDF input: "
                f"{'reused' if existing else 'prepared'} operator-supplied "
                "LeRobotDataset selection (source identifiers redacted)"
            )
        else:
            report(
                "PAIDF input: "
                f"{'reused' if existing else 'prepared'} {provenance['input_origin_label']} "
                f"sha256={provenance['sha256']} staged={base_uri}"
            )
        return PreparedPaidfInput(
            str(provenance.get("source_kind") or selection),
            provenance,
            reused=bool(existing),
            cache_status=cache_status,
        )


def probe_video(path: Path) -> dict[str, Any]:
    """Fail fast unless *path* is a decodable H.264 MP4 with positive dimensions."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate,nb_frames:format=format_name,duration,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PaidfInputError(
            f"video validation could not run ffprobe for {path.name}: {exc}. "
            "Install FFmpeg and verify the input locally before retrying."
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown ffprobe error").strip()
        raise PaidfInputError(f"video validation failed for {path.name}: {detail}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fmt = payload["format"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise PaidfInputError(
            f"video validation found no readable video stream in {path.name}"
        ) from exc
    formats = {
        part.strip().lower() for part in str(fmt.get("format_name") or "").split(",")
    }
    codec = str(stream.get("codec_name") or "").lower()
    width = _ffprobe_positive_int("width", stream.get("width"), path)
    height = _ffprobe_positive_int("height", stream.get("height"), path)
    duration = _ffprobe_positive_float("duration", fmt.get("duration"), path)
    frame_rate_raw = str(stream.get("avg_frame_rate") or "").strip()
    frame_rate = _ffprobe_frame_rate(frame_rate_raw, path)
    if "mp4" not in formats:
        raise PaidfInputError(
            f"unsupported video container for {path.name}: expected MP4, got "
            f"{fmt.get('format_name') or 'unknown'}"
        )
    if codec not in SUPPORTED_CODECS:
        raise PaidfInputError(
            f"unsupported video codec for {path.name}: expected H.264, got {codec or 'unknown'}. "
            "Transcode with ffmpeg -i INPUT -c:v libx264 -pix_fmt yuv420p OUTPUT.mp4"
        )
    frame_count_raw = stream.get("nb_frames")
    try:
        frame_count = _ffprobe_positive_int("nb_frames", frame_count_raw, path)
    except PaidfInputError:
        derived = duration * frame_rate
        if not math.isfinite(derived) or derived < 1 or derived > 100_000_000:
            raise PaidfInputError(
                f"ffprobe field nb_frames is {frame_count_raw!r} for {path.name}, and "
                "a safe bounded frame count cannot be derived from duration and "
                "avg_frame_rate. Re-encode the MP4 with explicit timestamps/frame rate."
            ) from None
        frame_count = max(1, int(round(derived)))
    return {
        "container": "mp4",
        "codec": codec,
        "codec_profile": str(stream.get("profile") or ""),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "frame_rate": frame_rate_raw,
        "frame_count": frame_count,
        "pixel_format": str(stream.get("pix_fmt") or ""),
    }


def _derive_conditioning(source: Path, output: Path) -> dict[str, Any]:
    vf = (
        f"fps={CONDITIONING_FPS},"
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
        "-vf",
        vf,
        "-frames:v",
        str(CONDITIONING_FRAME_COUNT),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg(command, "derive the PAIDF conditioning clip")
    return {
        "name": "ffmpeg",
        "version": _ffmpeg_version(),
        "arguments": _conditioning_arguments(),
    }


def _conditioning_arguments() -> list[str]:
    """Stable path-free ffmpeg arguments that determine derived clip bytes."""

    return [
        "-stream_loop",
        "-1",
        "-i",
        "<source-by-sha256>",
        "-vf",
        (
            f"fps={CONDITIONING_FPS},"
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black"
        ),
        "-frames:v",
        str(CONDITIONING_FRAME_COUNT),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "<conditioning-by-sha256>",
    ]


def _ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unreported"
    first_line = (result.stdout or result.stderr or "").splitlines()
    return first_line[0].strip()[:240] if first_line else "unreported"


def _ffprobe_positive_int(field: str, value: Any, path: Path) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise PaidfInputError(
            f"ffprobe field {field} is {value!r} for {path.name}; expected a positive "
            "integer. Re-encode the MP4 or inspect it with `ffprobe -show_streams`."
        ) from None
    if parsed <= 0:
        raise PaidfInputError(
            f"ffprobe field {field} is {value!r} for {path.name}; expected a positive "
            "integer. Re-encode the MP4 or inspect it with `ffprobe -show_streams`."
        )
    return parsed


def _ffprobe_positive_float(field: str, value: Any, path: Path) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        parsed = 0.0
    if not math.isfinite(parsed) or parsed <= 0:
        raise PaidfInputError(
            f"ffprobe field {field} is {value!r} for {path.name}; expected a finite "
            "positive number. Re-encode the MP4 with valid timestamps."
        )
    return parsed


def _ffprobe_frame_rate(value: str, path: Path) -> float:
    from fractions import Fraction

    try:
        parsed = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        parsed = 0.0
    if not math.isfinite(parsed) or parsed <= 0:
        raise PaidfInputError(
            f"ffprobe field avg_frame_rate is {value!r} for {path.name}; expected a "
            "positive rational such as 30/1. Re-encode with an explicit frame rate."
        )
    return parsed


def _extract_conditioning_frames(conditioning: Path, output_dir: Path) -> list[Path]:
    interval = max(1, CONDITIONING_FRAME_COUNT // CONDITIONING_FRAMES)
    pattern = output_dir / "conditioning-frame-%04d.png"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(conditioning),
        "-vf",
        f"select=not(mod(n\\,{interval}))",
        "-vsync",
        "vfr",
        "-frames:v",
        str(CONDITIONING_FRAMES),
        str(pattern),
    ]
    _run_ffmpeg(command, "extract PAIDF conditioning frames")
    frames = sorted(output_dir.glob("conditioning-frame-*.png"))
    if len(frames) != CONDITIONING_FRAMES:
        raise PaidfInputError(
            f"conditioning-frame extraction produced {len(frames)} frames; "
            f"expected {CONDITIONING_FRAMES}"
        )
    return frames


def _frame_extraction_arguments() -> list[str]:
    interval = max(1, CONDITIONING_FRAME_COUNT // CONDITIONING_FRAMES)
    return [
        "-i",
        "<conditioning-by-sha256>",
        "-vf",
        f"select=not(mod(n\\,{interval}))",
        "-vsync",
        "vfr",
        "-frames:v",
        str(CONDITIONING_FRAMES),
        "<frame-pattern>",
    ]


def _run_ffmpeg(command: list[str], action: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        raise PaidfInputError(f"could not {action}: {detail}")


def _fetch_starter(
    contract: dict[str, Any],
    *,
    cache_dir: Path | None,
    offline: bool | None,
    reporter: Callable[[str], None],
) -> tuple[Path, str]:
    integrity = dict(contract["integrity"])
    license_data = dict(contract.get("license") or {})
    delivery = dict(contract.get("delivery") or {})
    digest = str(integrity["sha256"])
    size = int(integrity["byte_size"])
    source_url = str(contract["source"]["asset_url"])
    root = cache_dir or _default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{contract['asset_id']}-{digest[:16]}.mp4"
    offline_mode = _env_truthy("NPA_PAIDF_OFFLINE") if offline is None else offline
    acceptance_env = str(
        delivery.get("acceptance_environment_variable") or "NPA_PAIDF_ACCEPT_LICENSE"
    )
    if license_data.get("acceptance_required") and not _env_truthy(acceptance_env):
        raise PaidfInputError(
            "the pinned PAIDF starter requires explicit upstream license acceptance; "
            f"review {license_data.get('url') or 'the asset license'} and set "
            f"{acceptance_env}=1 before fetching"
        )
    token_env = str(delivery.get("authentication_environment_variable") or "HF_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if license_data.get("authentication_required") and not token:
        raise PaidfInputError(
            "the pinned PAIDF starter requires upstream authentication; configure "
            f"{token_env} before fetching. The token is never stored in provenance."
        )
    with _cache_lock(target):
        if target.is_file():
            try:
                _verify_file(target, digest, size, context="cached PAIDF starter")
            except PaidfInputError:
                if offline_mode:
                    raise PaidfInputError(
                        f"offline PAIDF cache entry failed integrity verification: {target}; "
                        "disable NPA_PAIDF_OFFLINE to refetch it"
                    )
                target.unlink()
                reporter(f"PAIDF input cache: invalid entry removed ({target})")
            else:
                reporter(f"PAIDF input cache: verified hit ({target})")
                return target, "verified_hit"
        if offline_mode:
            raise PaidfInputError(
                f"offline PAIDF cache miss: {target}. Populate it with one online submit "
                "or unset NPA_PAIDF_OFFLINE; no synthetic fallback is used."
            )
        reporter(f"PAIDF input cache: miss; fetching pinned asset {source_url}")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            part = target.with_suffix(f".attempt-{attempt}.part")
            try:
                with (
                    urlopen(
                        Request(
                            source_url,
                            headers=(
                                {"Authorization": f"Bearer {token}"} if token else {}
                            ),
                        ),
                        timeout=30,
                    ) as response,
                    part.open("wb") as handle,
                ):
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                _verify_file(part, digest, size, context="downloaded PAIDF starter")
                os.replace(part, target)
                reporter(f"PAIDF input cache: verified fetch stored at {target}")
                return target, "verified_fetch"
            except (HTTPError, URLError, OSError, PaidfInputError) as exc:
                last_error = exc
                part.unlink(missing_ok=True)
                if isinstance(exc, PaidfInputError):
                    # A digest/size mismatch is deterministic and must fail closed;
                    # retrying the same immutable bytes only obscures the evidence.
                    break
                if attempt < 3:
                    time.sleep(min(attempt, 2))
        attempt_detail = (
            "integrity validation failed on the downloaded bytes"
            if isinstance(last_error, PaidfInputError)
            else "all 3 bounded request attempts failed"
        )
        raise PaidfInputError(
            f"failed to fetch the pinned PAIDF starter: {attempt_detail}: {last_error}. "
            "No synthetic fallback is used; check network access or pre-populate "
            "the verified cache."
        ) from last_error


def _default_cache_dir() -> Path:
    override = os.environ.get("NPA_PAIDF_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "npa" / "physical-ai-data-factory"


@contextmanager
def _cache_lock(target: Path) -> Iterator[None]:
    lock = target.with_suffix(target.suffix + ".lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _starter_source_metadata(contract: dict[str, Any]) -> dict[str, Any]:
    source = dict(contract["source"])
    license_data = dict(contract["license"])
    return {
        "source_kind": "upstream_sample",
        "input_origin": "actual_capture",
        "input_origin_label": "Upstream real sample",
        "authoritative_upstream_url": source["authoritative_url"],
        "immutable_revision": source["immutable_revision"],
        "source_asset_path": source["asset_path"],
        "source_asset_url": source["asset_url"],
        "episode_metadata_url": source["episode_metadata_url"],
        "authenticity": dict(contract["authenticity"]),
        "asset_license": license_data["spdx_id"],
        "asset_license_url": license_data["url"],
        "asset_attribution": license_data["attribution"],
        "redistribution_permitted": license_data["redistribution_permitted"],
        "hosted_service_use_permitted": license_data["hosted_service_use_permitted"],
        "field_of_use_restrictions": license_data["field_of_use_restrictions"],
        "acceptance_required": license_data["acceptance_required"],
        "authentication_required": license_data["authentication_required"],
        "delivery_mode": "operator_runtime_fetch",
    }


def _user_source_metadata(*, source_ref: str, transport: str) -> dict[str, Any]:
    return {
        "source_kind": "user_supplied",
        "input_origin": "operator_supplied",
        "input_origin_label": "User-supplied input",
        "source_ref": source_ref,
        "transport": transport,
        "authenticity": {
            "classification": "operator_supplied_unverified",
            "evidence": "NPA does not infer whether user-supplied media is captured or generated.",
        },
        "asset_license": "operator-managed",
        "asset_license_url": "",
        "asset_attribution": "operator-managed",
        "redistribution_permitted": None,
        "hosted_service_use_permitted": None,
        "field_of_use_restrictions": "operator-managed",
        "acceptance_required": None,
        "authentication_required": None,
        "delivery_mode": transport,
    }


def validate_lerobot_selector(
    *,
    selection: str,
    camera: str,
    episode: int,
    require_explicit_selection: bool = False,
    episode_was_explicit: bool = False,
) -> None:
    """Validate LeRobot-only selectors before any object-store access."""

    if episode < 0:
        raise PaidfInputError("--lerobot-episode must be a non-negative integer")
    if require_explicit_selection:
        if selection != "lerobot_dataset":
            raise PaidfInputError(
                "--require-explicit-lerobot-selection requires --lerobot-uri"
            )
        missing: list[str] = []
        if not camera.strip():
            missing.append("--lerobot-camera")
        if not episode_was_explicit:
            missing.append("--lerobot-episode")
        if missing:
            raise PaidfInputError(
                "--require-explicit-lerobot-selection fails closed unless the "
                "operator supplies " + " and ".join(missing)
            )
    if selection != "lerobot_dataset" and (
        camera.strip() or episode != 0 or episode_was_explicit
    ):
        raise PaidfInputError(
            "--lerobot-camera and --lerobot-episode require --lerobot-uri"
        )


def _lerobot_source_metadata(
    *, camera: str, explicit_selection: bool = False
) -> dict[str, Any]:
    """Describe a selected LeRobot trajectory without inferring its ownership."""

    # The source prefix, episode, and feature names can themselves be customer
    # metadata. Durable provenance records only the generic selection policy.
    return {
        "source_kind": "lerobot_dataset",
        "source_format": "lerobot",
        "input_origin": "operator_supplied_dataset",
        "input_origin_label": "Operator-supplied LeRobotDataset",
        "source_ref": "",
        "transport": "s3_lerobot_dataset",
        "lerobot_selection": {
            "episode_selector": "operator-supplied",
            "camera_selector": "explicit" if camera.strip() else "automatic",
            "selection_contract": (
                "explicit-camera-and-episode"
                if explicit_selection
                else "compatibility-defaults"
            ),
        },
        "authenticity": {
            "classification": "operator_supplied_unverified",
            "evidence": (
                "NPA validated the LeRobot metadata/media layout but does not infer "
                "whether operator-supplied observations are captured or generated."
            ),
        },
        "asset_license": "operator-managed",
        "asset_license_url": "",
        "asset_attribution": "operator-managed",
        "redistribution_permitted": None,
        "hosted_service_use_permitted": None,
        "field_of_use_restrictions": "operator-managed",
        "acceptance_required": None,
        "authentication_required": None,
        "delivery_mode": "operator_runtime_selection",
    }


def _materialize_lerobot_episode(
    client: Any,
    *,
    lerobot_uri: str,
    camera: str,
    episode: int,
    explicit_selection: bool,
    destination: Path,
) -> dict[str, Any]:
    """Select one LeRobot camera/episode MP4 without downloading the dataset.

    LeRobot v2/v3 datasets place ``meta/info.json`` below the dataset root and
    videos below ``videos/``. The selector intentionally downloads only the
    chosen trajectory: production datasets can be many terabytes and input
    preparation must finish before GPU provisioning.
    """

    bucket, prefix = _split_s3(lerobot_uri)
    normalized_prefix = prefix.rstrip("/") + "/" if prefix else ""
    info_key = f"{normalized_prefix}meta/info.json"
    try:
        response = client.s3.get_object(Bucket=bucket, Key=info_key)
        body = response["Body"]
        try:
            raw_info = body.read(MAX_LEROBOT_INFO_BYTES + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # noqa: BLE001
        if _missing_s3(exc):
            raise PaidfInputError(
                "--lerobot-uri does not contain a LeRobot meta/info.json contract"
            ) from exc
        raise PaidfInputError(
            "could not read the configured LeRobot meta/info.json contract"
        ) from exc
    if len(raw_info) > MAX_LEROBOT_INFO_BYTES:
        raise PaidfInputError(
            "LeRobot meta/info.json exceeds the 1 MB validation limit"
        )
    try:
        info = json.loads(raw_info.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaidfInputError(
            "could not validate the configured LeRobot meta/info.json contract"
        ) from exc
    if not isinstance(info, dict) or not isinstance(info.get("features"), dict):
        raise PaidfInputError(
            "LeRobot meta/info.json must contain a features mapping"
        )
    version_family = _lerobot_version_family(info)
    video_features = sorted(
        str(name)
        for name, feature in info["features"].items()
        if isinstance(feature, dict) and str(feature.get("dtype") or "") == "video"
    )
    if not video_features:
        raise PaidfInputError(
            "LeRobot meta/info.json declares no video observation features"
        )
    try:
        raw_total_episodes = info.get("total_episodes")
        if isinstance(raw_total_episodes, bool):
            raise ValueError
        total_episodes = int(raw_total_episodes)
    except (TypeError, ValueError) as exc:
        raise PaidfInputError(
            "LeRobot total_episodes must be a positive integer"
        ) from exc
    if total_episodes < 1:
        raise PaidfInputError(
            "LeRobot total_episodes must be a positive integer"
        )
    if episode >= total_episodes:
        raise PaidfInputError(
            "the requested LeRobot episode is outside the declared dataset range"
        )
    try:
        raw_chunks_size = info.get("chunks_size", 1_000)
        if isinstance(raw_chunks_size, bool):
            raise ValueError
        chunks_size = int(raw_chunks_size)
    except (TypeError, ValueError) as exc:
        raise PaidfInputError("LeRobot chunks_size must be an integer") from exc
    if chunks_size < 1:
        raise PaidfInputError("LeRobot chunks_size must be a positive integer")
    camera_name = camera.strip()
    if camera_name and camera_name not in video_features:
        raise PaidfInputError(
            "the requested LeRobot camera is not a declared video feature"
        )
    candidate_features = [camera_name] if camera_name else video_features
    selected_key = ""
    clip_start = clip_end = None
    if version_family == 3:
        # LeRobot v3 groups many episodes into each video file. Resolve its
        # metadata row and trim only the selected episode rather than staging the
        # entire shared file as one trajectory.
        record = _read_lerobot_episode_record(
            client,
            bucket=bucket,
            prefix=normalized_prefix,
            episode=episode,
            chunks_size=chunks_size,
            destination_dir=destination.parent,
        )
        template = str(info.get("video_path") or "").strip() or (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        missing_time_range = False
        for feature in candidate_features:
            chunk_index = record.get(f"videos/{feature}/chunk_index")
            file_index = record.get(f"videos/{feature}/file_index")
            if chunk_index is None or file_index is None:
                if camera_name:
                    raise PaidfInputError(
                        "LeRobot v3 video metadata is missing the selected camera location"
                    )
                continue
            rel = _format_lerobot_video_path(
                template,
                feature=feature,
                episode=episode,
                episode_chunk=episode // chunks_size,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            key = f"{normalized_prefix}{rel}"
            if _s3_object_exists(client, bucket=bucket, key=key):
                candidate_start = record.get(f"videos/{feature}/from_timestamp")
                candidate_end = record.get(f"videos/{feature}/to_timestamp")
                if candidate_start is None or candidate_end is None:
                    missing_time_range = True
                    continue
                selected_key = key
                clip_start = candidate_start
                clip_end = candidate_end
                break
        if not selected_key and missing_time_range:
            raise PaidfInputError(
                "LeRobot v3 video metadata is missing the selected episode time range"
            )
    else:
        template = str(info.get("video_path") or "").strip() or (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        )
        for feature in candidate_features:
            rel = _format_lerobot_video_path(
                template,
                feature=feature,
                episode=episode,
                episode_chunk=episode // chunks_size,
                chunk_index=episode // chunks_size,
                file_index=episode,
            )
            key = f"{normalized_prefix}{rel}"
            if _s3_object_exists(client, bucket=bucket, key=key):
                selected_key = key
                break
    if not selected_key:
        raise PaidfInputError(
            "the requested LeRobot episode/camera has no resolvable observation video"
        )
    download_target = (
        destination.with_name("lerobot-shared-source.mp4")
        if clip_start is not None or clip_end is not None
        else destination
    )
    try:
        client.s3.download_file(bucket, selected_key, str(download_target))
    except Exception as exc:  # noqa: BLE001
        raise PaidfInputError(
            "could not read the selected LeRobot observation video; verify "
            "least-privilege GetObject access"
        ) from exc
    if clip_start is not None or clip_end is not None:
        _trim_lerobot_episode(
            download_target,
            destination,
            start_seconds=clip_start,
            end_seconds=clip_end,
        )
        download_target.unlink(missing_ok=True)
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise PaidfInputError("the selected LeRobot observation video is empty")
    metadata = _lerobot_source_metadata(
        camera=camera,
        explicit_selection=explicit_selection,
    )
    metadata["lerobot_selection"].update(
        {
            "media_kind": "video",
            "selected_object": "redacted",
        }
    )
    return metadata


def _read_lerobot_episode_record(
    client: Any,
    *,
    bucket: str,
    prefix: str,
    episode: int,
    chunks_size: int,
    destination_dir: Path,
) -> dict[str, Any]:
    """Read one LeRobot v3 episode row without materializing dataset media."""

    if chunks_size < 1:
        raise PaidfInputError("LeRobot chunks_size must be a positive integer")
    chunk = episode // chunks_size
    marker = f"{prefix}meta/episodes/chunk-{chunk:03d}/"
    try:
        candidates = sorted(
            key
            for key in _list_s3_keys(client, bucket=bucket, prefix=marker)
            if key.lower().endswith(".parquet")
        )
    except Exception as exc:  # noqa: BLE001
        raise PaidfInputError(
            "could not inspect the selected LeRobot episode-metadata shard"
        ) from exc
    if not candidates:
        raise PaidfInputError(
            "the requested LeRobot episode has no episode-metadata shard"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PaidfInputError(
            "pyarrow is required to resolve LeRobot v3 episode metadata"
        ) from exc
    for index, key in enumerate(candidates):
        local = destination_dir / f"lerobot-episode-metadata-{index:03d}.parquet"
        try:
            head = client.s3.head_object(Bucket=bucket, Key=key)
            size = int(head.get("ContentLength") or 0)
            if not 0 < size <= MAX_LEROBOT_EPISODE_METADATA_BYTES:
                raise PaidfInputError(
                    "LeRobot v3 episode metadata exceeds the safe shard-size limit"
                )
            client.s3.download_file(bucket, key, str(local))
            rows = pq.read_table(
                local, filters=[("episode_index", "=", episode)]
            ).to_pylist()
        except PaidfInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaidfInputError(
                "could not read LeRobot v3 episode metadata"
            ) from exc
        finally:
            local.unlink(missing_ok=True)
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                row_episode = int(row.get("episode_index", -1))
            except (TypeError, ValueError):
                continue
            if row_episode == episode:
                return row
    raise PaidfInputError(
        "the requested LeRobot episode is absent from its metadata shard"
    )


def _lerobot_version_family(info: dict[str, Any]) -> int:
    """Return a supported LeRobot major format version without guessing."""

    raw = str(info.get("codebase_version") or "").strip().lower()
    normalized = raw[1:] if raw.startswith("v") else raw
    try:
        major = int(normalized.split(".", 1)[0])
    except (TypeError, ValueError) as exc:
        raise PaidfInputError(
            "LeRobot meta/info.json has no supported codebase_version"
        ) from exc
    if major not in {2, 3}:
        raise PaidfInputError(
            "PAIDF supports LeRobotDataset v2.x and v3.x video layouts"
        )
    return major


def _format_lerobot_video_path(
    template: str,
    *,
    feature: str,
    episode: int,
    episode_chunk: int,
    chunk_index: Any,
    file_index: Any,
) -> str:
    """Resolve the official LeRobot video template and keep it under videos/."""

    if len(template.encode("utf-8")) > MAX_LEROBOT_VIDEO_PATH_BYTES:
        raise PaidfInputError("LeRobot video_path exceeds the safe length limit")
    allowed_fields = {
        "video_key",
        "episode_index",
        "episode_chunk",
        "chunk_index",
        "file_index",
    }
    try:
        fields = list(Formatter().parse(template))
    except ValueError as exc:
        raise PaidfInputError("LeRobot video_path has invalid format syntax") from exc
    for _literal, field_name, format_spec, conversion in fields:
        if field_name is None:
            continue
        if (
            field_name not in allowed_fields
            or conversion is not None
            or (format_spec and re.fullmatch(r"0?[1-9][0-9]*d", format_spec) is None)
        ):
            raise PaidfInputError(
                "LeRobot video_path contains an unsupported format field"
            )
    try:
        resolved = template.format(
            video_key=feature,
            episode_index=episode,
            episode_chunk=episode_chunk,
            chunk_index=int(chunk_index),
            file_index=int(file_index),
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise PaidfInputError(
            "LeRobot video_path cannot resolve the requested episode"
        ) from exc
    normalized = resolved.replace("\\", "/").strip()
    parsed = urlparse(normalized)
    parts = Path(normalized).parts
    if (
        not normalized
        or parsed.scheme
        or normalized.startswith("/")
        or not normalized.lower().endswith(".mp4")
        or not parts
        or parts[0] != "videos"
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PaidfInputError(
            "LeRobot video_path must resolve to one relative videos/... MP4 object"
        )
    return normalized


def _s3_object_exists(client: Any, *, bucket: str, key: str) -> bool:
    """Check one declared object without enumerating a production dataset."""

    try:
        client.s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        if _missing_s3(exc):
            return False
        raise PaidfInputError(
            "could not verify the selected LeRobot observation object"
        ) from exc
    return True


def _trim_lerobot_episode(
    source: Path,
    destination: Path,
    *,
    start_seconds: Any,
    end_seconds: Any,
) -> None:
    """Extract one exact LeRobot v3 episode from a shared H.264 video file."""

    try:
        start = float(start_seconds)
        end = float(end_seconds)
    except (TypeError, ValueError) as exc:
        raise PaidfInputError(
            "LeRobot v3 video metadata requires numeric episode timestamps"
        ) from exc
    if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end):
        raise PaidfInputError(
            "LeRobot v3 video metadata has an invalid episode time range"
        )
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start:.9f}",
        "-t",
        f"{end - start:.9f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PaidfInputError("could not extract the selected LeRobot episode") from exc
    if result.returncode != 0 or not destination.is_file():
        raise PaidfInputError("could not extract the selected LeRobot episode")


def _list_s3_keys(client: Any, *, bucket: str, prefix: str) -> list[str]:
    """List a prefix across pages while remaining easy to fake in unit tests."""

    paginator_factory = getattr(client.s3, "get_paginator", None)
    if callable(paginator_factory):
        pages = paginator_factory("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
    else:
        pages = [client.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)]
    return [
        str(item.get("Key") or "")
        for page in pages
        for item in page.get("Contents", [])
        if str(item.get("Key") or "")
    ]


def _fixture_provenance(run_id: str, base_uri: str) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "run_id": run_id,
        "source_kind": "synthetic_fixture",
        "kind": "npa_seeded_fixture",
        "input_origin": "synthetic_generated",
        "input_origin_label": "Synthetic seeded fixture",
        "authenticity": {
            "classification": "synthetic_generated",
            "evidence": "NPA-generated geometric frames; explicit --seed-fixture only.",
        },
        "asset_license": "repository-authored test fixture",
        "sha256": "",
        "staged_canonical_s3_uri": base_uri,
        "staged_source_uri": "",
        "provenance_uri": f"{base_uri}provenance.json",
        "derivation": {
            "kind": "fixture_frames_to_conditioning_clip",
            "staged_uri": f"{base_uri}conditioning.mp4",
        },
        "cosmos_conditioning": {
            "enabled": True,
            "staged_uri": f"{base_uri}conditioning.mp4",
        },
    }


def _build_provenance(
    *, run_id: str, base_uri: str, source: dict[str, Any], derivation: dict[str, Any]
) -> dict[str, Any]:
    safe_source = dict(source)
    safe_derivation = json.loads(json.dumps(derivation))
    if safe_source.get("source_kind") == "lerobot_dataset":
        # Source and derived hashes are confidential customer-content identifiers.
        # Keep them in memory for immutable stage comparisons, but never serialize
        # them into provenance, downstream reports, or object metadata.
        safe_source.pop("sha256", None)
        safe_derivation.pop("derived_from_sha256", None)
        safe_derivation.pop("sha256", None)
        frame_derivation = safe_derivation.get("frame_derivation")
        if isinstance(frame_derivation, dict):
            for item in frame_derivation.get("items") or []:
                if isinstance(item, dict):
                    item.pop("sha256", None)
    payload = {
        "schema_version": PROVENANCE_SCHEMA,
        "run_id": run_id,
        **safe_source,
        "staged_canonical_s3_uri": base_uri,
        "staged_source_uri": f"{base_uri}source.mp4",
        "provenance_uri": f"{base_uri}provenance.json",
        "derivation": safe_derivation,
        "cosmos_conditioning": {
            "enabled": True,
            "mode": "input_conditioned",
            "staged_uri": f"{base_uri}conditioning.mp4",
            "environment_equivalent": "NPA_COSMOS_CONDITION_ON_INPUT=1",
            "cli_equivalent": "--condition-on-input",
        },
    }
    return payload


def _assert_existing_matches(
    existing: dict[str, Any] | None, requested: dict[str, Any], *, explicit: bool
) -> None:
    if not existing:
        return
    existing_sha = str(existing.get("sha256") or "")
    requested_sha = str(requested.get("sha256") or "")
    if (
        not existing_sha
        and (
            existing.get("source_kind") == "lerobot_dataset"
            or requested.get("source_kind") == "lerobot_dataset"
        )
    ):
        # _stage_file compares the selected bytes against the immutable staged
        # source without persisting the confidential digest.
        return
    if existing_sha != requested_sha:
        if (
            existing.get("source_kind") == "lerobot_dataset"
            or requested.get("source_kind") == "lerobot_dataset"
        ):
            raise PaidfInputError(
                "run input is immutable: the selected LeRobot observation differs "
                "from the committed run input. Use a new --run-id."
            )
        qualifier = "explicit input" if explicit else "selected input"
        raise PaidfInputError(
            f"run input is immutable: {qualifier} SHA-256 {requested_sha} does not match "
            f"the committed run input {existing_sha}. Use a new --run-id."
        )


def _assert_existing_derivation_matches(
    existing: dict[str, Any], derived: dict[str, Any]
) -> None:
    """Require every recomputed artifact byte hash to match committed provenance."""

    if existing.get("source_kind") == "lerobot_dataset":
        # Confidential provenance has no content hashes; each existing derived
        # object is compared byte-for-byte through _stage_file instead.
        return

    committed = existing.get("derivation")
    if not isinstance(committed, dict):
        raise PaidfInputError(
            "committed PAIDF provenance has no derivation contract; use a new --run-id"
        )
    expected_clip = str(committed.get("sha256") or "")
    actual_clip = str(derived.get("sha256") or "")
    expected_frames = committed.get("frame_derivation")
    actual_frames = derived.get("frame_derivation")
    expected_items = (
        expected_frames.get("items") if isinstance(expected_frames, dict) else None
    )
    actual_items = (
        actual_frames.get("items") if isinstance(actual_frames, dict) else None
    )
    expected_hashes = {
        str(item.get("name") or ""): str(item.get("sha256") or "")
        for item in (expected_items or [])
        if isinstance(item, dict)
    }
    actual_hashes = {
        str(item.get("name") or ""): str(item.get("sha256") or "")
        for item in (actual_items or [])
        if isinstance(item, dict)
    }
    if (
        not expected_clip
        or expected_clip != actual_clip
        or not expected_hashes
        or expected_hashes != actual_hashes
    ):
        committed_tool = committed.get("tool")
        current_tool = derived.get("tool")
        raise PaidfInputError(
            "run input is immutable: recomputed conditioning bytes differ from the "
            "committed provenance (for example, because ffmpeg/libx264 changed). "
            f"Committed tool={committed_tool!r}; current tool={current_tool!r}. "
            "Use a new --run-id instead of mixing artifact builds."
        )


def _read_provenance(client: Any, base_uri: str) -> dict[str, Any] | None:
    bucket, key = _split_s3(f"{base_uri}provenance.json")
    try:
        response = client.s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        if _missing_s3(exc):
            return None
        raise PaidfInputError(
            f"could not read existing PAIDF input provenance: {exc}"
        ) from exc
    try:
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaidfInputError(
            f"existing PAIDF input provenance is invalid at {base_uri}provenance.json"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != PROVENANCE_SCHEMA
    ):
        raise PaidfInputError(
            f"existing PAIDF input provenance has an unsupported schema at {base_uri}"
        )
    return payload


def _legacy_staged_video(client: Any, base_uri: str) -> str:
    bucket, prefix = _split_s3(base_uri)
    try:
        response = client.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception as exc:  # noqa: BLE001
        raise PaidfInputError(
            f"could not inspect the canonical PAIDF input prefix: {exc}"
        ) from exc
    videos = [
        str(item.get("Key") or "")
        for item in response.get("Contents", [])
        if str(item.get("Key") or "").lower().endswith(".mp4")
        and not str(item.get("Key") or "").lower().endswith("conditioning.mp4")
    ]
    if len(videos) > 1:
        raise PaidfInputError(
            f"multiple uncommitted videos already exist under {base_uri}; pass one "
            "explicitly with --input-uri or use a new --run-id"
        )
    if videos:
        return f"s3://{bucket}/{videos[0]}"
    other = [
        str(item.get("Key") or "")
        for item in response.get("Contents", [])
        if str(item.get("Key") or "")
    ]
    if other:
        raise PaidfInputError(
            f"uncommitted input artifacts already exist under {base_uri}, but no source "
            "MP4 can be adopted. Use --input-video/--input-uri, --seed-fixture, or a new run id."
        )
    return ""


def _download_staged_source(client: Any, base_uri: str, tmp: Path) -> Path:
    target = tmp / "committed-source.mp4"
    try:
        downloaded = Path(client.download_path(f"{base_uri}source.mp4", str(target)))
    except Exception as exc:  # noqa: BLE001
        raise PaidfInputError(
            f"committed PAIDF source is unavailable at {base_uri}source.mp4: {exc}"
        ) from exc
    if downloaded.is_file() and downloaded != target:
        target = downloaded
    if not target.is_file():
        raise PaidfInputError(
            f"committed PAIDF source is missing at {base_uri}source.mp4"
        )
    return target


def _stage_file(
    client: Any,
    local: Path,
    uri: str,
    digest: str,
    *,
    immutable_source: bool = False,
    persist_digest_metadata: bool = True,
) -> None:
    bucket, key = _split_s3(uri)
    try:
        head = client.s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        if not _missing_s3(exc):
            raise PaidfInputError(
                f"could not inspect staged PAIDF input {uri}: {exc}"
            ) from exc
    else:
        recorded = str((head.get("Metadata") or {}).get("sha256") or "")
        if recorded == digest:
            return
        with tempfile.TemporaryDirectory(prefix="npa-paidf-verify-") as tmp:
            existing = Path(tmp) / Path(key).name
            client.s3.download_file(bucket, key, str(existing))
            if _sha256(existing) == digest:
                return
        role = "canonical user source" if immutable_source else "derived artifact"
        raise PaidfInputError(
            f"refusing to overwrite an existing PAIDF {role} with different bytes at "
            f"{uri}; use a new --run-id"
        )
    try:
        metadata = {"npa-role": "paidf-input"}
        if persist_digest_metadata:
            metadata["sha256"] = digest
        client.s3.upload_file(
            str(local),
            bucket,
            key,
            ExtraArgs={"Metadata": metadata},
        )
    except Exception as exc:  # noqa: BLE001
        raise PaidfInputError(
            f"could not stage verified PAIDF input at {uri}: {exc}"
        ) from exc


def _upload_json(
    client: Any,
    payload: dict[str, Any],
    uri: str,
    tmp: Path,
    *,
    persist_digest_metadata: bool = True,
) -> None:
    local = tmp / "provenance.json"
    local.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _sha256(local)
    _stage_file(
        client,
        local,
        uri,
        digest,
        persist_digest_metadata=persist_digest_metadata,
    )


def _verify_file(path: Path, digest: str, size: int, *, context: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise PaidfInputError(
            f"{context} byte-size mismatch: expected {size}, got {actual_size} ({path})"
        )
    _verify_digest(path, digest, context=context)


def _verify_digest(path: Path, digest: str, *, context: str) -> None:
    actual = _sha256(path)
    if not digest or actual != digest:
        raise PaidfInputError(
            f"{context} SHA-256 mismatch: expected {digest or '<missing>'}, got {actual} ({path})"
        )


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise PaidfInputError(f"expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _missing_s3(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    code = str((response.get("Error") or {}).get("Code") or "")
    return code in {"404", "NoSuchKey", "NotFound"}
