"""VLM evaluation helpers for sim-to-real pipeline validation."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Iterator

import httpx
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


DEFAULT_BACKEND = "self-hosted"
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_ENDPOINT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_FRAME_SELECTION = "keyframes"
DEFAULT_MAX_FRAMES = 4
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_API_KEY_ENV = "VLM_EVAL_API_KEY"
DEFAULT_RUBRIC = (
    "Score whether the rollout completes the requested physical task. "
    "Use 1.0 only for clear task completion, 0.0 for clear failure, and "
    "intermediate values for partial progress. Penalize unsafe, incomplete, "
    "or ambiguous outcomes."
)
RESULT_FILENAME = "vlm_eval_stub.json"
SUPPORTED_BACKENDS = ("self-hosted", "api", "stub")
SUPPORTED_FRAME_SELECTIONS = ("final", "keyframes", "sequence")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class VlmEvalError(ValueError):
    """Raised when a VLM evaluation request is invalid."""


@dataclass(frozen=True)
class VlmEvalResult:
    status: str
    backend: str
    input_path: str
    output_path: str
    result_uri: str
    task: str
    model: str
    score: float
    success_threshold: float
    passed: bool
    generated_at: str
    frame_selection: str = DEFAULT_FRAME_SELECTION
    frame_count: int = 0
    rationale: str = ""


@dataclass(frozen=True)
class VlmStructuredResponse:
    success: bool
    score: float
    rationale: str


@dataclass(frozen=True)
class SelectedFrame:
    label: str
    media_type: str
    data: bytes


__all__ = [
    "VlmEvalResult",
    "VlmStructuredResponse",
    "evaluate_stub",
    "evaluate_vlm",
    "parse_structured_response",
    "result_uri_for",
    "select_rollout_frames",
    "write_result",
]


def evaluate_vlm(
    *,
    input_path: str,
    output_path: str,
    task: str = "sim-to-real",
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
    endpoint_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    rubric: str = DEFAULT_RUBRIC,
    rubric_path: str = "",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    score: float | None = None,
) -> VlmEvalResult:
    """Evaluate rollout frames with a VLM and return a scalar score in [0, 1]."""

    _validate_common(
        input_path=input_path,
        output_path=output_path,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        max_frames=max_frames,
        timeout_s=timeout_s,
    )
    backend = _normalize_backend(backend)
    if backend == "stub":
        return evaluate_stub(
            input_path=input_path,
            output_path=output_path,
            task=task,
            model=model or "vlm-eval-stub",
            success_threshold=success_threshold,
            frame_selection=frame_selection,
            score=score,
        )

    effective_model = model or DEFAULT_MODEL
    effective_rubric = _load_rubric(rubric=rubric, rubric_path=rubric_path)
    if score is not None:
        _validate_score_override(score)
        structured = VlmStructuredResponse(
            success=score >= success_threshold,
            score=score,
            rationale="Score override supplied; VLM call skipped.",
        )
        frame_count = 0
        effective_task = task
    else:
        with _materialized_input(input_path) as local_input:
            effective_task = _resolve_task_text(local_input, task)
            frames = select_rollout_frames(
                local_input,
                frame_selection=frame_selection,
                max_frames=max_frames,
            )
            prompt = _build_prompt(
                task=effective_task,
                rubric=effective_rubric,
                frame_selection=frame_selection,
                frame_count=len(frames),
            )
            structured = _call_openai_compatible(
                backend=backend,
                model=effective_model,
                endpoint_url=endpoint_url,
                api_key_env=api_key_env,
                prompt=prompt,
                frames=frames,
                timeout_s=timeout_s,
            )
            frame_count = len(frames)

    return _result_from_structured(
        backend=backend,
        input_path=input_path,
        output_path=output_path,
        task=effective_task,
        model=effective_model,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        frame_count=frame_count,
        structured=structured,
    )


def evaluate_stub(
    *,
    input_path: str,
    output_path: str,
    task: str = "sim-to-real",
    model: str = "vlm-eval-stub",
    success_threshold: float = 0.8,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    score: float | None = None,
) -> VlmEvalResult:
    """Return deterministic schema-compatible metrics without calling a VLM."""

    _validate_common(
        input_path=input_path,
        output_path=output_path,
        success_threshold=success_threshold,
        frame_selection=frame_selection,
        max_frames=DEFAULT_MAX_FRAMES,
        timeout_s=DEFAULT_TIMEOUT_S,
    )
    effective_score = _deterministic_score(input_path, task, model) if score is None else score
    _validate_score_override(effective_score)
    passed = effective_score >= success_threshold
    return VlmEvalResult(
        status="passed" if passed else "needs_iteration",
        backend="stub",
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        task=task,
        model=model,
        score=round(effective_score, 4),
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_selection=frame_selection,
        frame_count=0,
        rationale="Deterministic compatibility score.",
    )


def select_rollout_frames(
    input_path: str | Path,
    *,
    frame_selection: str = DEFAULT_FRAME_SELECTION,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[SelectedFrame]:
    """Load selected rollout frames from image files, numpy episodes, or video files."""

    frame_selection = _normalize_frame_selection(frame_selection)
    if max_frames <= 0:
        raise VlmEvalError("--max-frames must be positive")

    path = Path(input_path)
    image_frames = _frames_from_images(path, frame_selection=frame_selection, max_frames=max_frames)
    if image_frames:
        return image_frames

    numpy_frames = _frames_from_numpy(path, frame_selection=frame_selection, max_frames=max_frames)
    if numpy_frames:
        return numpy_frames

    video_frames = _frames_from_videos(path, frame_selection=frame_selection, max_frames=max_frames)
    if video_frames:
        return video_frames

    raise VlmEvalError(
        f"No rollout frames found in {path}. Expected image files, RGB .npy/.npz arrays, or videos."
    )


def parse_structured_response(text: str) -> VlmStructuredResponse:
    """Parse a VLM JSON response and clamp its score into [0, 1]."""

    payload = _load_json_object(text)
    if "score" not in payload:
        raise VlmEvalError("VLM response JSON must include score")
    if "rationale" not in payload:
        raise VlmEvalError("VLM response JSON must include rationale")
    score = _clamp_score(payload["score"])
    success = _coerce_bool(payload.get("success", score >= 0.5))
    return VlmStructuredResponse(
        success=success,
        score=score,
        rationale=str(payload["rationale"]),
    )


def result_uri_for(output_path: str) -> str:
    """Return the JSON artifact URI for an output path."""

    if output_path.endswith(".json"):
        return output_path
    return output_path.rstrip("/") + f"/{RESULT_FILENAME}"


def write_result(
    payload: dict[str, Any],
    *,
    result_uri: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a VLM eval result to local disk or S3."""

    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-") as tmp:
            local_path = Path(tmp) / RESULT_FILENAME
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    if path.suffix != ".json":
        path = path / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _result_from_structured(
    *,
    backend: str,
    input_path: str,
    output_path: str,
    task: str,
    model: str,
    success_threshold: float,
    frame_selection: str,
    frame_count: int,
    structured: VlmStructuredResponse,
) -> VlmEvalResult:
    score = round(_clamp_score(structured.score), 4)
    passed = score >= success_threshold
    return VlmEvalResult(
        status="passed" if passed else "needs_iteration",
        backend=backend,
        input_path=input_path,
        output_path=output_path,
        result_uri=result_uri_for(output_path),
        task=task,
        model=model,
        score=score,
        success_threshold=success_threshold,
        passed=passed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_selection=frame_selection,
        frame_count=frame_count,
        rationale=structured.rationale,
    )


def _validate_common(
    *,
    input_path: str,
    output_path: str,
    success_threshold: float,
    frame_selection: str,
    max_frames: int,
    timeout_s: float,
) -> None:
    if not input_path:
        raise VlmEvalError("input_path is required")
    if not output_path:
        raise VlmEvalError("output_path is required")
    if not 0.0 <= success_threshold <= 1.0:
        raise VlmEvalError("--success-threshold must be between 0 and 1")
    _normalize_frame_selection(frame_selection)
    if max_frames <= 0:
        raise VlmEvalError("--max-frames must be positive")
    if timeout_s <= 0:
        raise VlmEvalError("--timeout-s must be positive")


def _validate_score_override(score: float) -> None:
    if not 0.0 <= float(score) <= 1.0:
        raise VlmEvalError("--score must be between 0 and 1")


def _normalize_backend(backend: str) -> str:
    value = (backend or DEFAULT_BACKEND).strip().lower()
    if value not in SUPPORTED_BACKENDS:
        allowed = ", ".join(SUPPORTED_BACKENDS)
        raise VlmEvalError(f"--backend must be one of: {allowed}")
    return value


def _normalize_frame_selection(frame_selection: str) -> str:
    value = (frame_selection or DEFAULT_FRAME_SELECTION).strip().lower()
    if value not in SUPPORTED_FRAME_SELECTIONS:
        allowed = ", ".join(SUPPORTED_FRAME_SELECTIONS)
        raise VlmEvalError(f"--frame-selection must be one of: {allowed}")
    return value


def _load_rubric(*, rubric: str, rubric_path: str) -> str:
    if rubric_path:
        path = Path(rubric_path)
        if not path.exists():
            raise VlmEvalError(f"--rubric-path does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()
    return (rubric or DEFAULT_RUBRIC).strip()


@contextmanager
def _materialized_input(input_path: str) -> Iterator[Path]:
    if not input_path.startswith("s3://"):
        yield Path(input_path)
        return

    from npa.clients.storage import StorageClient

    with tempfile.TemporaryDirectory(prefix="npa-vlm-eval-input-") as tmp:
        local = StorageClient.from_environment().download_path(input_path, tmp)
        yield Path(local)


def _resolve_task_text(local_input: Path, task: str) -> str:
    if task and task != "sim-to-real":
        return task

    for candidate in (
        local_input / "meta" / "tasks.parquet",
        local_input.parent / "meta" / "tasks.parquet",
    ):
        if not candidate.exists():
            continue
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(candidate)
            if "task" in table.column_names and table.num_rows:
                value = table.column("task")[0].as_py()
                if value:
                    return str(value)
        except Exception:
            continue

    for candidate in (
        local_input / "meta" / "info.json",
        local_input / "info.json",
        local_input / "manifest.json",
    ):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("task", "instruction", "description"):
            value = payload.get(key)
            if value:
                return str(value)
    return task


def _build_prompt(
    *,
    task: str,
    rubric: str,
    frame_selection: str,
    frame_count: int,
) -> str:
    return "\n".join(
        [
            "You are scoring a robot rollout from visual evidence.",
            f"Task/instruction: {task}",
            f"Rubric: {rubric}",
            f"Frame selection: {frame_selection}; frames supplied: {frame_count}.",
            "Return only a JSON object with this schema:",
            '{"success": boolean, "score": number between 0 and 1, "rationale": string}',
            "The score is the only downstream contract; make it repeatable and calibrated.",
        ]
    )


def _call_openai_compatible(
    *,
    backend: str,
    model: str,
    endpoint_url: str,
    api_key_env: str,
    prompt: str,
    frames: list[SelectedFrame],
    timeout_s: float,
) -> VlmStructuredResponse:
    url = _chat_completions_url(_resolve_endpoint_url(backend=backend, endpoint_url=endpoint_url))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        encoded = base64.b64encode(frame.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{frame.media_type};base64,{encoded}"},
            }
        )

    headers = {"Content-Type": "application/json"}
    api_key = _resolve_api_key(backend=backend, api_key_env=api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, headers=headers, json=request)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise VlmEvalError(f"VLM backend request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VlmEvalError("VLM backend returned non-JSON response") from exc

    try:
        message = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VlmEvalError("VLM backend response missing choices[0].message.content") from exc
    return parse_structured_response(str(message))


def _resolve_endpoint_url(*, backend: str, endpoint_url: str) -> str:
    if endpoint_url:
        return endpoint_url
    if backend == "api":
        return (
            os.environ.get("VLM_EVAL_API_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_ENDPOINT_URL
        )
    return (
        os.environ.get("VLM_EVAL_ENDPOINT_URL")
        or os.environ.get("VLM_EVAL_BASE_URL")
        or DEFAULT_ENDPOINT_URL
    )


def _resolve_api_key(*, backend: str, api_key_env: str) -> str:
    key = os.environ.get(api_key_env or DEFAULT_API_KEY_ENV, "")
    if backend == "api":
        key = key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise VlmEvalError(
                f"--backend api requires an API key in {api_key_env or DEFAULT_API_KEY_ENV} or OPENAI_API_KEY"
            )
    return key


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _frames_from_images(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    image_paths = _discover_image_paths(path)
    if not image_paths:
        return []
    indices = _selected_indices(len(image_paths), frame_selection=frame_selection, max_frames=max_frames)
    return [
        SelectedFrame(
            label=image_paths[index].name,
            media_type="image/png",
            data=_image_file_to_png(image_paths[index]),
        )
        for index in indices
    ]


def _discover_image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in file.relative_to(path).parts)
    )


def _frames_from_numpy(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    arrays = _discover_numpy_arrays(path)
    for label, array in arrays:
        if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] == 0:
            continue
        indices = _selected_indices(array.shape[0], frame_selection=frame_selection, max_frames=max_frames)
        return [
            SelectedFrame(
                label=f"{label}:{index}",
                media_type="image/png",
                data=_array_frame_to_png(array[index]),
            )
            for index in indices
        ]
    return []


def _discover_numpy_arrays(path: Path) -> list[tuple[str, np.ndarray]]:
    candidates: list[Path] = []
    if path.is_file() and path.suffix.lower() in {".npy", ".npz"}:
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in {".npy", ".npz"}
        )
    candidates.sort(key=_numpy_preference_key)

    arrays: list[tuple[str, np.ndarray]] = []
    for candidate in candidates:
        try:
            if candidate.suffix.lower() == ".npz":
                bundle = np.load(candidate)
                for key in bundle.files:
                    arrays.append((f"{candidate.name}:{key}", np.asarray(bundle[key])))
            else:
                arrays.append((candidate.name, np.load(candidate, mmap_mode="r")))
        except (OSError, ValueError):
            continue
    return arrays


def _numpy_preference_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "workspace" in name:
        return (0, name)
    if "image" in name or "frame" in name:
        return (1, name)
    if "wrist" in name:
        return (2, name)
    return (3, name)


def _frames_from_videos(
    path: Path,
    *,
    frame_selection: str,
    max_frames: int,
) -> list[SelectedFrame]:
    video_paths = _discover_video_paths(path)
    if not video_paths:
        return []
    if not shutil.which("ffmpeg"):
        raise VlmEvalError("Video rollout input requires ffmpeg to extract frames")

    video_path = video_paths[0]
    with tempfile.TemporaryDirectory(prefix="npa-vlm-video-") as tmp:
        output_dir = Path(tmp)
        count = _video_frame_count(video_path)
        if count:
            indices = _selected_indices(count, frame_selection=frame_selection, max_frames=max_frames)
            _extract_video_indices(video_path, output_dir, indices)
        elif frame_selection == "final":
            _extract_final_video_frame(video_path, output_dir)
        else:
            _extract_video_sample(video_path, output_dir, max_frames=max_frames)
        return [
            SelectedFrame(
                label=frame.name,
                media_type="image/png",
                data=_image_file_to_png(frame),
            )
            for frame in sorted(output_dir.glob("frame-*.png"))
        ]


def _discover_video_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in VIDEO_SUFFIXES)


def _video_frame_count(video_path: Path) -> int | None:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    for key in ("nb_read_frames", "nb_frames"):
        value = streams[0].get(key)
        if value and str(value).isdigit():
            count = int(value)
            if count > 0:
                return count
    return None


def _extract_video_indices(video_path: Path, output_dir: Path, indices: list[int]) -> None:
    if not indices:
        return
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        str(output_dir / "frame-%03d.png"),
    ]
    _run_ffmpeg(cmd)


def _extract_final_video_frame(video_path: Path, output_dir: Path) -> None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_dir / "frame-001.png"),
    ]
    _run_ffmpeg(cmd)


def _extract_video_sample(video_path: Path, output_dir: Path, *, max_frames: int) -> None:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-frames:v",
        str(max_frames),
        str(output_dir / "frame-%03d.png"),
    ]
    _run_ffmpeg(cmd)


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {exc}") from exc
    if proc.returncode != 0:
        raise VlmEvalError(f"ffmpeg frame extraction failed: {proc.stderr[-500:]}")


def _selected_indices(count: int, *, frame_selection: str, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if frame_selection == "final":
        return [count - 1]
    selected = min(max_frames, count)
    if selected == 1:
        return [count - 1]
    return sorted({round(i * (count - 1) / (selected - 1)) for i in range(selected)})


def _image_file_to_png(path: Path) -> bytes:
    with Image.open(path) as image:
        return _pil_image_to_png(image)


def _array_frame_to_png(frame: np.ndarray) -> bytes:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    return _pil_image_to_png(image)


def _pil_image_to_png(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise VlmEvalError("VLM response did not contain a JSON object") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VlmEvalError("VLM response JSON could not be parsed") from exc
    if not isinstance(payload, dict):
        raise VlmEvalError("VLM response JSON must be an object")
    return payload


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise VlmEvalError("VLM response score must be numeric") from exc
    if math.isnan(score) or math.isinf(score):
        raise VlmEvalError("VLM response score must be finite")
    return max(0.0, min(1.0, score))


def _deterministic_score(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
