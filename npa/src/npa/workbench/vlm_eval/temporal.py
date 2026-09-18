"""Judge timestamped manipulation frames through Token Factory with auditable visual claims."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from npa.clients.token_factory import TokenFactoryClient

RUBRIC_VERSION = "manipulation-visible-events-v2"
RUBRIC = """Judge only the supplied timestamped images of one continuous robot episode.
Identify the requested target by its shape and color; other parts are distractors.
For lifted, judge visible object elevation: require the target clearly elevated
above its initial support in at least two supplied frames. This event does not
require a grasp; a tossed object also counts as elevated. Gripper movement alone
is insufficient. Judge grasp and retention separately using held_at_end.
For held_at_end, require the target still visibly grasped and approximately steady
in the last two supplied frames. This is a visual observation, not proof of exact
position, speed, duration between samples, force, or physical robot readiness.
For scene_disturbed, look for unintended contact with the tray/fixture or movement
of a distractor. Mark uncertain when occlusion or sampling prevents a conclusion.
Every verdict cites supplied frame indices. Explain failures concretely. Do not
infer missing events, policy training status, or success from the task description.
Return only the requested JSON object. No markdown, extra keys, or prose outside JSON.
"""


class VisualEvent(BaseModel):
    """A visible event with references to the actual supplied frames."""

    model_config = ConfigDict(extra="forbid", strict=True)
    verdict: Literal["yes", "no", "uncertain"]
    frames: list[int] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ManipulationVerdict(BaseModel):
    """Independent visual lift, final hold, and scene-disturbance assessments."""

    model_config = ConfigDict(extra="forbid", strict=True)
    lifted: VisualEvent
    held_at_end: VisualEvent
    scene_disturbed: VisualEvent
    failure_modes: list[
        Literal[
            "missed_grasp",
            "slip_or_drop",
            "unstable_hold",
            "wrong_object",
            "fixture_contact",
            "occluded",
            "none",
        ]
    ] = Field(min_length=1)
    rationale: str = Field(min_length=1)


def sample_indices(length: int, fps: float, count: int) -> list[int]:
    """Sample the whole episode and a denser final half-second without using outcomes.

    Args:
        length: Actual number of captured frames.
        fps: Measured camera/control frequency.
        count: Sealed visual sampling resolution.
    Returns:
        Sorted distinct indices including the initial and final frame.
    Raises:
        ValueError: Timing, frame count, or sampling resolution is invalid.
    """
    if length < 2 or not np.isfinite(fps) or fps <= 0 or count < 4:
        raise ValueError(
            "Temporal VLM evaluation requires valid timing and at least two frames"
        )
    tail_count = count // 2
    tail_start = max(0, length - 1 - round(0.5 * fps))
    broad = np.linspace(0, length - 1, count - tail_count, dtype=int)
    tail = np.linspace(tail_start, length - 1, tail_count, dtype=int)
    return sorted(set(broad.tolist() + tail.tolist()))


def _frame_content(
    rgb_path: Path, output: Path, fps: float, count: int
) -> tuple[list, list]:
    pixels = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Temporal VLM input must be a real uint8 RGB trajectory")
    frames, content = [], []
    for index in sample_indices(len(pixels), fps, count):
        path = output / f"frame-{index:06d}.jpg"
        Image.fromarray(pixels[index]).save(path, quality=95)
        data = path.read_bytes()
        timestamp = index / fps
        frames.append(
            {
                "index": index,
                "seconds": timestamp,
                "file": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Frame {index}; time {timestamp:.6f} seconds",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(data).decode("ascii")
                    },
                },
            ]
        )
    return frames, content


def _validate_response(response: dict, model: str, frames: list[dict]) -> dict:
    choices = response.get("choices")
    if (
        response.get("model") != model
        or not isinstance(choices, list)
        or len(choices) != 1
    ):
        raise ValueError(
            "Temporal VLM response has a substituted model or invalid choice coverage"
        )
    if not isinstance(choices[0], dict) or not isinstance(
        choices[0].get("message"), dict
    ):
        raise ValueError("Temporal VLM response has an invalid message")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError("Temporal VLM response did not finish completely")
    if not isinstance(choices[0]["message"].get("content"), str):
        raise ValueError("Temporal VLM response has no textual judgment")
    verdict = ManipulationVerdict.model_validate_json(choices[0]["message"]["content"])
    available = {frame["index"] for frame in frames}
    for event in (verdict.lifted, verdict.held_at_end, verdict.scene_disturbed):
        if len(event.frames) != len(set(event.frames)) or not set(
            event.frames
        ).issubset(available):
            raise ValueError("Temporal VLM cited a missing or duplicate frame")
    if verdict.lifted.verdict == "yes" and len(verdict.lifted.frames) < 2:
        raise ValueError("Visual lift requires at least two supporting frames")
    if verdict.held_at_end.verdict == "yes":
        final_frames = {frame["index"] for frame in frames[-2:]}
        if verdict.lifted.verdict != "yes" or not final_frames.issubset(
            verdict.held_at_end.frames
        ):
            raise ValueError("Visual hold requires lift and both final sampled frames")
    if "none" in verdict.failure_modes and len(verdict.failure_modes) != 1:
        raise ValueError("Temporal VLM returned contradictory failure tags")
    return verdict.model_dump()


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _replay_response(
    previous: Path, output: Path, request: dict
) -> tuple[dict, dict | None]:
    if json.loads((previous / "request.json").read_text()) != request:
        raise ValueError(
            "Recorded VLM request differs from the current pixels, prompt, or model"
        )
    for frame in request["frames"]:
        if (
            hashlib.sha256((previous / frame["file"]).read_bytes()).hexdigest()
            != frame["sha256"]
        ):
            raise ValueError("Recorded VLM frame bytes differ from the supplied image")
    raw = (previous / "response.json").read_bytes()
    response = json.loads(raw)
    if not isinstance(response, dict):
        raise ValueError("Recorded VLM response must be an object")
    (output / "response.json").write_bytes(raw)
    receipt = previous / "verdict.json"
    transport = (
        json.loads(receipt.read_text()).get("transport") if receipt.is_file() else None
    )
    return response, transport


def _judgment_result(
    response: dict, model: str, frames: list[dict], transport: dict | None
) -> dict:
    from pydantic import ValidationError

    verdict, error = None, None
    try:
        verdict = _validate_response(response, model, frames)
    except ValidationError as invalid:
        error = {
            "type": "schema",
            "message": "; ".join(item["type"] for item in invalid.errors()),
        }
    except ValueError as invalid:
        error = {"type": "evidence_contract", "message": str(invalid)}
    choices = response.get("choices")
    finish = (
        choices[0].get("finish_reason")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else None
    )
    return {
        "schema": "npa.vlm.manipulation.v1",
        "backend": "token_factory",
        "model": model,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "status": "valid" if error is None else "invalid_response",
        "validation_error": error,
        "frames": frames,
        "verdict": verdict,
        "request_id": response.get("id"),
        "usage": response.get("usage"),
        "cost": response.get("cost"),
        "transport": transport,
        "finish_reason": finish,
    }


def _request_content(
    rgb_path: Path, output: Path, task: str, fps: float, frame_count: int, model: str
) -> tuple[dict, list]:
    frames, content = _frame_content(rgb_path, output, fps, frame_count)
    schema = ManipulationVerdict.model_json_schema()
    schema["$defs"]["VisualEvent"]["properties"]["frames"]["items"]["enum"] = [
        frame["index"] for frame in frames
    ]
    prompt = RUBRIC + "\nTask: " + task + "\nJSON schema: " + json.dumps(schema)
    request = {
        "model": model,
        "temperature": 0,
        "prompt": prompt,
        "rubric_version": RUBRIC_VERSION,
        "frames": frames,
    }
    _write(output / "request.json", request)
    return request, [{"type": "text", "text": prompt}, *content]


def judge_manipulation(
    *,
    rgb_path: Path,
    output: Path,
    task: str,
    fps: float,
    frame_count: int,
    model: str,
    client: TokenFactoryClient,
    previous: Path | None = None,
) -> dict:
    """Judge one unlabeled rollout and retain frame, prompt, response, and usage evidence.

    Args:
        rgb_path: Actual synchronized RGB numpy trajectory.
        output: Empty directory for sampled frames and judge evidence.
        task: Target identification and visible task instruction, without outcome labels.
        fps: Measured capture frequency.
        frame_count: Sealed frame sampling resolution.
        model: Exact Token Factory model ID verified by the caller.
        client: Shared authenticated Token Factory client.
        previous: Optional prior request/response and JPEG evidence, revalidated without another model call.
    Returns:
        Visual events or an explicit invalid response, provenance, and available provider accounting.
    Raises:
        ValueError: Input pixels or recorded request/frame evidence is invalid.
        TokenFactoryError: Hosted inference fails.
        OSError: Evidence cannot be written.
    """
    output.mkdir(parents=True)
    request, content = _request_content(rgb_path, output, task, fps, frame_count, model)
    if previous is not None:
        response, transport = _replay_response(previous, output, request)
    else:
        response = client.chat_completion(
            model=model, temperature=0, messages=[{"role": "user", "content": content}]
        )
        transport = client.last_request_metrics
        _write(output / "response.json", response)
    result = _judgment_result(response, model, request["frames"], transport)
    result["response_source"] = "replayed" if previous is not None else "live"
    result["response_sha256"] = hashlib.sha256(
        (output / "response.json").read_bytes()
    ).hexdigest()
    _write(output / "verdict.json", result)
    return result
