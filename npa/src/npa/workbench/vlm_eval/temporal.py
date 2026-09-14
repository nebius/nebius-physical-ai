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

RUBRIC_VERSION = "manipulation-visible-events-v1"
RUBRIC = """Judge only the supplied timestamped images of one continuous robot episode.
Identify the requested target by its shape and color; other parts are distractors.
For lifted, require the target visibly separated from its initial support and held
by the gripper in at least two frames. Gripper movement alone is insufficient.
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
    failure_modes: list[Literal["missed_grasp", "slip_or_drop", "unstable_hold", "wrong_object",
                                "fixture_contact", "occluded", "none"]] = Field(min_length=1)
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
        raise ValueError("Temporal VLM evaluation requires valid timing and at least two frames")
    tail_count = count // 2
    tail_start = max(0, length - 1 - round(0.5 * fps))
    broad = np.linspace(0, length - 1, count - tail_count, dtype=int)
    tail = np.linspace(tail_start, length - 1, tail_count, dtype=int)
    return sorted(set(broad.tolist() + tail.tolist()))


def _frame_content(rgb_path: Path, output: Path, fps: float, count: int) -> tuple[list, list]:
    pixels = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Temporal VLM input must be a real uint8 RGB trajectory")
    frames, content = [], []
    for index in sample_indices(len(pixels), fps, count):
        path = output / f"frame-{index:06d}.jpg"
        Image.fromarray(pixels[index]).save(path, quality=95)
        data = path.read_bytes()
        timestamp = index / fps
        frames.append({"index": index, "seconds": timestamp, "file": path.name,
                       "sha256": hashlib.sha256(data).hexdigest()})
        content.extend([{"type": "text", "text": f"Frame {index}; time {timestamp:.6f} seconds"},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}}])
    return frames, content


def _validate_response(response: dict, model: str, frames: list[dict]) -> dict:
    choices = response.get("choices")
    if response.get("model") != model or not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Temporal VLM response has a substituted model or invalid choice coverage")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError("Temporal VLM response did not finish completely")
    verdict = ManipulationVerdict.model_validate_json(choices[0]["message"]["content"])
    available = {frame["index"] for frame in frames}
    for event in (verdict.lifted, verdict.held_at_end, verdict.scene_disturbed):
        if len(event.frames) != len(set(event.frames)) or not set(event.frames).issubset(available):
            raise ValueError("Temporal VLM cited a missing or duplicate frame")
    if verdict.lifted.verdict == "yes" and len(verdict.lifted.frames) < 2:
        raise ValueError("Visual lift requires at least two supporting frames")
    if verdict.held_at_end.verdict == "yes":
        final_frames = {frame["index"] for frame in frames[-2:]}
        if verdict.lifted.verdict != "yes" or not final_frames.issubset(verdict.held_at_end.frames):
            raise ValueError("Visual hold requires lift and both final sampled frames")
    if "none" in verdict.failure_modes and len(verdict.failure_modes) != 1:
        raise ValueError("Temporal VLM returned contradictory failure tags")
    return verdict.model_dump()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def judge_manipulation(*, rgb_path: Path, output: Path, task: str, fps: float,
                      frame_count: int, model: str, client: TokenFactoryClient) -> dict:
    """Judge one unlabeled rollout and retain frame, prompt, response, and usage evidence.

    Args:
        rgb_path: Actual synchronized RGB numpy trajectory.
        output: Empty directory for sampled frames and judge evidence.
        task: Target identification and visible task instruction, without outcome labels.
        fps: Measured capture frequency.
        frame_count: Sealed frame sampling resolution.
        model: Exact Token Factory model ID verified by the caller.
        client: Shared authenticated Token Factory client.
    Returns:
        Validated visual events, provenance, and measured provider accounting.
    Raises:
        ValueError: Input pixels, returned model, structured verdict, or evidence references are invalid.
        TokenFactoryError: Hosted inference fails.
        OSError: Evidence cannot be written.
    """
    output.mkdir(parents=True)
    frames, content = _frame_content(rgb_path, output, fps, frame_count)
    schema = ManipulationVerdict.model_json_schema()
    schema["$defs"]["VisualEvent"]["properties"]["frames"]["items"]["enum"] = [frame["index"] for frame in frames]
    prompt = RUBRIC + "\nTask: " + task + "\nJSON schema: " + json.dumps(schema)
    _write(output / "request.json", {"model": model, "temperature": 0, "prompt": prompt,
                                    "rubric_version": RUBRIC_VERSION, "frames": frames})
    response = client.chat_completion(model=model, temperature=0,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, *content]}])
    _write(output / "response.json", response)
    verdict = _validate_response(response, model, frames)
    result = {"schema": "npa.vlm.manipulation.v1", "backend": "token_factory", "model": model,
              "rubric_version": RUBRIC_VERSION, "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
              "frames": frames, "verdict": verdict, "request_id": response.get("id"),
              "usage": response.get("usage"), "cost": response.get("cost"),
              "transport": client.last_request_metrics,
              "finish_reason": response["choices"][0]["finish_reason"]}
    _write(output / "verdict.json", result)
    return result
