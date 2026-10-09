"""Describe videos and judge paired samples through explicit hosted models."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import av

from npa.clients.token_factory import TokenFactoryClient, split_reasoning


def sample_video(path: Path, samples: int) -> tuple[list[dict], dict]:
    """Decode a full video and sample evenly across its frame timeline.

    Args:
        path: Local video.
        samples: Number of visual observations.
    Returns:
        Image content blocks and decoded metadata.
    Raises:
        ValueError: No video frames or invalid sampling count.
    """
    if samples < 2:
        raise ValueError("At least two temporal observations are required")
    with av.open(str(path)) as container:
        count = sum(1 for _ in container.decode(video=0))
    if count < 2:
        raise ValueError("Video must decode at least two frames")
    indices = sorted({round(i * (count - 1) / (samples - 1)) for i in range(samples)})
    blocks, timestamps = [], []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in indices:
                continue
            encoded = _encode_frame(frame)
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
            timestamps.append(frame.time)
    return blocks, {
        "frame_count": count,
        "sample_indices": indices,
        "sample_times": timestamps,
    }


def completion(
    client: TokenFactoryClient, model: str, content: list[dict]
) -> tuple[str, dict]:
    """Request an explicit model and retain measured provider provenance.

    Args:
        client: Authenticated hosted client.
        model: Exact model ID.
        content: Text and image blocks.
    Returns:
        Visible answer and response metadata.
    Raises:
        ValueError: Output is empty, truncated, or from another model.
    """
    response = client.chat_completion(
        model=model, messages=[{"role": "user", "content": content}]
    )
    choice = response["choices"][0]
    text, _ = split_reasoning(choice["message"])
    if (
        response.get("model") != model
        or choice.get("finish_reason") != "stop"
        or not text.strip()
    ):
        raise ValueError(
            "Hosted response model, completion status, or visible answer is invalid"
        )
    return text.strip(), {
        "model": model,
        "request_id": response.get("id"),
        "usage": response.get("usage"),
        "finish_reason": choice["finish_reason"],
    }


def text_block(text: str) -> dict:
    """Construct a hosted text content block.

    Args:
        text: Prompt content.
    Returns:
        Content block.
    Raises:
        None.
    """
    return {"type": "text", "text": text}


def parse_review(text: str, threshold: float) -> dict:
    """Require an explicit boolean and a finite score before accepting media.

    Args:
        text: Judge JSON with passed, score, and reason.
        threshold: Required score between zero and one.
    Returns:
        Validated review and effective acceptance.
    Raises:
        ValueError: Malformed or incomplete judgment.
    """
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {"passed", "score", "reason"}:
        raise ValueError("Judge must return passed, score, and reason")
    score = result["score"]
    valid_score = (
        type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1
    )
    if type(result["passed"]) is not bool or not valid_score:
        raise ValueError("Judge returned an invalid pass flag or score")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("Judge must explain its decision")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid acceptance threshold")
    return {**result, "accepted": result["passed"] and score >= threshold}


def _encode_frame(frame) -> str:
    picture = frame.to_image()
    picture.thumbnail((768, 768))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()
