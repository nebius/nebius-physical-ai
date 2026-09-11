"""Typed sample and artifact contracts for guarded native Cosmos 3 edge transfer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.workflows.paidf_cosmos3_media import probe_video, validate_reference


@dataclass(frozen=True)
class TransferSettings:
    """Sampling controls supported by the pinned native transfer implementation.

    Args:
        fps: Prepared and generated frame rate, an integer from 10 through 30.
        chunk_frames: Native 4k+1 generation window, from 9 through 297 frames.
        control_guidance: Native structural guidance, greater than 0 and at most 10.
    Returns:
        Immutable settings; call validate before model work.
    Raises:
        None; validation is explicit.
    """

    fps: int = 24
    chunk_frames: int = 93
    control_guidance: float = 1.5

    def validate(self) -> None:
        """Validate controls before any model work.

        Args:
            None.
        Returns:
            None.
        Raises:
            ValueError: A control is outside the model's supported contract.
        """
        if type(self.fps) is not int or not 10 <= self.fps <= 30:
            raise ValueError("conditioning_fps must be an integer from 10 through 30")
        if type(self.chunk_frames) is not int or not 9 <= self.chunk_frames <= 297 or (self.chunk_frames - 1) % 4:
            raise ValueError("transfer_chunk_frames must be 4k+1 between 9 and 297")
        if (type(self.control_guidance) not in {int, float}
                or not math.isfinite(self.control_guidance) or not 0 < self.control_guidance <= 10):
            raise ValueError("control_guidance must be finite, positive and at most 10")


def transfer_sample(base: dict[str, Any], settings: TransferSettings, output: Path, seed: int) -> dict[str, Any]:
    """Configure native full-source transfer using a prepared local reference.

    Args:
        base: Validated video2video sample with its local vision_path.
        settings: Supported transfer controls.
        output: Private generation directory.
        seed: Explicit seed for the first native transfer chunk.
    Returns:
        Native sample arguments with an exact complete-source frame bound.
    Raises:
        ValueError: Mode or controls are unsupported.
        VideoAlignmentError: The source is not a valid prepared reference.
        OSError: The source or media executables are unavailable.
    """
    settings.validate()
    if base["model_mode"] != "video2video":
        raise ValueError("Structural transfer requires video2video mode")
    timeline = probe_video(Path(base["vision_path"]))
    validate_reference(timeline, settings.fps)
    return {**base, "resolution": "480", "aspect_ratio": "16,9", "fps": settings.fps,
            "num_frames": settings.chunk_frames, "seed": seed, "shift": 10.0,
            "edge": {"control_path": str(output / "controls" / f"{base['name']}.mkv"),
                     "preset_edge_threshold": "medium"},
            "control_guidance": settings.control_guidance,
            "num_video_frames_per_chunk": settings.chunk_frames,
            "num_conditional_frames": 5, "num_first_chunk_conditional_frames": 1,
            "max_frames": timeline["decoded_frames"], "share_vision_temporal_positions": True,
            "show_input": False, "show_control_condition": False}


def transfer_artifact(sample_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Accept only the named generated video with completed model guardrails.

    Args:
        sample_dir: Native sample output directory.
    Returns:
        Generated video path and guardrail/control evidence.
    Raises:
        ValueError: Execution, guardrails or output identity cannot be established.
        OSError: Required artifacts cannot be read.
    """
    outputs = json.loads((sample_dir / "sample_outputs.json").read_text())
    evidence = json.loads((sample_dir / "transfer_evidence.json").read_text())
    artifact = sample_dir / "vision.mp4"
    files = [str(item) for entry in outputs.get("outputs", []) for item in entry.get("files", [])]
    if (outputs.get("status") != "success" or str(artifact) not in files
            or not artifact.is_file() or artifact.stat().st_size <= 0
            or evidence.get("schema") != "npa.cosmos3.structural-transfer.v1"
            or evidence.get("text_guardrail_passed") is not True
            or evidence.get("video_guardrail_passed") is not True
            or evidence.get("guardrail_postprocessing_applied") is not True):
        raise ValueError("Structural transfer lacks successful guarded generation evidence")
    return artifact, evidence
