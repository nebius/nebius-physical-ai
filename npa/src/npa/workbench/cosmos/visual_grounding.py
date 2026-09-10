"""Bind visual critiques to recorded simulation times before they shape training."""
from __future__ import annotations

from pathlib import PurePosixPath
import math
from typing import Any

VISUAL_GROUNDING_SCHEMA = "npa.sim2real.visual_grounding.v1"
HOSTED_EVAL_SCHEMA = "npa.sim2real.vlm_eval.v4"


def _index(value: Any) -> bool:
    return type(value) is int and value >= 0


def insufficient_visual_evidence(step: int) -> str:
    return f"Insufficient visual evidence for step {step}."


def bind_action_frames(
    *, actions: list[dict[str, Any]], frame_metadata: list[dict[str, Any]] | None,
    frame_names: list[str], rollout_id: str,
) -> dict[int, dict[str, Any]]:
    """Join selected primary frames to actions by exact recorded simulation step.

    A final frame or a nearby sampled frame supplies rollout context, but cannot
    stand in for an unobserved action. No filename or list-position inference is
    used to recover missing time metadata.
    """
    if (not isinstance(rollout_id, str) or not rollout_id.strip()
            or not isinstance(frame_metadata, list) or not frame_metadata):
        raise ValueError("visual grounding requires recorded frame metadata")
    by_name: dict[str, dict[str, Any]] = {}
    times: set[int] = set()
    for frame in frame_metadata:
        if not isinstance(frame, dict):
            raise ValueError("visual frame metadata must contain objects")
        name = frame.get("path")
        if (not isinstance(name, str) or not name
                or PurePosixPath(name).name != name or name in {".", ".."}
                or name in by_name or not _index(frame.get("sim_step"))
                or frame["sim_step"] in times
                or frame.get("view_name") != "primary"
                or frame.get("episode_id") != rollout_id):
            raise ValueError("visual frame metadata has an ambiguous identity or time")
        by_name[name] = frame
        times.add(frame["sim_step"])
    if (not frame_names or len(set(frame_names)) != len(frame_names)
            or any(name not in by_name for name in frame_names)):
        raise ValueError("selected visual frames lack unique recorded metadata")
    selected = {by_name[name]["sim_step"]: name for name in frame_names}
    bindings: dict[int, dict[str, Any]] = {}
    action_times: set[int] = set()
    for action in actions:
        if (not isinstance(action, dict) or not _index(action.get("step"))
                or not _index(action.get("sim_step"))
                or action["step"] in bindings or action["sim_step"] in action_times):
            raise ValueError("visual grounding requires unique action indices and times")
        step, sim_step = action["step"], action["sim_step"]
        name = selected.get(sim_step)
        bindings[step] = {
            "schema": VISUAL_GROUNDING_SCHEMA,
            "action_step": step, "action_sim_step": sim_step,
            "camera_observation": name,
            "frame_sim_step": sim_step if name is not None else None,
            "supported": name is not None,
        }
        action_times.add(sim_step)
    if not bindings:
        raise ValueError("visual grounding requires at least one action")
    return bindings


def valid_visual_binding(binding: Any, *, step: Any) -> bool:
    if (not isinstance(binding, dict) or binding.get("schema") != VISUAL_GROUNDING_SCHEMA
            or not {"schema", "action_step", "action_sim_step", "camera_observation",
                    "frame_sim_step", "supported"} <= binding.keys()
            or not _index(step) or binding.get("action_step") != step
            or not _index(binding.get("action_step"))
            or not _index(binding.get("action_sim_step"))):
        return False
    if binding.get("supported") is False:
        return binding.get("camera_observation") is None and binding.get("frame_sim_step") is None
    name = binding.get("camera_observation")
    return (binding.get("supported") is True and isinstance(name, str) and bool(name)
            and PurePosixPath(name).name == name and name not in {".", ".."}
            and _index(binding.get("frame_sim_step"))
            and binding["frame_sim_step"] == binding["action_sim_step"])


def supported_visual_event(event: dict[str, Any]) -> bool:
    binding = event.get("visual_grounding")
    return (valid_visual_binding(binding, step=event.get("step"))
            and binding["supported"] is True
            and _index(event.get("sim_step")) and event["sim_step"] == binding["action_sim_step"]
            and event.get("camera_observation") == binding["camera_observation"])


def validate_stored_visual_grounding(evaluation: dict[str, Any]) -> None:
    """Reconstruct the event bindings at the archived Stage 8→9 boundary."""
    frames = evaluation.get("selected_frames")
    rows = evaluation.get("per_step")
    if (not isinstance(frames, list) or not all(isinstance(name, str) for name in frames)
            or not isinstance(rows, list) or not rows
            or type(evaluation.get("frame_count")) is not int
            or evaluation["frame_count"] != len(frames)):
        raise ValueError("stored visual grounding has invalid frame or event coverage")
    bindings = bind_action_frames(
        actions=rows, frame_metadata=evaluation.get("selected_frame_metadata"),
        frame_names=frames, rollout_id=evaluation.get("rollout_id"),
    )
    for event in rows:
        binding = bindings[event["step"]]
        confidence = event.get("confidence")
        if ("camera_observation" not in event or event.get("visual_grounding") != binding
                or event.get("camera_observation") != binding["camera_observation"]
                or type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError("stored visual event differs from its recorded action/frame binding")
        if not binding["supported"] and (
            confidence != 0 or event.get("error_tags") != ["ok"]
            or event.get("critique_text") != insufficient_visual_evidence(event["step"])
        ):
            raise ValueError("stored unobserved action contains a visual claim")
