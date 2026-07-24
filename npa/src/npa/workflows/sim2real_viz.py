"""Rerun visualization emitter for completed Sim2Real loop runs.

This module turns a completed Sim2Real run's artifact tree into a single Rerun
``.rrd`` recording (and, optionally, per-rollout MP4s) so the VLM->RL loop can be
inspected visually: rollout camera frames as image streams, per-rollout VLM
critique text and score overlays, the per-step reward/advantage signal as scalar
timeseries, and the held-out per-env scores as a scalar/bar view.

It reuses the repo's existing Rerun capability (the ``rerun-sdk`` recording API
that ``npa.viz.adapters.lerobot_to_rerun`` and ``npa.viz.backends.rerun`` build
on) rather than reinventing a logger. ``rerun`` is imported lazily so the loop
degrades gracefully (WARN, not hard-fail) when the SDK is not installed locally,
but it MUST produce a non-empty ``.rrd`` whenever the SDK is available.
"""

from __future__ import annotations
import logging

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_ROLLOUT_SCHEMA = "npa.sim2real.action_rollout.v1"
REFERENCE_STUB_FRAME_SHAPE = (32, 32)
APPLICATION_ID = "npa_sim2real_loop"
TIMELINE = "frame_time"
ROLLOUT_FRAME_SECONDS = 0.5
HELDOUT_STEP_SECONDS = 1.0
CRITIQUE_COLOR = (255, 136, 0, 255)
FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


class Sim2RealVizError(Exception):
    """Raised when the Sim2Real Rerun emitter cannot produce a recording."""


class RerunUnavailableError(Sim2RealVizError):
    """Raised when the ``rerun`` SDK is not importable (caller WARNs and skips)."""


class McapUnavailableError(Sim2RealVizError):
    """Raised when the ``mcap`` writer is not importable (caller WARNs and skips)."""


MCAP_FRAME_ID = "sim2real"
# foxglove.Log level for INFO-severity critique/summary messages.
_LOG_LEVEL_INFO = 2
_LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "foxglove.Log",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "level": {"type": "integer"},
        "message": {"type": "string"},
        "name": {"type": "string"},
        "file": {"type": "string"},
        "line": {"type": "integer"},
    },
}
# Generic numeric sample so a Foxglove/Lichtblick Plot panel can chart any signal.
_SCALAR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "npa.sim2real.Scalar",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
        },
        "value": {"type": "number"},
        "label": {"type": "string"},
    },
}


@dataclass(frozen=True)
class Sim2RealVizResult:
    """Result of emitting a Sim2Real Rerun recording."""

    status: str
    output_rrd_path: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    rollout_count: int = 0
    frame_count: int = 0
    heldout_env_count: int = 0
    heldout_frame_count: int = 0
    mp4_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_rrd_path": self.output_rrd_path,
            "entity_counts": dict(self.entity_counts),
            "rollout_count": self.rollout_count,
            "frame_count": self.frame_count,
            "heldout_env_count": self.heldout_env_count,
            "heldout_frame_count": self.heldout_frame_count,
            "mp4_paths": list(self.mp4_paths),
        }


@dataclass(frozen=True)
class Sim2RealMcapResult:
    """Result of emitting a Sim2Real Lichtblick/Foxglove MCAP recording."""

    status: str
    output_mcap_path: str
    channel_counts: dict[str, int] = field(default_factory=dict)
    message_count: int = 0
    camera_message_count: int = 0
    scalar_message_count: int = 0
    log_message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_mcap_path": self.output_mcap_path,
            "channel_counts": dict(self.channel_counts),
            "message_count": self.message_count,
            "camera_message_count": self.camera_message_count,
            "scalar_message_count": self.scalar_message_count,
            "log_message_count": self.log_message_count,
        }


def emit_sim2real_rerun(
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    output_rrd: Path | None = None,
    write_mp4: bool = False,
) -> Sim2RealVizResult:
    """Write ``reports/sim2real.rrd`` for a completed run's artifacts."""

    rr, rrb = _import_rerun()
    local_dir = Path(local_dir)
    output_rrd = Path(output_rrd) if output_rrd is not None else local_dir / "reports" / "sim2real.rrd"
    if output_rrd.suffix.lower() != ".rrd":
        raise Sim2RealVizError(f"Rerun output path must end in .rrd, got: {output_rrd}")
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    heldout_episodes = _heldout_render_episodes(local_dir, heldout_report)
    has_heldout_cameras = bool(heldout_episodes)
    blueprint = _build_blueprint(
        rrb,
        has_heldout_cameras=has_heldout_cameras,
        heldout_env_ids=[env_id for env_id, _frames in heldout_episodes],
    )
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd, default_blueprint=blueprint, recording=recording)
    _send_blueprint(rr, blueprint, recording)

    counts: dict[str, int] = {}
    seconds = 0.0
    rollout_count = 0
    frame_count = 0
    heldout_frame_count = 0
    mp4_paths: list[str] = []
    critique_panel_rows: list[str] = []
    _log_scene_overview(rr, recording, inner_evidence, counts)

    iterations = inner_evidence.get("iterations") or []
    for record in iterations:
        iteration = int(record.get("iteration", len(mp4_paths) + 1))
        actions_dir = _maybe_path(record.get("actions_dir"))
        eval_dir = _maybe_path(record.get("vlm_eval_dir"))
        signal_dir = _maybe_path(record.get("signal_dir"))
        for rollout_dir in _rollout_dirs(actions_dir):
            frames = _rollout_frames(rollout_dir)
            if has_heldout_cameras and is_reference_stub_rollout(rollout_dir, frames):
                continue
            rollout_id = rollout_dir.name
            iter_root = f"rollouts/iter_{iteration:02d}/{rollout_id}"
            evaluation = _read_json(eval_dir / f"{rollout_id}.json") if eval_dir else {}
            signal = _read_json(signal_dir / f"{rollout_id}.json") if signal_dir else {}
            manifest = _read_json(rollout_dir / "manifest.json")
            seconds = _log_rollout(
                rr,
                recording,
                root=iter_root,
                frames=frames,
                evaluation=evaluation,
                signal=signal,
                manifest=manifest,
                start_seconds=seconds,
                counts=counts,
                critique_panel_rows=critique_panel_rows,
            )
            rollout_count += 1
            frame_count += len(frames)
            if write_mp4 and frames:
                mp4_path = _maybe_write_mp4(rollout_dir, frames)
                if mp4_path is not None:
                    mp4_paths.append(str(mp4_path))

    _log_vlm_critique_panel(rr, recording, critique_panel_rows, counts)
    _log_reward_trend(rr, recording, inner_evidence.get("reward_trend") or [], counts)
    heldout_frame_count, heldout_seconds = _log_heldout_cameras(
        rr,
        recording,
        heldout_episodes,
        counts,
        start_seconds=seconds,
    )
    seconds = max(seconds, heldout_seconds)
    heldout_env_count = _log_heldout(
        rr,
        recording,
        (heldout_report or {}).get("per_env") or [],
        (heldout_report or {}).get("success_rate"),
        counts,
    )

    _disconnect(rr, recording)

    if not output_rrd.exists() or output_rrd.stat().st_size == 0:
        raise Sim2RealVizError(f"Rerun recording was not written: {output_rrd}")
    if (
        frame_count == 0
        and rollout_count == 0
        and heldout_env_count == 0
        and heldout_frame_count == 0
    ):
        raise Sim2RealVizError(
            "Sim2Real Rerun recording has no rollout frames, held-out cameras, signal, or held-out content"
        )

    return Sim2RealVizResult(
        status="written",
        output_rrd_path=str(output_rrd),
        entity_counts=counts,
        rollout_count=rollout_count,
        frame_count=frame_count,
        heldout_env_count=heldout_env_count,
        heldout_frame_count=heldout_frame_count,
        mp4_paths=mp4_paths,
    )


def _log_rollout(
    rr: Any,
    recording: Any,
    *,
    root: str,
    frames: list[np.ndarray],
    evaluation: dict[str, Any],
    signal: dict[str, Any],
    manifest: dict[str, Any],
    start_seconds: float,
    counts: dict[str, int],
    critique_panel_rows: list[str],
) -> float:
    seconds = start_seconds
    per_step_eval = {int(item.get("step", index)): item for index, item in enumerate(evaluation.get("per_step") or [])}
    per_step_signal = {int(item.get("step", index)): item for index, item in enumerate(signal.get("per_step") or [])}
    per_step_actions = _actions_by_step(manifest.get("actions"))
    score = evaluation.get("score")
    summary = str(evaluation.get("summary") or "")
    last_critique = ""

    for step, frame in enumerate(frames):
        _set_time(rr, recording, seconds)
        rr.log(f"{root}/camera", _rerun_image(rr, frame), recording=recording)
        _bump(counts, f"{root}/camera")

        eval_step = per_step_eval.get(step, {})
        critique = str(eval_step.get("critique_text") or summary or "")
        tags = eval_step.get("error_tags") or []
        if critique:
            overlay = critique if not tags else f"{critique}\n\nerror_tags: {', '.join(str(tag) for tag in tags)}"
            rr.log(
                f"{root}/critique",
                rr.TextDocument(overlay, media_type="text/markdown"),
                recording=recording,
            )
            _bump(counts, f"{root}/critique")
            last_critique = overlay
        if score is not None:
            rr.log(f"{root}/score", _scalar(rr, float(score)), recording=recording)
            _bump(counts, f"{root}/score")

        action_values = _as_float_list(eval_step.get("action"))
        if not action_values:
            action_values = per_step_actions.get(step, [])
        for dim, value in enumerate(action_values):
            rr.log(f"{root}/actions/dim_{dim:02d}", _scalar(rr, float(value)), recording=recording)
            _bump(counts, f"{root}/actions/dim_{dim:02d}")
        if action_values:
            rr.log(
                f"{root}/actions/l2_norm",
                _scalar(rr, float(np.linalg.norm(np.asarray(action_values, dtype=float)))),
                recording=recording,
            )
            _bump(counts, f"{root}/actions/l2_norm")

        signal_step = per_step_signal.get(step, {})
        if "reward" in signal_step:
            rr.log("signal/reward", _scalar(rr, float(signal_step["reward"])), recording=recording)
            _bump(counts, "signal/reward")
        if signal_step.get("advantage") is not None:
            rr.log("signal/advantage", _scalar(rr, float(signal_step["advantage"])), recording=recording)
            _bump(counts, "signal/advantage")
        seconds += ROLLOUT_FRAME_SECONDS
    critique_body = summary or last_critique
    if critique_body:
        score_value = f"{float(score):.3f}" if score is not None else "n/a"
        critique_panel_rows.append(f"### `{root}`\n\nscore: `{score_value}`\n\n{critique_body}")
    return seconds


def _log_reward_trend(rr: Any, recording: Any, reward_trend: list[Any], counts: dict[str, int]) -> None:
    for index, value in enumerate(reward_trend):
        _set_time(rr, recording, float(index))
        rr.log("signal/reward_trend", _scalar(rr, float(value)), recording=recording)
        _bump(counts, "signal/reward_trend")


def _log_vlm_critique_panel(
    rr: Any,
    recording: Any,
    entries: list[str],
    counts: dict[str, int],
) -> None:
    if not entries:
        return
    _set_time(rr, recording, 0.0)
    rr.log(
        "rollouts/summary/critique",
        rr.TextDocument("# VLM critiques by rollout\n\n" + "\n\n---\n\n".join(entries), media_type="text/markdown"),
        recording=recording,
    )
    _bump(counts, "rollouts/summary/critique")


def _franka_joint_positions(joint_angles: tuple[float, ...]) -> list[list[float]]:
    dh = [
        (0.0, 0.0, 0.333),
        (0.0, -math.pi / 2.0, 0.0),
        (0.0, math.pi / 2.0, 0.316),
        (0.0825, math.pi / 2.0, 0.0),
        (-0.0825, -math.pi / 2.0, 0.384),
        (0.0, math.pi / 2.0, 0.0),
        (0.088, math.pi / 2.0, 0.0),
    ]

    def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    positions = [[0.0, 0.0, 0.0]]
    for index, (a, alpha, d) in enumerate(dh):
        theta = float(joint_angles[index])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = _matmul(transform, step)
        positions.append([transform[0][3], transform[1][3], transform[2][3]])
    ee = [transform[0][3], transform[1][3], transform[2][3] + 0.103]
    positions.append(ee)
    positions.append([ee[0], ee[1] + 0.04, ee[2]])
    positions.append([ee[0], ee[1] - 0.04, ee[2]])
    return positions


def _scene_joint_angles(frame_index: int, frame_count: int) -> tuple[float, ...]:
    phase = (float(frame_index) / max(1.0, float(frame_count - 1))) * math.tau
    return (
        FRANKA_HOME_JOINTS[0] + 0.20 * math.sin(phase),
        FRANKA_HOME_JOINTS[1] + 0.14 * math.sin(phase + 0.5),
        FRANKA_HOME_JOINTS[2] + 0.16 * math.sin(phase + 1.2),
        FRANKA_HOME_JOINTS[3] + 0.10 * math.sin(phase + 1.7),
        FRANKA_HOME_JOINTS[4] + 0.22 * math.sin(phase + 2.1),
        FRANKA_HOME_JOINTS[5] + 0.09 * math.sin(phase + 2.7),
        FRANKA_HOME_JOINTS[6] + 0.18 * math.sin(phase + 3.4),
    )


def _log_franka_scene_frame(
    rr: Any,
    recording: Any,
    *,
    frame_index: int,
    frame_count: int,
    counts: dict[str, int],
) -> None:
    positions = _franka_joint_positions(_scene_joint_angles(frame_index, frame_count))
    arm_points = positions[:8]
    segments = [[left, right] for left, right in zip(arm_points, arm_points[1:]) if left != right]
    gripper_segments = [[positions[7], positions[8]], [positions[8], positions[9]], [positions[8], positions[10]]]
    progress = frame_index / max(1.0, float(frame_count - 1))
    cube_y = 0.28 - 0.36 * progress

    rr.log(
        "world/cube",
        rr.Boxes3D(
            centers=[[0.5, cube_y, 0.04]],
            half_sizes=[[0.025, 0.025, 0.025]],
            colors=[[59, 130, 246, 255]],
        ),
        recording=recording,
    )
    rr.log(
        "world/franka/joints",
        rr.Points3D(arm_points, colors=[[234, 88, 12, 255]] * len(arm_points), radii=[0.028] * len(arm_points)),
        recording=recording,
    )
    if segments:
        rr.log(
            "world/franka/links",
            rr.LineStrips3D(segments, colors=[[234, 88, 12]] * len(segments), radii=[0.018] * len(segments)),
            recording=recording,
        )
    rr.log(
        "world/franka/gripper",
        rr.LineStrips3D(gripper_segments, colors=[[59, 130, 246]] * len(gripper_segments), radii=[0.012] * len(gripper_segments)),
        recording=recording,
    )
    _bump(counts, "world/cube")
    _bump(counts, "world/franka/joints")
    _bump(counts, "world/franka/links")


def _log_scene_overview(
    rr: Any,
    recording: Any,
    inner_evidence: dict[str, Any],
    counts: dict[str, int],
) -> None:
    rr.log(
        "world/table",
        rr.Boxes3D(
            centers=[[0.5, 0.0, 0.0]],
            half_sizes=[[0.4, 0.3, 0.02]],
            colors=[[180, 180, 180, 255]],
        ),
        recording=recording,
    )
    rr.log(
        "world/summary",
        rr.TextDocument(
            "# Sim2Real scene overview\n\n"
            "Reference local run visualization: animated Franka proxy, moving cube, rollout signals, and held-out scores.",
            media_type="text/markdown",
        ),
        recording=recording,
    )
    frame_count = max(12, sum(
        len(_rollout_frames(rollout_dir))
        for record in inner_evidence.get("iterations") or []
        for rollout_dir in _rollout_dirs(_maybe_path(record.get("actions_dir")))
    ))
    for frame_index in range(frame_count):
        _set_time(rr, recording, frame_index * ROLLOUT_FRAME_SECONDS)
        _log_franka_scene_frame(rr, recording, frame_index=frame_index, frame_count=frame_count, counts=counts)
    _bump(counts, "world/table")
    _bump(counts, "world/summary")


def _log_heldout(
    rr: Any,
    recording: Any,
    per_env: list[dict[str, Any]],
    success_rate: Any,
    counts: dict[str, int],
) -> int:
    seconds = 0.0
    logged = 0
    if success_rate is not None:
        _set_time(rr, recording, 0.0)
        rr.log("heldout/success_rate", _scalar(rr, float(success_rate)), recording=recording)
        _bump(counts, "heldout/success_rate")
    for index, item in enumerate(per_env):
        if not isinstance(item, dict):
            continue
        env_id = str(item.get("env_id") or f"heldout-{index:04d}")
        score = float(item.get("score", 0.0))
        _set_time(rr, recording, seconds)
        rr.log("heldout/scores", _scalar(rr, score), recording=recording)
        rr.log(f"heldout/per_env/{env_id}", _scalar(rr, score), recording=recording)
        _bump(counts, "heldout/scores")
        _bump(counts, f"heldout/per_env/{env_id}")
        seconds += HELDOUT_STEP_SECONDS
        logged += 1
    return logged


def _log_heldout_cameras(
    rr: Any,
    recording: Any,
    episodes: list[tuple[str, list[np.ndarray]]],
    counts: dict[str, int],
    *,
    start_seconds: float,
) -> tuple[int, float]:
    logged = 0
    end_seconds = start_seconds
    for episode_index, (env_id, frames) in enumerate(episodes):
        root = f"heldout/camera/{env_id}"
        # Reset to the same start for every env so all held-out episodes share one
        # time window and play in sync (frame i of every env at the same t). Without
        # this, envs are laid end-to-end and only one is ever visible at the cursor.
        seconds = start_seconds
        for frame in frames:
            _set_time(rr, recording, seconds)
            image = _rerun_image(rr, frame)
            rr.log(f"{root}/camera", image, recording=recording)
            _bump(counts, f"{root}/camera")
            if episode_index == 0:
                rr.log("camera", image, recording=recording)
                _bump(counts, "camera")
            seconds += ROLLOUT_FRAME_SECONDS
            logged += 1
        end_seconds = max(end_seconds, seconds)
    return logged, end_seconds


def is_reference_stub_rollout(rollout_dir: Path, frames: list[np.ndarray]) -> bool:
    """Return True for stage-7 reference adapter solid-color PPM fixtures."""

    manifest = _read_json(rollout_dir / "manifest.json")
    if manifest.get("schema") != REFERENCE_ROLLOUT_SCHEMA:
        return False
    observations = list(manifest.get("camera_observations") or [])
    if observations and not all(str(item).endswith(".ppm") for item in observations):
        return False
    if frames and not all(frame.shape[:2] == REFERENCE_STUB_FRAME_SHAPE for frame in frames):
        return False
    return "quality" in manifest


def _heldout_render_episodes(
    local_dir: Path,
    heldout_report: dict[str, Any] | None,
) -> list[tuple[str, list[np.ndarray]]]:
    renders_root = local_dir / "eval" / "heldout" / "renders"
    manifest = (heldout_report or {}).get("render_manifest") or {}
    episodes: list[tuple[str, list[np.ndarray]]] = []
    for item in manifest.get("episodes") or []:
        if not isinstance(item, dict):
            continue
        env_id = str(item.get("env_id") or "")
        if not env_id:
            continue
        env_dir = renders_root / env_id
        frames = _usable_camera_frames(
            [
                frame
                for name in item.get("frames") or []
                if (frame := _read_image(env_dir / str(name))) is not None
            ]
        )
        if frames:
            episodes.append((env_id, frames))
    if episodes:
        return episodes
    if not renders_root.is_dir():
        return []
    for env_dir in sorted(path for path in renders_root.iterdir() if path.is_dir()):
        frames = _usable_camera_frames(
            [
                frame
                for frame_path in sorted(env_dir.glob("camera-*.png"))
                if (frame := _read_image(frame_path)) is not None
            ]
        )
        if frames:
            episodes.append((env_dir.name, frames))
    return episodes


def _build_blueprint(
    rrb: Any,
    *,
    has_heldout_cameras: bool = False,
    heldout_env_ids: list[str] | None = None,
) -> Any:
    env_ids = list(heldout_env_ids or [])
    has_heldout_cameras = has_heldout_cameras or bool(env_ids)
    if env_ids:
        # Keep a top-level camera alias first: the web viewer reliably opens this
        # single Spatial2DView, while the per-env grid remains available for
        # deeper inspection in the Streams tree.
        camera_view = rrb.Vertical(
            rrb.Spatial2DView(origin="camera", name="Franka held-out sim camera"),
            rrb.Grid(
                *[
                    rrb.Spatial2DView(
                        origin=f"heldout/camera/{env_id}",
                        name=f"Held-out {env_id}",
                    )
                    for env_id in env_ids
                ],
                name="Held-out sim cameras",
            ),
            row_shares=[3.0, 1.0],
        )
    elif has_heldout_cameras:
        camera_view = rrb.Spatial2DView(
            origin="heldout",
            contents="heldout/**/camera",
            name="Held-out sim cameras",
        )
    else:
        camera_view = rrb.Spatial2DView(
            origin="rollouts",
            contents="rollouts/**",
            name="Rollout cameras",
        )
    secondary_camera = (
        rrb.Spatial2DView(
            origin="rollouts",
            contents="rollouts/**/camera",
            name="Policy rollouts",
        )
        if has_heldout_cameras
        else None
    )
    left_column = (
        rrb.Vertical(camera_view, secondary_camera, row_shares=[2.0, 1.0])
        if secondary_camera is not None
        else camera_view
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", contents="world/**", name="Scene overview"),
            left_column,
            rrb.TextDocumentView(origin="rollouts", contents="rollouts/**/critique", name="VLM critiques"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="signal", contents="signal/**", name="VLM->RL signal"),
                rrb.TimeSeriesView(origin="heldout", contents="heldout/**", name="Held-out scores"),
            ),
            column_shares=[2.0, 1.4, 1.2, 1.3],
        ),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=TIMELINE),
        auto_layout=False,
    )


def _rollout_dirs(actions_dir: Path | None) -> list[Path]:
    if actions_dir is None or not actions_dir.exists():
        return []
    return sorted(path for path in actions_dir.iterdir() if path.is_dir() and path.name.startswith("rollout-"))


def _rollout_frames(rollout_dir: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_path in sorted(rollout_dir.glob("camera-*.ppm")):
        frame = _read_image(frame_path)
        if frame is not None:
            frames.append(frame)
    if frames:
        return frames
    for frame_path in sorted(rollout_dir.iterdir()):
        if frame_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            frame = _read_image(frame_path)
            if frame is not None:
                frames.append(frame)
    return frames


def _rollout_frame_paths(rollout_dir: Path) -> list[Path]:
    """Ordered rollout camera frame paths (``.ppm`` preferred, else PNG/JPEG)."""

    ppm = sorted(rollout_dir.glob("camera-*.ppm"))
    if ppm:
        return ppm
    return [
        path
        for path in sorted(rollout_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]


def _read_image(path: Path) -> np.ndarray | None:
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        return _read_ppm(path)
    if suffix == ".png":
        return _read_png(path)
    return None


def _read_png(path: Path) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    import struct
    import zlib

    index = 8
    width = height = 0
    idat = bytearray()
    while index + 8 <= len(data):
        length = struct.unpack("!I", data[index : index + 4])[0]
        chunk_type = data[index + 4 : index + 8]
        chunk = data[index + 8 : index + 8 + length]
        index += 12 + length
        if chunk_type == b"IHDR" and len(chunk) >= 8:
            width, height = struct.unpack("!II", chunk[:8])
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if width <= 0 or height <= 0 or not idat:
        return None
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    stride = width * 3 + 1
    if len(raw) < height * stride:
        return None
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    offset = 0
    for row in range(height):
        offset += 1
        pixels[row] = np.frombuffer(raw, dtype=np.uint8, count=width * 3, offset=offset).reshape(
            width, 3
        )
        offset += width * 3
    return pixels.copy()


def _read_ppm(path: Path) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.startswith(b"P6"):
        return None
    fields: list[bytes] = []
    index = 2
    while len(fields) < 3 and index < len(data):
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index:index + 1] == b"#":
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        fields.append(data[start:index])
    if len(fields) < 3:
        return None
    width, height, _maxval = (int(field) for field in fields)
    index += 1
    pixels = data[index:index + width * height * 3]
    if len(pixels) < width * height * 3:
        return None
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()


def _maybe_write_mp4(rollout_dir: Path, frames: list[np.ndarray]) -> Path | None:
    import shutil
    import subprocess
    import tempfile

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    output_path = rollout_dir / "rollout.mp4"
    with tempfile.TemporaryDirectory(prefix="npa-sim2real-mp4-") as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            _write_png(tmp_dir / f"frame_{index:06d}.png", frame)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "2",
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        return None
    return output_path


def _png_bytes(frame: np.ndarray) -> bytes:
    import struct
    import zlib

    array = np.ascontiguousarray(frame, dtype=np.uint8)
    height, width = int(array.shape[0]), int(array.shape[1])
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(array[row].tobytes())

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + tag + payload + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _chunk(b"IEND", b"")
    return png


def _write_png(path: Path, frame: np.ndarray) -> None:
    path.write_bytes(_png_bytes(frame))


def _usable_camera_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Drop blank Isaac warmup frames that otherwise render as black/purple tiles."""

    usable: list[np.ndarray] = []
    for frame in frames:
        if frame.size == 0:
            continue
        if float(frame.mean()) < 1.0:
            continue
        usable.append(frame)
    return usable


def _rerun_image(rr: Any, frame: np.ndarray) -> Any:
    array = np.ascontiguousarray(frame, dtype=np.uint8)
    if hasattr(rr, "Image"):
        try:
            return rr.Image(array, color_model="RGB")
        except TypeError:
            return rr.Image(array)
    return array


def _scalar(rr: Any, value: float) -> Any:
    if hasattr(rr, "Scalars"):
        return rr.Scalars(value)
    return rr.Scalar(value)


def _set_time(rr: Any, recording: Any, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(TIMELINE, seconds, recording=recording)
    else:
        rr.set_time(TIMELINE, duration=seconds, recording=recording)


def _send_blueprint(rr: Any, blueprint: Any, recording: Any) -> None:
    sender = getattr(rr, "send_blueprint", None)
    if callable(sender):
        sender(blueprint, recording=recording)


def _disconnect(rr: Any, recording: Any) -> None:
    disconnect = getattr(rr, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect(recording=recording)
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)


def _bump(counts: dict[str, int], entity: str) -> None:
    normalized = "/" + entity.strip("/")
    counts[normalized] = counts.get(normalized, 0) + 1


def _maybe_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _actions_by_step(values: Any) -> dict[int, list[float]]:
    actions: dict[int, list[float]] = {}
    for index, item in enumerate(values or []):
        if not isinstance(item, dict):
            continue
        step = int(item.get("step", index))
        payload = _as_float_list(item.get("action"))
        if payload:
            actions[step] = payload
    return actions


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


class _McapEmitter:
    """Lazily-registered MCAP channel writer for Sim2Real recordings."""

    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._schema_ids: dict[str, int] = {}
        self._channel_ids: dict[str, int] = {}
        self.channel_counts: dict[str, int] = {}
        self.camera_message_count = 0
        self.scalar_message_count = 0
        self.log_message_count = 0

    def _schema(self, name: str, schema: dict[str, Any]) -> int:
        if name not in self._schema_ids:
            self._schema_ids[name] = self._writer.register_schema(
                name=name, encoding="jsonschema", data=json.dumps(schema).encode("utf-8")
            )
        return self._schema_ids[name]

    def _channel(self, topic: str, schema_name: str, schema: dict[str, Any]) -> int:
        if topic not in self._channel_ids:
            schema_id = self._schema(schema_name, schema)
            self._channel_ids[topic] = self._writer.register_channel(
                topic=topic, message_encoding="json", schema_id=schema_id
            )
        return self._channel_ids[topic]

    def _add(self, topic: str, channel_id: int, message: dict[str, Any], stamp_ns: int) -> None:
        self._writer.add_message(
            channel_id=channel_id,
            log_time=stamp_ns,
            publish_time=stamp_ns,
            data=json.dumps(message).encode("utf-8"),
        )
        self.channel_counts[topic] = self.channel_counts.get(topic, 0) + 1

    def log_image_bytes(self, topic: str, payload: bytes, fmt: str, stamp_ns: int) -> None:
        from npa.workbench.lichtblick import (
            _COMPRESSED_IMAGE_SCHEMA,
            compressed_image_message,
        )

        channel_id = self._channel(topic, "foxglove.CompressedImage", _COMPRESSED_IMAGE_SCHEMA)
        message = compressed_image_message(
            payload, fmt=fmt, stamp_ns=stamp_ns, frame_id=MCAP_FRAME_ID
        )
        self._add(topic, channel_id, message, stamp_ns)
        self.camera_message_count += 1

    def log_scalar(self, topic: str, value: float, stamp_ns: int, *, label: str = "") -> None:
        channel_id = self._channel(topic, "npa.sim2real.Scalar", _SCALAR_SCHEMA)
        message = {
            "timestamp": {"sec": stamp_ns // 1_000_000_000, "nsec": stamp_ns % 1_000_000_000},
            "value": float(value),
            "label": label,
        }
        self._add(topic, channel_id, message, stamp_ns)
        self.scalar_message_count += 1

    def log_text(self, topic: str, message_text: str, stamp_ns: int, *, name: str = "") -> None:
        channel_id = self._channel(topic, "foxglove.Log", _LOG_SCHEMA)
        message = {
            "timestamp": {"sec": stamp_ns // 1_000_000_000, "nsec": stamp_ns % 1_000_000_000},
            "level": _LOG_LEVEL_INFO,
            "message": message_text,
            "name": name,
            "file": "",
            "line": 0,
        }
        self._add(topic, channel_id, message, stamp_ns)
        self.log_message_count += 1


def emit_sim2real_mcap(
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    output_mcap: Path | None = None,
) -> Sim2RealMcapResult:
    """Write ``reports/sim2real.mcap`` from the same inputs as the ``.rrd``.

    Emits the rollout/held-out camera frames as ``foxglove.CompressedImage`` (raw
    ``.ppm`` dumps are transcoded to PNG), VLM critiques as ``foxglove.Log``, and
    reward/advantage/score signals as numeric samples a Plot panel can chart, so a
    Foxglove-compatible viewer (Lichtblick) can play back the same rollout as
    Rerun. Reuses ``npa.workbench.lichtblick`` for the CompressedImage encoding.
    """

    writer_cls = _import_mcap()
    local_dir = Path(local_dir)
    output_mcap = (
        Path(output_mcap) if output_mcap is not None else local_dir / "reports" / "sim2real.mcap"
    )
    if output_mcap.suffix.lower() != ".mcap":
        raise Sim2RealVizError(f"MCAP output path must end in .mcap, got: {output_mcap}")
    output_mcap.parent.mkdir(parents=True, exist_ok=True)

    frame_period_ns = int(ROLLOUT_FRAME_SECONDS * 1_000_000_000)
    heldout_period_ns = int(HELDOUT_STEP_SECONDS * 1_000_000_000)

    heldout_episodes = _heldout_render_episodes(local_dir, heldout_report)
    has_heldout_cameras = bool(heldout_episodes)

    from npa.workbench.lichtblick import encode_frame_to_compressed_bytes

    with open(output_mcap, "wb") as handle:
        writer = writer_cls(handle)
        writer.start(profile="", library="npa-sim2real")
        emitter = _McapEmitter(writer)

        stamp_ns = 0
        for record in inner_evidence.get("iterations") or []:
            iteration = int(record.get("iteration", 1))
            actions_dir = _maybe_path(record.get("actions_dir"))
            eval_dir = _maybe_path(record.get("vlm_eval_dir"))
            signal_dir = _maybe_path(record.get("signal_dir"))
            for rollout_dir in _rollout_dirs(actions_dir):
                frame_paths = _rollout_frame_paths(rollout_dir)
                if has_heldout_cameras and is_reference_stub_rollout(
                    rollout_dir, [f for p in frame_paths if (f := _read_image(p)) is not None]
                ):
                    continue
                rollout_id = rollout_dir.name
                root = f"/rollouts/iter_{iteration:02d}/{rollout_id}"
                evaluation = _read_json(eval_dir / f"{rollout_id}.json") if eval_dir else {}
                signal = _read_json(signal_dir / f"{rollout_id}.json") if signal_dir else {}
                stamp_ns = _emit_mcap_rollout(
                    emitter,
                    root=root,
                    frame_paths=frame_paths,
                    evaluation=evaluation,
                    signal=signal,
                    start_ns=stamp_ns,
                    frame_period_ns=frame_period_ns,
                    encode=encode_frame_to_compressed_bytes,
                )

        for index, value in enumerate(inner_evidence.get("reward_trend") or []):
            emitter.log_scalar(
                "/signal/reward_trend",
                float(value),
                index * frame_period_ns,
                label="reward_trend",
            )

        _emit_mcap_heldout_cameras(
            emitter, heldout_episodes, frame_period_ns=frame_period_ns
        )
        _emit_mcap_heldout_scores(emitter, heldout_report, heldout_period_ns=heldout_period_ns)

        writer.finish()

    if not output_mcap.exists() or output_mcap.stat().st_size == 0:
        raise Sim2RealVizError(f"MCAP recording was not written: {output_mcap}")
    total = (
        emitter.camera_message_count
        + emitter.scalar_message_count
        + emitter.log_message_count
    )
    if total == 0:
        raise Sim2RealVizError(
            "Sim2Real MCAP recording has no camera, signal, critique, or held-out content"
        )
    return Sim2RealMcapResult(
        status="written",
        output_mcap_path=str(output_mcap),
        channel_counts=emitter.channel_counts,
        message_count=total,
        camera_message_count=emitter.camera_message_count,
        scalar_message_count=emitter.scalar_message_count,
        log_message_count=emitter.log_message_count,
    )


def emit_sim2real_mcap_if_enabled(
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    output_mcap: Path | None = None,
) -> dict[str, Any]:
    """Best-effort ``reports/sim2real.mcap`` emission for the finalize stage.

    Gated behind ``NPA_SIM2REAL_MCAP`` (default on when rerun viz is on). Degrades
    gracefully (returns a ``skipped``/``disabled`` status dict, never raises) so a
    missing ``mcap`` writer or unreadable frame can never fail the finalize stage,
    mirroring the ``.rrd`` path. Shared by both loop engines.
    """

    import os

    toggle = str(os.environ.get("NPA_SIM2REAL_MCAP", "1")).strip().lower()
    if toggle in {"0", "false", "no", "off", ""}:
        return {"status": "disabled", "reason": "NPA_SIM2REAL_MCAP is off"}
    try:
        result = emit_sim2real_mcap(
            local_dir=local_dir,
            inner_evidence=inner_evidence,
            heldout_report=heldout_report,
            output_mcap=output_mcap,
        )
    except McapUnavailableError as exc:
        return {"status": "skipped", "reason": str(exc)}
    except Sim2RealVizError as exc:
        logging.getLogger(__name__).warning("Sim2Real MCAP emission failed: %s", exc)
        return {"status": "skipped", "reason": str(exc)}
    return result.to_dict()


def _emit_mcap_rollout(
    emitter: _McapEmitter,
    *,
    root: str,
    frame_paths: list[Path],
    evaluation: dict[str, Any],
    signal: dict[str, Any],
    start_ns: int,
    frame_period_ns: int,
    encode: Any,
) -> int:
    per_step_eval = {
        int(item.get("step", index)): item
        for index, item in enumerate(evaluation.get("per_step") or [])
    }
    per_step_signal = {
        int(item.get("step", index)): item
        for index, item in enumerate(signal.get("per_step") or [])
    }
    score = evaluation.get("score")
    summary = str(evaluation.get("summary") or "")
    stamp_ns = start_ns
    for step, path in enumerate(frame_paths):
        try:
            payload, fmt = encode(str(path))
        except Exception:
            logging.getLogger(__name__).debug("skipping unreadable frame %s", path, exc_info=True)
            stamp_ns += frame_period_ns
            continue
        emitter.log_image_bytes(f"{root}/camera", payload, fmt, stamp_ns)

        eval_step = per_step_eval.get(step, {})
        critique = str(eval_step.get("critique_text") or summary or "")
        tags = eval_step.get("error_tags") or []
        if critique:
            overlay = critique if not tags else f"{critique} [error_tags: {', '.join(str(t) for t in tags)}]"
            emitter.log_text(f"{root}/critique", overlay, stamp_ns, name=root.strip("/"))
        if score is not None:
            emitter.log_scalar(f"{root}/score", float(score), stamp_ns, label="score")

        signal_step = per_step_signal.get(step, {})
        if "reward" in signal_step:
            emitter.log_scalar("/signal/reward", float(signal_step["reward"]), stamp_ns, label="reward")
        if signal_step.get("advantage") is not None:
            emitter.log_scalar(
                "/signal/advantage", float(signal_step["advantage"]), stamp_ns, label="advantage"
            )
        stamp_ns += frame_period_ns
    if summary:
        score_value = f"{float(score):.3f}" if score is not None else "n/a"
        emitter.log_text(
            f"{root}/summary",
            f"score={score_value} :: {summary}",
            start_ns,
            name=root.strip("/"),
        )
    return stamp_ns


def _emit_mcap_heldout_cameras(
    emitter: _McapEmitter,
    episodes: list[tuple[str, list[np.ndarray]]],
    *,
    frame_period_ns: int,
) -> None:
    for episode_index, (env_id, frames) in enumerate(episodes):
        root = f"/heldout/camera/{env_id}"
        stamp_ns = 0
        for frame in frames:
            payload = _png_bytes(frame)
            emitter.log_image_bytes(f"{root}/camera", payload, "png", stamp_ns)
            if episode_index == 0:
                emitter.log_image_bytes("/camera", payload, "png", stamp_ns)
            stamp_ns += frame_period_ns


def _emit_mcap_heldout_scores(
    emitter: _McapEmitter,
    heldout_report: dict[str, Any] | None,
    *,
    heldout_period_ns: int,
) -> None:
    report = heldout_report or {}
    success_rate = report.get("success_rate")
    if success_rate is not None:
        emitter.log_scalar("/heldout/success_rate", float(success_rate), 0, label="success_rate")
    stamp_ns = 0
    for index, item in enumerate(report.get("per_env") or []):
        if not isinstance(item, dict):
            continue
        env_id = str(item.get("env_id") or f"heldout-{index:04d}")
        score = float(item.get("score", 0.0))
        emitter.log_scalar("/heldout/scores", score, stamp_ns, label=env_id)
        emitter.log_scalar(f"/heldout/per_env/{env_id}", score, stamp_ns, label=env_id)
        stamp_ns += heldout_period_ns


def _import_mcap() -> Any:
    try:
        from mcap.writer import Writer
    except ImportError as exc:  # pragma: no cover
        raise McapUnavailableError(
            "mcap is not installed; skipping Sim2Real MCAP visualization"
        ) from exc
    return Writer


def _import_rerun() -> tuple[Any, Any]:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover
        raise RerunUnavailableError(
            "rerun-sdk is not installed; skipping Sim2Real Rerun visualization"
        ) from exc
    return rr, rrb
