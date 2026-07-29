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

import hashlib
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
SYNTHETIC_STEP_SECONDS = 1.0
SYNTHETIC_DATASET_SAMPLE_LIMIT = 6
SYNTHETIC_AUGMENT_SAMPLE_LIMIT = 12
CRITIQUE_COLOR = (255, 136, 0, 255)
FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


class Sim2RealVizError(Exception):
    """Raised when the Sim2Real Rerun emitter cannot produce a recording."""


class RerunUnavailableError(Sim2RealVizError):
    """Raised when the ``rerun`` SDK is not importable (caller WARNs and skips)."""


class McapUnavailableError(Sim2RealVizError):
    """Raised when the ``mcap`` writer is not importable (caller WARNs and skips)."""


MCAP_FRAME_ID = "sim2real"
# Root/world frame the sim2real frame is anchored to. The 3D panel needs a defined
# coordinate frame to place the point cloud, so the emitter publishes a static
# ``world`` -> ``sim2real`` transform on ``/tf``.
MCAP_ROOT_FRAME_ID = "world"
MCAP_TF_TOPIC = "/tf"
# Topic the embedded viewer's default layout binds its Image panel to (see
# ``npa.cli.agent._lichtblick_default_layout_json``). Held-out and rollout cameras
# also get their own per-episode topics; this one is the single well-known stream
# the default layout can rely on, so the emitter must always populate it.
MCAP_PRIMARY_CAMERA_TOPIC = "/camera"
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
    pointcloud_frame_count: int = 0
    synthetic_frame_count: int = 0
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
            "pointcloud_frame_count": self.pointcloud_frame_count,
            "synthetic_frame_count": self.synthetic_frame_count,
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
    pointcloud_message_count: int = 0
    transform_message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_mcap_path": self.output_mcap_path,
            "channel_counts": dict(self.channel_counts),
            "message_count": self.message_count,
            "camera_message_count": self.camera_message_count,
            "scalar_message_count": self.scalar_message_count,
            "log_message_count": self.log_message_count,
            "pointcloud_message_count": self.pointcloud_message_count,
            "transform_message_count": self.transform_message_count,
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
    has_synthetic_data = _has_synthetic_visual_data(local_dir)
    blueprint = _build_blueprint(
        rrb,
        has_heldout_cameras=has_heldout_cameras,
        heldout_env_ids=[env_id for env_id, _frames in heldout_episodes],
        has_synthetic_data=has_synthetic_data,
    )
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd, default_blueprint=blueprint, recording=recording)
    _send_blueprint(rr, blueprint, recording)

    counts: dict[str, int] = {}
    seconds = 0.0
    rollout_count = 0
    frame_count = 0
    heldout_frame_count = 0
    synthetic_frame_count = 0
    mp4_paths: list[str] = []
    critique_panel_rows: list[str] = []
    if not has_heldout_cameras:
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

    _log_reward_trend(rr, recording, inner_evidence.get("reward_trend") or [], counts)
    synthetic_frame_count = _log_synthetic_data(rr, recording, local_dir, counts)
    heldout_frame_count, heldout_seconds = _log_heldout_cameras(
        rr,
        recording,
        heldout_episodes,
        counts,
        start_seconds=0.0 if has_heldout_cameras else seconds,
    )
    seconds = max(seconds, heldout_seconds)
    pointcloud_frame_count = _log_heldout_pointclouds(
        rr,
        recording,
        _heldout_pointcloud_frames(local_dir),
        counts,
    )
    heldout_env_count = _log_heldout(
        rr,
        recording,
        (heldout_report or {}).get("per_env") or [],
        (heldout_report or {}).get("success_rate"),
        counts,
    )
    _log_vlm_critique_panel(rr, recording, critique_panel_rows, counts)
    _log_summary_documents(
        rr,
        recording,
        local_dir=local_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        critique_panel_rows=critique_panel_rows,
        counts=counts,
    )

    _disconnect(rr, recording)

    if not output_rrd.exists() or output_rrd.stat().st_size == 0:
        raise Sim2RealVizError(f"Rerun recording was not written: {output_rrd}")
    if frame_count == 0 and rollout_count == 0 and heldout_env_count == 0 and heldout_frame_count == 0:
        raise Sim2RealVizError(
            "Sim2Real Rerun recording has no real rollout frames, held-out cameras, or held-out scores; "
            f"synthetic descriptor previews logged={synthetic_frame_count}"
        )

    return Sim2RealVizResult(
        status="written",
        output_rrd_path=str(output_rrd),
        entity_counts=counts,
        rollout_count=rollout_count,
        frame_count=frame_count,
        heldout_env_count=heldout_env_count,
        heldout_frame_count=heldout_frame_count,
        pointcloud_frame_count=pointcloud_frame_count,
        synthetic_frame_count=synthetic_frame_count,
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
    body = "# VLM critiques by rollout\n\n" + "\n\n---\n\n".join(entries)
    rr.log(
        "rollouts/summary/critique",
        rr.TextDocument(body, media_type="text/markdown"),
        recording=recording,
    )
    _bump(counts, "rollouts/summary/critique")
    rr.log(
        "summary/vlm_critiques",
        rr.TextDocument(body, media_type="text/markdown"),
        recording=recording,
    )
    _bump(counts, "summary/vlm_critiques")


def _log_summary_documents(
    rr: Any,
    recording: Any,
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    critique_panel_rows: list[str],
    counts: dict[str, int],
) -> None:
    index = _build_visual_index(
        local_dir=local_dir,
        inner_evidence=inner_evidence,
        heldout_report=heldout_report,
        critique_panel_rows=critique_panel_rows,
    )
    reports_dir = Path(local_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "sim2real-visual-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    documents = {
        "summary/run_success": _run_success_markdown(index),
        "summary/augmentation": _augmentation_markdown(index),
        "summary/artifacts": _artifacts_markdown(index),
    }
    _set_time(rr, recording, 0.0)
    for entity, body in documents.items():
        if not body.strip():
            continue
        rr.log(entity, rr.TextDocument(body, media_type="text/markdown"), recording=recording)
        _bump(counts, entity)


def _build_visual_index(
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    critique_panel_rows: list[str],
) -> dict[str, Any]:
    report = _read_json(Path(local_dir) / "reports" / "sim2real-report.json")
    decision = _read_json(Path(local_dir) / "outer_loop" / "decision.json")
    augment = _read_json(Path(local_dir) / "augment" / "manifest.json")
    augment_index = _read_json(Path(local_dir) / "augment" / "frames" / "index.json")
    tokens = _read_json(Path(local_dir) / "tokens" / "manifest.json")
    split = _read_json(Path(local_dir) / "envs" / "manifest" / "split-manifest.json")
    if not split:
        split = _read_json(Path(local_dir) / "envs" / "split-manifest.json")
    full_heldout = dict(heldout_report or {})
    if not full_heldout:
        full_heldout = dict((report.get("outer_loop") or {}).get("latest_heldout_report") or {})

    return {
        "schema": "npa.sim2real.visual_index.v1",
        "run_id": report.get("run_id") or decision.get("run_id") or "",
        "success": {
            "status": report.get("status") or "",
            "decision": decision.get("decision") or "",
            "success_rate": _first_present(decision.get("success_rate"), full_heldout.get("success_rate")),
            "threshold": _first_present(decision.get("threshold"), report.get("threshold")),
            "checkpoint_uri": decision.get("checkpoint_uri") or "",
            "heldout_envs": [
                {
                    "env_id": item.get("env_id"),
                    "score": item.get("score"),
                    "success": item.get("success"),
                    "distance_m": item.get("distance_m"),
                }
                for item in (full_heldout.get("per_env") or [])[:8]
                if isinstance(item, dict)
            ],
        },
        "augmentation": {
            "status": augment.get("status") or "",
            "mode": augment.get("mode") or "",
            "stage": augment.get("stage") or "",
            "frame_count": _first_present(augment.get("frame_count"), augment_index.get("frame_count")),
            "image": augment.get("image") or "",
            "input_uri": augment.get("input_uri") or "",
            "output_uri": augment.get("output_uri") or "",
            "sample_frames": (augment_index.get("frames") or [])[:8],
        },
        "dataset": {
            "raw_count": split.get("raw_count"),
            "train_count": _first_present(split.get("train_count"), tokens.get("train_env_count")),
            "heldout_count": _first_present(split.get("heldout_count"), tokens.get("heldout_env_count")),
            "disjoint": split.get("disjoint"),
            "seed": split.get("seed"),
            "train_samples": _jsonl_samples(Path(local_dir) / "envs" / "train" / "envs.jsonl"),
            "heldout_samples": _jsonl_samples(Path(local_dir) / "envs" / "heldout" / "envs.jsonl"),
        },
        "vlm": {
            "critique_count": len(critique_panel_rows),
            "model": _first_sample_value(inner_evidence, "sample_vlm_eval", "model"),
            "score": _first_sample_value(inner_evidence, "sample_vlm_eval", "score"),
        },
        "synthetic": _synthetic_visual_index(Path(local_dir)),
        "artifact_counts": _artifact_counts(Path(local_dir)),
        "key_artifacts": _key_artifacts(Path(local_dir)),
    }


def _run_success_markdown(index: dict[str, Any]) -> str:
    success = index.get("success") or {}
    rows = [
        ("Run", index.get("run_id") or "unknown"),
        ("Status", success.get("status") or "unknown"),
        ("Decision", success.get("decision") or "unknown"),
        ("Held-out success rate", _format_value(success.get("success_rate"))),
        ("Threshold", _format_value(success.get("threshold"))),
        ("Final checkpoint", success.get("checkpoint_uri") or "not recorded"),
    ]
    per_env = success.get("heldout_envs") or []
    lines = ["# Sim2Real success", "", _markdown_table(["Metric", "Value"], rows)]
    if per_env:
        lines.extend(
            [
                "",
                "## Held-out eval",
                "",
                _markdown_table(
                    ["Env", "Success", "Score", "Distance m"],
                    [
                        (
                            item.get("env_id") or "",
                            _format_value(item.get("success")),
                            _format_value(item.get("score")),
                            _format_value(item.get("distance_m")),
                        )
                        for item in per_env
                    ],
                ),
            ]
        )
    return "\n".join(lines)


def _augmentation_markdown(index: dict[str, Any]) -> str:
    aug = index.get("augmentation") or {}
    dataset = index.get("dataset") or {}
    synthetic = index.get("synthetic") or {}
    lines = [
        "# Augmentation and dataset",
        "",
        "## Cosmos transfer augmentation",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ("Status", aug.get("status") or "unknown"),
                ("Mode", aug.get("mode") or "unknown"),
                ("Frame descriptors", _format_value(aug.get("frame_count"))),
                ("Image", aug.get("image") or "not recorded"),
                ("Input", aug.get("input_uri") or "not recorded"),
                ("Output", aug.get("output_uri") or "not recorded"),
            ],
        ),
        "",
        "## Environment split",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ("Raw envs", _format_value(dataset.get("raw_count"))),
                ("Train envs", _format_value(dataset.get("train_count"))),
                ("Held-out envs", _format_value(dataset.get("heldout_count"))),
                ("Disjoint split", _format_value(dataset.get("disjoint"))),
                ("Seed", _format_value(dataset.get("seed"))),
            ],
        ),
        "",
        "## Rerun synthetic-data visuals",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ("Synthetic viewport", synthetic.get("view_origin") or "synthetic"),
                ("Dataset samples visualized", _format_value(synthetic.get("dataset_sample_count"))),
                ("Actual dataset camera PNGs", _format_value(synthetic.get("dataset_camera_image_count"))),
                ("Dataset descriptor previews", _format_value(synthetic.get("dataset_descriptor_preview_count"))),
                ("Augmentation samples visualized", _format_value(synthetic.get("augmentation_sample_count"))),
                ("Actual augmentation PNGs", _format_value(synthetic.get("augmentation_image_count"))),
                ("Augmentation descriptor previews", _format_value(synthetic.get("augmentation_descriptor_preview_count"))),
            ],
        ),
    ]
    samples = dataset.get("heldout_samples") or dataset.get("train_samples") or []
    if samples:
        lines.extend(["", "## Sample env descriptors", ""])
        for sample in samples[:3]:
            lines.append(
                "- `{env_id}` `{asset}` friction `{friction}` lighting `{lighting}` augmented `{augmented}`".format(
                    env_id=sample.get("env_id", "env"),
                    asset=((sample.get("scene") or {}).get("simready_asset") or "asset"),
                    friction=_format_value((sample.get("physics") or {}).get("friction")),
                    lighting=_format_value((sample.get("physics") or {}).get("lighting_lux")),
                    augmented=((sample.get("scene") or {}).get("augmented_frame_uri") or "not recorded"),
                )
            )
    return "\n".join(lines)


def _artifacts_markdown(index: dict[str, Any]) -> str:
    counts = index.get("artifact_counts") or {}
    artifacts = index.get("key_artifacts") or {}
    vlm = index.get("vlm") or {}
    lines = [
        "# Visual artifact index",
        "",
        "## Counts",
        "",
        _markdown_table(["Artifact group", "Count"], sorted((key, value) for key, value in counts.items())),
        "",
        "## VLM signal",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ("Model", vlm.get("model") or "unknown"),
                ("Sample score", _format_value(vlm.get("score"))),
                ("Critique documents", _format_value(vlm.get("critique_count"))),
            ],
        ),
        "",
        "## Key artifacts",
        "",
        _markdown_table(["Name", "Path"], sorted((key, value) for key, value in artifacts.items())),
    ]
    return "\n".join(lines)


def _synthetic_visual_index(local_dir: Path) -> dict[str, Any]:
    dataset_samples = _synthetic_dataset_samples(local_dir)
    dataset_camera_count = 0
    dataset_view_count = 0
    for _split, sample in dataset_samples:
        for camera_name in _sample_camera_names(sample):
            dataset_view_count += 1
            if _local_camera_image_for_sample(local_dir, sample, camera_name) is not None:
                dataset_camera_count += 1
    augment_samples = _augmentation_visual_samples(local_dir)
    augment_image_count = sum(1 for _frame_id, _payload, frame in augment_samples if frame is not None)
    return {
        "view_origin": "synthetic",
        "dataset_sample_count": len(dataset_samples),
        "dataset_camera_view_count": dataset_view_count,
        "dataset_camera_image_count": dataset_camera_count,
        "dataset_descriptor_preview_count": max(0, dataset_view_count - dataset_camera_count),
        "augmentation_sample_count": len(augment_samples),
        "augmentation_image_count": augment_image_count,
        "augmentation_descriptor_preview_count": max(0, len(augment_samples) - augment_image_count),
    }


def _has_synthetic_visual_data(local_dir: Path) -> bool:
    root = Path(local_dir)
    return any(
        path.is_file()
        for path in (
            root / "envs" / "train" / "envs.jsonl",
            root / "envs" / "heldout" / "envs.jsonl",
            root / "augment" / "frames" / "index.json",
            root / "augment" / "manifest.json",
        )
    ) or any((root / "augment" / "frames").glob("frame-*.*"))


def _log_synthetic_data(
    rr: Any,
    recording: Any,
    local_dir: Path,
    counts: dict[str, int],
) -> int:
    """Log synthetic dataset and augmentation samples as image streams.

    If real RGB camera PNGs are present locally, those are logged. Current staged
    runs often contain env/augment descriptors without persisted camera pixels;
    those are rendered as labeled descriptor-preview tiles so the viewer exposes
    the synthetic data surface without pretending descriptor metadata is footage.
    """

    logged = 0
    dataset_samples = _synthetic_dataset_samples(Path(local_dir))
    for sample_index, (split, sample) in enumerate(dataset_samples):
        env_id = str(sample.get("env_id") or f"{split}-{sample_index:04d}")
        cameras = _sample_camera_names(sample)
        seconds = sample_index * SYNTHETIC_STEP_SECONDS
        for camera_index, camera_name in enumerate(cameras):
            frame = _local_camera_image_for_sample(Path(local_dir), sample, camera_name)
            if frame is None:
                frame = _synthetic_env_preview(sample, split=split, camera_name=camera_name, index=sample_index)
            _set_time(rr, recording, seconds + camera_index * 0.01)
            image = _rerun_image(rr, frame)
            entity = f"synthetic/dataset/{split}/{env_id}/{camera_name}"
            rr.log(entity, image, recording=recording)
            _bump(counts, entity)
            if logged == 0:
                rr.log("synthetic/preview", image, recording=recording)
                _bump(counts, "synthetic/preview")
            logged += 1

    for frame_index, (frame_id, payload, frame) in enumerate(_augmentation_visual_samples(Path(local_dir))):
        if frame is None:
            frame = _augmentation_preview(payload, index=frame_index)
        _set_time(rr, recording, frame_index * SYNTHETIC_STEP_SECONDS)
        image = _rerun_image(rr, frame)
        entity = f"synthetic/augmentation/{frame_id}/image"
        rr.log(entity, image, recording=recording)
        _bump(counts, entity)
        if logged == 0:
            rr.log("synthetic/preview", image, recording=recording)
            _bump(counts, "synthetic/preview")
        logged += 1
    return logged


def _synthetic_dataset_samples(local_dir: Path, *, limit_per_split: int = SYNTHETIC_DATASET_SAMPLE_LIMIT) -> list[tuple[str, dict[str, Any]]]:
    samples: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "heldout"):
        for sample in _jsonl_samples(Path(local_dir) / "envs" / split / "envs.jsonl", limit=limit_per_split):
            samples.append((split, sample))
    return samples


def _sample_camera_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for section in ("camera_obs", "cameras"):
        values = sample.get(section)
        if isinstance(values, dict):
            names.extend(str(name) for name in values.keys() if str(name))
    ordered = sorted(dict.fromkeys(names))
    return ordered or ["workspace"]


def _local_camera_image_for_sample(local_dir: Path, sample: dict[str, Any], camera_name: str) -> np.ndarray | None:
    for section in ("camera_obs", "cameras"):
        cameras = sample.get(section)
        if not isinstance(cameras, dict):
            continue
        spec = cameras.get(camera_name)
        if not isinstance(spec, dict):
            continue
        uri = str(spec.get("uri") or "")
        if not uri:
            continue
        path = _local_artifact_path_from_uri(local_dir, uri)
        if path is not None:
            image = _read_image(path)
            if image is not None:
                return image
    return None


def _augmentation_visual_samples(
    local_dir: Path,
    *,
    limit: int = SYNTHETIC_AUGMENT_SAMPLE_LIMIT,
) -> list[tuple[str, dict[str, Any], np.ndarray | None]]:
    frames_dir = Path(local_dir) / "augment" / "frames"
    index = _read_json(frames_dir / "index.json")
    manifest = _read_json(Path(local_dir) / "augment" / "manifest.json")
    records: list[dict[str, Any]] = []
    if isinstance(index.get("frames"), list):
        records = [item for item in index["frames"] if isinstance(item, dict)]
    if not records:
        records = [{"frame_id": path.stem, "local": str(path)} for path in sorted(frames_dir.glob("frame-*.*"))]
    if not records:
        frame_count = int(_safe_float(manifest.get("frame_count"), 0.0))
        frames_uri = str(manifest.get("augmented_frames_uri") or manifest.get("output_uri") or "").rstrip("/")
        perturbations = ["lighting", "texture", "background", "contrast"]
        records = [
            {
                "frame_id": f"frame-{index_no:05d}",
                "uri": f"{frames_uri}/frame-{index_no:05d}.json" if frames_uri else "",
                "source_dataset_uri": manifest.get("input_uri") or "",
                "perturbation": perturbations[index_no % len(perturbations)],
                "status": manifest.get("status") or "recorded",
                "mode": manifest.get("mode") or "",
                "image": manifest.get("image") or "",
            }
            for index_no in range(min(limit, frame_count))
        ]

    samples: list[tuple[str, dict[str, Any], np.ndarray | None]] = []
    seen: set[str] = set()
    for record in records:
        if len(samples) >= limit:
            break
        frame_id = str(record.get("frame_id") or Path(str(record.get("local") or record.get("uri") or "")).stem)
        if not frame_id or frame_id in seen:
            continue
        seen.add(frame_id)
        json_path = frames_dir / f"{frame_id}.json"
        payload_path = json_path if json_path.is_file() else _local_artifact_path_from_uri(local_dir, str(record.get("uri") or ""))
        payload = _read_json(payload_path) if payload_path is not None else {}
        if not payload:
            payload = dict(record)
        image_path = frames_dir / f"{frame_id}.png"
        if not image_path.is_file():
            image_path = _local_artifact_path_from_uri(local_dir, str(record.get("uri") or ""))
        frame = _read_image(image_path) if image_path is not None else None
        samples.append((frame_id, payload, frame))
    return samples


def _local_artifact_path_from_uri(local_dir: Path, uri: str) -> Path | None:
    ref = str(uri or "").strip()
    if not ref:
        return None
    if ref.startswith("file://"):
        path = Path(ref.removeprefix("file://"))
        return path if path.is_file() else None
    path = Path(ref)
    if not ref.startswith("s3://") and path.is_file():
        return path
    markers = ("/envs/", "/augment/", "/eval/", "/actions/", "/vlm_eval/", "/training_signal/", "/reports/")
    for marker in markers:
        if marker in ref:
            rel = ref.split(marker, 1)[1]
            candidate = Path(local_dir) / marker.strip("/") / rel
            if candidate.is_file():
                return candidate
    return None


def _synthetic_env_preview(
    sample: dict[str, Any],
    *,
    split: str,
    camera_name: str,
    index: int,
) -> np.ndarray:
    physics = sample.get("physics") if isinstance(sample.get("physics"), dict) else {}
    scene = sample.get("scene") if isinstance(sample.get("scene"), dict) else {}
    env_id = str(sample.get("env_id") or f"{split}-{index:04d}")
    friction = _safe_float(physics.get("friction"), 0.8)
    lighting = _safe_float(physics.get("lighting_lux"), 650.0)
    mass = _safe_float(physics.get("mass_scale"), 1.0)
    asset = str(scene.get("simready_asset") or "simready://unknown")
    augmented = str(scene.get("augmented_frame_uri") or "")
    accent = _color_from_text(f"{env_id}:{camera_name}:{asset}")
    rows = [
        f"{split.upper()} {env_id}",
        f"CAM {camera_name}",
        f"ASSET {Path(asset).name or asset.rsplit('/', 1)[-1]}",
        f"FRICTION {friction:.2f} MASS {mass:.2f}",
        f"LIGHT {lighting:.0f} LUX",
        "AUGMENTED REF" if augmented else "NO AUGMENT REF",
    ]
    image = _descriptor_preview_image(rows, accent=accent, seed_text=f"{env_id}:{asset}")
    _draw_env_glyph(image, accent=accent, friction=friction, lighting=lighting, mass=mass, wrist=camera_name == "wrist")
    return image


def _augmentation_preview(payload: dict[str, Any], *, index: int) -> np.ndarray:
    frame_id = str(payload.get("frame_id") or f"frame-{index:05d}")
    perturbation = str(payload.get("perturbation") or payload.get("mode") or "augmentation")
    status = str(payload.get("status") or "recorded")
    source = str(payload.get("source_dataset_uri") or payload.get("input_uri") or "")
    accent = _color_from_text(f"{frame_id}:{perturbation}:{status}")
    rows = [
        "COSMOS TRANSFER",
        frame_id.upper(),
        f"PERTURB {perturbation.upper()}",
        f"STATUS {status.upper()}",
        Path(source.rstrip("/")).name.upper() if source else "SOURCE RECORDED",
    ]
    image = _descriptor_preview_image(rows, accent=accent, seed_text=f"{frame_id}:{perturbation}")
    _draw_augmentation_glyph(image, accent=accent, seed_text=f"{frame_id}:{perturbation}")
    return image


def _descriptor_preview_image(
    rows: list[str],
    *,
    accent: tuple[int, int, int],
    seed_text: str,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = np.clip(34 + 42 * xx + accent[0] * 0.18, 0, 255).astype(np.uint8)
    base[:, :, 1] = np.clip(38 + 52 * yy + accent[1] * 0.14, 0, 255).astype(np.uint8)
    base[:, :, 2] = np.clip(46 + 28 * (1.0 - yy) + accent[2] * 0.18, 0, 255).astype(np.uint8)
    for x in range(0, width, 32):
        base[:, x : x + 1] = np.maximum(base[:, x : x + 1], 96)
    for y in range(0, height, 32):
        base[y : y + 1, :] = np.maximum(base[y : y + 1, :], 96)
    for _ in range(18):
        x0 = int(rng.integers(0, max(1, width - 24)))
        y0 = int(rng.integers(64, max(65, height - 20)))
        w = int(rng.integers(10, 34))
        h = int(rng.integers(8, 26))
        color = np.array(_mix_color(accent, int(rng.integers(50, 180))), dtype=np.uint8)
        base[y0 : min(height, y0 + h), x0 : min(width, x0 + w)] = color

    try:
        from PIL import Image, ImageDraw

        image = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 46), fill=(12, 18, 28))
        draw.rectangle((0, 46, width, 50), fill=accent)
        for row_index, row in enumerate(rows):
            y = 10 + row_index * 26
            if row_index == 0:
                draw.text((12, y), row[:32], fill=(255, 255, 255))
            else:
                draw.text((12, y + 36), row[:38], fill=(230, 236, 245))
        return np.asarray(image, dtype=np.uint8).copy()
    except Exception:
        base[0:48, :] = np.array([12, 18, 28], dtype=np.uint8)
        base[48:52, :] = np.array(accent, dtype=np.uint8)
        return base


def _draw_env_glyph(
    image: np.ndarray,
    *,
    accent: tuple[int, int, int],
    friction: float,
    lighting: float,
    mass: float,
    wrist: bool,
) -> None:
    height, width = image.shape[:2]
    left = int(width * 0.48)
    right = width - 16
    table_y0 = int(height * 0.54)
    table_y1 = int(height * 0.86)
    image[table_y0:table_y1, left:right] = np.array([96, 104, 116], dtype=np.uint8)
    cx = int(width * (0.72 if wrist else 0.65))
    cy = int(height * 0.66)
    arm = np.array([236, 238, 240], dtype=np.uint8)
    image[cy - 38 : cy + 38, cx - 8 : cx + 8] = arm
    image[cy - 8 : cy + 8, max(left, cx - 36) : min(right, cx + 36)] = arm
    block_size = int(14 + 10 * max(0.0, min(1.0, mass - 0.85)) / 0.3)
    bx = int(width * 0.82)
    by = int(height * 0.64)
    image[by : by + block_size, bx : bx + block_size] = np.array(accent, dtype=np.uint8)
    grip = int(10 + 20 * max(0.0, min(1.0, friction)))
    image[cy + 34 : cy + 40, max(left, cx - grip) : min(right, cx + grip)] = np.array([30, 220, 140], dtype=np.uint8)
    light_width = int(max(8, min(width - 24, lighting / 1200.0 * (width - 24))))
    image[height - 18 : height - 10, 12 : 12 + light_width] = np.array([250, 204, 21], dtype=np.uint8)


def _draw_augmentation_glyph(image: np.ndarray, *, accent: tuple[int, int, int], seed_text: str) -> None:
    height, width = image.shape[:2]
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    for index in range(5):
        x0 = int(width * 0.48) + index * 28
        y0 = 145 + int(rng.integers(-16, 16))
        image[y0 : y0 + 40, x0 : x0 + 22] = np.array(_mix_color(accent, 40 + index * 18), dtype=np.uint8)
        image[y0 + 40 : y0 + 46, x0 - 2 : x0 + 24] = np.array([18, 24, 38], dtype=np.uint8)
    for _ in range(30):
        x = int(rng.integers(int(width * 0.45), width - 12))
        y = int(rng.integers(82, height - 24))
        image[y : y + 3, x : x + 3] = np.array([255, 255, 255], dtype=np.uint8)


def _color_from_text(value: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return (72 + digest[0] % 152, 72 + digest[1] % 152, 72 + digest[2] % 152)


def _mix_color(accent: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, channel + amount - 90))) for channel in accent)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _markdown_table(headers: list[str], rows: list[Any]) -> str:
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = list(row if isinstance(row, (list, tuple)) else [row])
        table.append("| " + " | ".join(_escape_markdown(_format_value(value)) for value in values) + " |")
    return "\n".join(table)


def _escape_markdown(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _first_sample_value(inner_evidence: dict[str, Any], sample_key: str, value_key: str) -> Any:
    for record in inner_evidence.get("iterations") or []:
        if not isinstance(record, dict):
            continue
        sample = record.get(sample_key)
        if isinstance(sample, dict) and sample.get(value_key) is not None:
            return sample.get(value_key)
    return None


def _jsonl_samples(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if len(samples) >= limit:
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    samples.append(item)
    except OSError:
        pass
    return samples


def _artifact_counts(local_dir: Path) -> dict[str, int]:
    roots = {
        "augment_frames": Path(local_dir) / "augment" / "frames",
        "raw_env_shards": Path(local_dir) / "envs" / "raw",
        "heldout_renders": Path(local_dir) / "eval" / "heldout" / "renders",
        "rollout_dirs": Path(local_dir) / "actions" / "train",
        "vlm_eval_json": Path(local_dir) / "vlm_eval" / "train",
        "training_signal_json": Path(local_dir) / "training_signal" / "train",
    }
    counts: dict[str, int] = {}
    for name, root in roots.items():
        if not root.exists():
            counts[name] = 0
        elif name == "rollout_dirs":
            counts[name] = sum(1 for path in root.rglob("rollout-*") if path.is_dir())
        else:
            counts[name] = sum(1 for path in root.rglob("*") if path.is_file())
    return counts


def _key_artifacts(local_dir: Path) -> dict[str, str]:
    rels = [
        "reports/sim2real-report.json",
        "reports/sim2real-visual-index.json",
        "outer_loop/decision.json",
        "augment/manifest.json",
        "augment/frames/index.json",
        "envs/raw/raw-shard-00-of-16.jsonl",
        "envs/raw/raw-shard-00-summary.json",
        "envs/manifest/split-manifest.json",
        "envs/train/envs.jsonl",
        "envs/heldout/envs.jsonl",
        "tokens/manifest.json",
        "eval/heldout/report.json",
    ]
    return {rel: str(Path(local_dir) / rel) for rel in rels if (Path(local_dir) / rel).exists()}


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


def _log_heldout_pointclouds(
    rr: Any,
    recording: Any,
    frames: list[tuple[np.ndarray, np.ndarray]],
    counts: dict[str, int],
) -> int:
    """Log GPU-reconstructed held-out point clouds under ``world/heldout/points``.

    The Scene-overview Spatial3DView (contents ``world/**``) renders these on the
    client GPU, time-aligned with the ``/camera`` stream.
    """

    if not frames:
        return 0
    seconds = 0.0
    logged = 0
    for xyz, rgb in frames:
        _set_time(rr, recording, seconds)
        rr.log(
            "world/heldout/points",
            rr.Points3D(np.ascontiguousarray(xyz, dtype=np.float32), colors=rgb),
            recording=recording,
        )
        _bump(counts, "world/heldout/points")
        seconds += ROLLOUT_FRAME_SECONDS
        logged += 1
    return logged


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


# Sub-directory (under eval/heldout/renders) where the Isaac held-out eval writes
# GPU-derived colored point clouds (one .npz per rendered frame, world frame).
POINTCLOUD_SUBDIR = "_pointcloud"


def _heldout_pointcloud_frames(local_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load GPU-rendered held-out point clouds as ``(xyz[N,3], rgb[N,3])`` frames.

    Returns the primary env's per-frame clouds (time-aligned to the ``/camera``
    stream) so both the Rerun 3D view and the Lichtblick 3D panel show the
    reconstructed sim geometry. Empty when no point clouds were captured.
    """

    root = local_dir / "eval" / "heldout" / "renders" / POINTCLOUD_SUBDIR
    if not root.is_dir():
        return []
    env_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not env_dirs:
        return []
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for cloud_path in sorted(env_dirs[0].glob("cloud-*.npz")):
        cloud = _read_pointcloud_npz(cloud_path)
        if cloud is not None:
            frames.append(cloud)
    return frames


def _read_pointcloud_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32).reshape(-1, 3)
            rgb = np.asarray(data["rgb"], dtype=np.uint8).reshape(-1, 3)
    except (OSError, ValueError, KeyError):
        logging.getLogger(__name__).debug("unreadable point cloud %s", path, exc_info=True)
        return None
    count = min(xyz.shape[0], rgb.shape[0])
    if count == 0:
        return None
    return xyz[:count], rgb[:count]


def _build_blueprint(
    rrb: Any,
    *,
    has_heldout_cameras: bool = False,
    heldout_env_ids: list[str] | None = None,
    has_synthetic_data: bool = False,
) -> Any:
    env_ids = list(heldout_env_ids or [])
    has_heldout_cameras = has_heldout_cameras or bool(env_ids)
    if env_ids:
        # Keep a top-level camera alias first: the web viewer reliably opens this
        # single Spatial2DView, while the per-env grid remains available for
        # deeper inspection in the Streams tree.
        camera_view = rrb.Vertical(
            rrb.Spatial2DView(origin="camera", name="Isaac held-out simulation camera"),
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
    synthetic_view = rrb.Vertical(
        rrb.Spatial2DView(
            origin="synthetic/preview",
            name="Synthetic data preview",
        ),
        rrb.Grid(
            rrb.Spatial2DView(
                origin="synthetic/dataset/train",
                contents="synthetic/dataset/train/**",
                name="Train env samples",
            ),
            rrb.Spatial2DView(
                origin="synthetic/dataset/heldout",
                contents="synthetic/dataset/heldout/**",
                name="Held-out env samples",
            ),
            rrb.Spatial2DView(
                origin="synthetic/augmentation",
                contents="synthetic/augmentation/**",
                name="Augmentation samples",
            ),
            name="Synthetic dataset and augmentation",
        ),
        row_shares=[1.2, 2.0],
    )
    signal_view = rrb.Vertical(
        rrb.TimeSeriesView(origin="signal", contents="signal/**", name="VLM->RL signal"),
        rrb.TimeSeriesView(origin="heldout", contents="heldout/**", name="Held-out scores"),
    )
    summary_view = rrb.Vertical(
        rrb.TextDocumentView(origin="summary/run_success", name="Success"),
        rrb.TextDocumentView(origin="summary/augmentation", name="Augmented data"),
        rrb.TextDocumentView(origin="summary/artifacts", name="Artifacts"),
        rrb.TextDocumentView(origin="summary/vlm_critiques", name="VLM critiques"),
        row_shares=[1.2, 1.1, 1.0, 1.0],
    )
    if has_heldout_cameras:
        columns = [left_column]
        shares = [2.4]
        if has_synthetic_data:
            columns.append(synthetic_view)
            shares.append(2.1)
        columns.extend([summary_view, signal_view])
        shares.extend([1.5, 1.1])
        layout = rrb.Horizontal(*columns, column_shares=shares)
    else:
        columns = [
            rrb.Spatial3DView(origin="world", contents="world/**", name="Scene overview"),
            left_column,
        ]
        shares = [2.0, 1.4]
        if has_synthetic_data:
            columns.append(synthetic_view)
            shares.append(1.8)
        columns.extend([summary_view, signal_view])
        shares.extend([1.2, 1.3])
        layout = rrb.Horizontal(*columns, column_shares=shares)
    return rrb.Blueprint(
        layout,
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
    # Prefer Pillow when available: it decodes every PNG colour type, bit depth,
    # and row filter correctly. The hand-rolled decoder below is a dependency-free
    # fallback (in-pod finalize can lack Pillow) and MUST apply PNG row filters —
    # real renders use Sub/Up/Paeth, and ignoring them turns the image into noise.
    pil = _read_png_with_pillow(data)
    if pil is not None:
        return pil
    return _decode_png_bytes(data)


def _read_png_with_pillow(data: bytes) -> np.ndarray | None:
    try:
        import io

        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception:
        logging.getLogger(__name__).debug("Pillow PNG decode failed", exc_info=True)
        return None


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png_bytes(data: bytes) -> np.ndarray | None:
    """Minimal but correct 8-bit PNG decoder (applies row filters)."""

    import struct
    import zlib

    index = 8
    width = height = 0
    bit_depth = color_type = interlace = 0
    idat = bytearray()
    while index + 8 <= len(data):
        length = struct.unpack("!I", data[index : index + 4])[0]
        chunk_type = data[index + 4 : index + 8]
        chunk = data[index + 8 : index + 8 + length]
        index += 12 + length
        if chunk_type == b"IHDR" and len(chunk) >= 13:
            width, height = struct.unpack("!II", chunk[:8])
            bit_depth = int(chunk[8])
            color_type = int(chunk[9])
            interlace = int(chunk[12])
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if (
        width <= 0
        or height <= 0
        or not idat
        or bit_depth != 8
        or interlace != 0
        or channels is None
    ):
        return None
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error:
        return None
    row_len = width * channels
    stride = row_len + 1
    if len(raw) < height * stride:
        return None
    bpp = channels
    # Rows are carried as Python int lists, not numpy rows. Average and Paeth are
    # recurrences in x (each byte needs the reconstructed byte bpp to its left), so
    # they cannot be vectorized along the row; the only choice is how expensive each
    # scalar step is, and numpy scalar indexing is far dearer than list indexing.
    # Measured on a 256x256 RGB frame: Average 65ms -> 29ms, Paeth 119ms -> 76ms.
    # (Vectorizing across the bpp colour lanes instead was tried and is 2.5-4x
    # SLOWER than the original -- per-slice numpy overhead dwarfs 3-element math.)
    # Sub and Up are vectorizable, so those two still go through numpy.
    recon = np.zeros((height, row_len), dtype=np.uint8)
    prev: list[int] = [0] * row_len
    offset = 0
    for row in range(height):
        ftype = raw[offset]
        offset += 1
        line = raw[offset : offset + row_len]
        offset += row_len
        if ftype == 0:
            cur = list(line)
        elif ftype == 1:  # Sub: reconstructed == per-channel cumulative sum of raw
            arr = np.frombuffer(line, dtype=np.uint8).astype(np.int32)
            for c in range(bpp):
                arr[c::bpp] = np.cumsum(arr[c::bpp]) % 256
            cur = arr.tolist()
        elif ftype == 2:  # Up: depends only on the previous row, so vectorizable
            arr = np.frombuffer(line, dtype=np.uint8).astype(np.int32)
            cur = ((arr + np.asarray(prev, dtype=np.int32)) % 256).tolist()
        elif ftype == 3:  # Average (recurrence in x)
            cur = list(line)
            for i in range(row_len):
                left = cur[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + ((left + prev[i]) // 2)) & 0xFF
        elif ftype == 4:  # Paeth (recurrence in x)
            cur = list(line)
            for i in range(row_len):
                left = cur[i - bpp] if i >= bpp else 0
                up_left = prev[i - bpp] if i >= bpp else 0
                cur[i] = (cur[i] + _paeth(left, prev[i], up_left)) & 0xFF
        else:
            return None
        recon[row] = cur
        prev = cur
    pixels = recon.reshape(height, width, channels)
    if channels == 3:
        rgb = pixels
    elif channels == 4:
        rgb = pixels[..., :3]
    elif channels == 1:
        rgb = np.repeat(pixels, 3, axis=2)
    else:  # channels == 2 (gray + alpha)
        rgb = np.repeat(pixels[..., :1], 3, axis=2)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


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
        self.pointcloud_message_count = 0
        self.transform_message_count = 0

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

    def log_pointcloud(
        self, topic: str, points: Any, colors: Any, stamp_ns: int
    ) -> None:
        from npa.workbench.lichtblick import _POINTCLOUD_SCHEMA, pointcloud_message

        channel_id = self._channel(topic, "foxglove.PointCloud", _POINTCLOUD_SCHEMA)
        message = pointcloud_message(
            points, colors, stamp_ns=stamp_ns, frame_id=MCAP_FRAME_ID
        )
        self._add(topic, channel_id, message, stamp_ns)
        self.pointcloud_message_count += 1

    def log_transform(
        self,
        *,
        parent_frame_id: str,
        child_frame_id: str,
        stamp_ns: int,
        topic: str = MCAP_TF_TOPIC,
    ) -> None:
        """Emit a static (identity) ``foxglove.FrameTransform`` so the frame exists."""

        from npa.workbench.lichtblick import _FRAME_TRANSFORM_SCHEMA, frame_transform_message

        channel_id = self._channel(topic, "foxglove.FrameTransform", _FRAME_TRANSFORM_SCHEMA)
        message = frame_transform_message(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            stamp_ns=stamp_ns,
        )
        self._add(topic, channel_id, message, stamp_ns)
        self.transform_message_count += 1


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

    Whenever any camera frame is emitted, ``MCAP_PRIMARY_CAMERA_TOPIC`` is
    populated too (from the held-out episode when there is one, else mirrored from
    the first rollout), because the embedded viewer's default layout binds its
    Image panel to that one well-known topic.
    """

    writer_cls, compression_none = _import_mcap()
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
        # Uncompressed chunks: valid MCAP that needs no lz4/zstandard C-extension,
        # so the finalize stage works in minimal in-pod environments.
        writer = writer_cls(handle, compression=compression_none)
        writer.start(profile="", library="npa-sim2real")
        emitter = _McapEmitter(writer)

        # Publish a static world->sim2real transform first so the 3D panel has a
        # defined coordinate frame to place the held-out point cloud (without it,
        # a Foxglove-compatible 3D panel cannot render the cloud).
        emitter.log_transform(
            parent_frame_id=MCAP_ROOT_FRAME_ID,
            child_frame_id=MCAP_FRAME_ID,
            stamp_ns=0,
        )

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
                    # Held-out episodes own the primary camera topic when present.
                    # Without them (a run with rollout cameras only), mirror the
                    # first rollout onto it so the default layout's Image panel is
                    # never empty.
                    mirror_primary_camera=(
                        not has_heldout_cameras
                        and MCAP_PRIMARY_CAMERA_TOPIC not in emitter.channel_counts
                    ),
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
        _emit_mcap_pointclouds(
            emitter, _heldout_pointcloud_frames(local_dir), frame_period_ns=frame_period_ns
        )
        _emit_mcap_heldout_scores(emitter, heldout_report, heldout_period_ns=heldout_period_ns)

        writer.finish()

    if not output_mcap.exists() or output_mcap.stat().st_size == 0:
        raise Sim2RealVizError(f"MCAP recording was not written: {output_mcap}")
    # The transform is scaffolding (a coordinate frame), not content — a recording
    # with only a transform is still empty, so it must not satisfy this guard.
    content_total = (
        emitter.camera_message_count
        + emitter.scalar_message_count
        + emitter.log_message_count
        + emitter.pointcloud_message_count
    )
    if content_total == 0:
        raise Sim2RealVizError(
            "Sim2Real MCAP recording has no camera, signal, critique, or held-out content"
        )
    return Sim2RealMcapResult(
        status="written",
        output_mcap_path=str(output_mcap),
        channel_counts=emitter.channel_counts,
        message_count=content_total + emitter.transform_message_count,
        camera_message_count=emitter.camera_message_count,
        scalar_message_count=emitter.scalar_message_count,
        log_message_count=emitter.log_message_count,
        pointcloud_message_count=emitter.pointcloud_message_count,
        transform_message_count=emitter.transform_message_count,
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
    mirror_primary_camera: bool = False,
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
        if mirror_primary_camera:
            emitter.log_image_bytes(MCAP_PRIMARY_CAMERA_TOPIC, payload, fmt, stamp_ns)

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
                # Mirror the primary episode onto the well-known topic the default
                # layout binds to.
                emitter.log_image_bytes(MCAP_PRIMARY_CAMERA_TOPIC, payload, "png", stamp_ns)
            stamp_ns += frame_period_ns


def _emit_mcap_pointclouds(
    emitter: _McapEmitter,
    frames: list[tuple[np.ndarray, np.ndarray]],
    *,
    frame_period_ns: int,
) -> None:
    """Emit GPU-reconstructed held-out point clouds on ``/heldout/points``.

    Lichtblick renders ``foxglove.PointCloud`` in its GPU-accelerated 3D panel,
    time-aligned with the ``/camera`` stream.
    """

    stamp_ns = 0
    for xyz, rgb in frames:
        emitter.log_pointcloud("/heldout/points", xyz, rgb, stamp_ns)
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


def _import_mcap() -> tuple[Any, Any]:
    try:
        from mcap.writer import CompressionType, Writer
    except ImportError as exc:  # pragma: no cover
        raise McapUnavailableError(
            "mcap is not installed; skipping Sim2Real MCAP visualization"
        ) from exc
    return Writer, CompressionType.NONE


def _import_rerun() -> tuple[Any, Any]:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover
        raise RerunUnavailableError(
            "rerun-sdk is not installed; skipping Sim2Real Rerun visualization"
        ) from exc
    return rr, rrb
