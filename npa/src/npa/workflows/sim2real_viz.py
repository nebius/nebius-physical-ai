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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


APPLICATION_ID = "npa_sim2real_loop"
TIMELINE = "frame_time"
ROLLOUT_FRAME_SECONDS = 0.5
HELDOUT_STEP_SECONDS = 1.0
CRITIQUE_COLOR = (255, 136, 0, 255)


class Sim2RealVizError(Exception):
    """Raised when the Sim2Real Rerun emitter cannot produce a recording."""


class RerunUnavailableError(Sim2RealVizError):
    """Raised when the ``rerun`` SDK is not importable (caller WARNs and skips)."""


@dataclass(frozen=True)
class Sim2RealVizResult:
    """Result of emitting a Sim2Real Rerun recording."""

    status: str
    output_rrd_path: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    rollout_count: int = 0
    frame_count: int = 0
    heldout_env_count: int = 0
    mp4_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_rrd_path": self.output_rrd_path,
            "entity_counts": dict(self.entity_counts),
            "rollout_count": self.rollout_count,
            "frame_count": self.frame_count,
            "heldout_env_count": self.heldout_env_count,
            "mp4_paths": list(self.mp4_paths),
        }


def emit_sim2real_rerun(
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    heldout_report: dict[str, Any] | None,
    output_rrd: Path | None = None,
    write_mp4: bool = False,
) -> Sim2RealVizResult:
    """Write ``reports/sim2real.rrd`` for a completed run's artifacts.

    Raises ``RerunUnavailableError`` when the ``rerun`` SDK is missing so the
    caller can WARN and continue. Any other failure (rerun present but logging
    failed) raises ``Sim2RealVizError`` so a broken recording is never silently
    accepted.
    """

    rr, rrb = _import_rerun()
    local_dir = Path(local_dir)
    output_rrd = Path(output_rrd) if output_rrd is not None else local_dir / "reports" / "sim2real.rrd"
    if output_rrd.suffix.lower() != ".rrd":
        raise Sim2RealVizError(f"Rerun output path must end in .rrd, got: {output_rrd}")
    output_rrd.parent.mkdir(parents=True, exist_ok=True)

    blueprint = _build_blueprint(rrb)
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd, default_blueprint=blueprint, recording=recording)
    _send_blueprint(rr, blueprint, recording)

    counts: dict[str, int] = {}
    seconds = 0.0
    rollout_count = 0
    frame_count = 0
    mp4_paths: list[str] = []
    critique_panel_rows: list[str] = []

    iterations = inner_evidence.get("iterations") or []
    for record in iterations:
        iteration = int(record.get("iteration", len(mp4_paths) + 1))
        actions_dir = _maybe_path(record.get("actions_dir"))
        eval_dir = _maybe_path(record.get("vlm_eval_dir"))
        signal_dir = _maybe_path(record.get("signal_dir"))
        for rollout_dir in _rollout_dirs(actions_dir):
            rollout_id = rollout_dir.name
            iter_root = f"rollouts/iter_{iteration:02d}/{rollout_id}"
            evaluation = _read_json(eval_dir / f"{rollout_id}.json") if eval_dir else {}
            signal = _read_json(signal_dir / f"{rollout_id}.json") if signal_dir else {}
            manifest = _read_json(rollout_dir / "manifest.json")
            frames = _rollout_frames(rollout_dir)
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
    if frame_count == 0 and rollout_count == 0 and heldout_env_count == 0:
        raise Sim2RealVizError(
            "Sim2Real Rerun recording has no rollout frames, signal, or held-out content"
        )

    return Sim2RealVizResult(
        status="written",
        output_rrd_path=str(output_rrd),
        entity_counts=counts,
        rollout_count=rollout_count,
        frame_count=frame_count,
        heldout_env_count=heldout_env_count,
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
        rr.log(f"{root}/camera", rr.Image(frame), recording=recording)
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


def _build_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="rollouts", contents="rollouts/**", name="Rollout cameras"),
            rrb.TextDocumentView(origin="rollouts", contents="rollouts/**/critique", name="VLM critiques"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="signal", contents="signal/**", name="VLM->RL signal"),
                rrb.TimeSeriesView(origin="heldout", contents="heldout/**", name="Held-out scores"),
            ),
            column_shares=[2.0, 1.5, 1.5],
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
        frame = _read_ppm(frame_path)
        if frame is not None:
            frames.append(frame)
    return frames


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
    index += 1  # single whitespace separator after maxval
    pixels = data[index:index + width * height * 3]
    if len(pixels) < width * height * 3:
        return None
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()


def _maybe_write_mp4(rollout_dir: Path, frames: list[np.ndarray]) -> Path | None:
    """Best-effort per-rollout MP4 via ffmpeg; returns None if ffmpeg is missing."""

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


def _write_png(path: Path, frame: np.ndarray) -> None:
    import struct
    import zlib

    height, width = int(frame.shape[0]), int(frame.shape[1])
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(frame[row].tobytes())

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + tag + payload + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


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
            pass


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


def _import_rerun() -> tuple[Any, Any]:
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RerunUnavailableError(
            "rerun-sdk is not installed; skipping Sim2Real Rerun visualization"
        ) from exc
    return rr, rrb
