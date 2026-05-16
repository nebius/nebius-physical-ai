"""matplotlib MP4 renderer for LeRobot trajectory visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


BACKGROUND = "#1a1a1a"
INPUT_COLOR = "#00d9ff"
PREDICTION_COLOR = "#ff8800"
TEXT_COLOR = "#f2f2f2"
MUTED_TEXT_COLOR = "#b8b8b8"


class MatplotlibRenderError(Exception):
    """Raised when matplotlib cannot render the requested visualization."""


@dataclass
class _SkeletonArtists:
    lines: list[object]
    joints: object

    @property
    def all(self) -> list[object]:
        return [*self.lines, self.joints]


def render(
    skeleton_data,
    predictions_data,
    layout: str,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    duration_s: float,
    title: str,
    joint_connections: list[tuple[int, int]],
) -> None:
    """Render skeleton trajectory data to an MP4 with matplotlib FuncAnimation."""
    skeleton = np.asarray(skeleton_data, dtype=np.float32)
    predictions = None if predictions_data is None else np.asarray(predictions_data, dtype=np.float32)
    _validate_inputs(skeleton, predictions, layout, output_path, resolution, fps, duration_s, joint_connections)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    width, height = resolution
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=BACKGROUND)
    fig.suptitle(title or "LeRobot trajectory", color=TEXT_COLOR, fontsize=18, y=0.98)

    axes_3d, trace_ax = _build_layout(fig, layout)
    bounds_data = [skeleton]
    if predictions is not None:
        bounds_data.append(predictions)
    limits = _axis_limits(bounds_data)

    artists: list[_SkeletonArtists] = []
    if layout == "side-by-side":
        input_artists = _add_skeleton(axes_3d[0], skeleton[0], joint_connections, INPUT_COLOR, "Input")
        pred_artists = _add_skeleton(axes_3d[1], predictions[0], joint_connections, PREDICTION_COLOR, "Predictions")
        artists.extend([input_artists, pred_artists])
        _style_3d_axis(axes_3d[0], "Isaac Lab input", limits)
        _style_3d_axis(axes_3d[1], "GR00T predictions", limits)
    else:
        input_artists = _add_skeleton(axes_3d[0], skeleton[0], joint_connections, INPUT_COLOR, "Input")
        artists.append(input_artists)
        if layout == "overlay" and predictions is not None:
            pred_artists = _add_skeleton(axes_3d[0], predictions[0], joint_connections, PREDICTION_COLOR, "Predictions")
            artists.append(pred_artists)
        _style_3d_axis(axes_3d[0], "Overlay" if layout == "overlay" else "Trajectory", limits)

    time_marker = _add_motion_trace(trace_ax, skeleton, predictions, fps=fps, duration_s=duration_s)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.08, hspace=0.16, wspace=0.04)

    def update(frame: int) -> list[object]:
        _update_skeleton(artists[0], skeleton[frame], joint_connections)
        if layout == "side-by-side":
            if frame < int(predictions.shape[0]):
                _set_skeleton_visible(artists[1], True)
                _update_skeleton(artists[1], predictions[frame], joint_connections)
            else:
                _set_skeleton_visible(artists[1], False)
        elif layout == "overlay" and predictions is not None and len(artists) > 1:
            if frame < int(predictions.shape[0]):
                _set_skeleton_visible(artists[1], True)
                _update_skeleton(artists[1], predictions[frame], joint_connections)
            else:
                _set_skeleton_visible(artists[1], False)
        time_marker.set_xdata([frame / fps, frame / fps])
        updated: list[object] = [time_marker]
        for skeleton_artists in artists:
            updated.extend(skeleton_artists.all)
        return updated

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=skeleton.shape[0],
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer, dpi=dpi)
    if hasattr(ani, "_draw_was_started"):
        ani._draw_was_started = True
    plt.close(fig)


def _validate_inputs(
    skeleton: np.ndarray,
    predictions: np.ndarray | None,
    layout: str,
    output_path: Path,
    resolution: tuple[int, int],
    fps: int,
    duration_s: float,
    joint_connections: list[tuple[int, int]],
) -> None:
    if layout not in {"single", "side-by-side", "overlay"}:
        raise MatplotlibRenderError(f"Unsupported layout '{layout}'")
    if layout in {"side-by-side", "overlay"} and predictions is None:
        raise MatplotlibRenderError(f"predictions_data is required for layout '{layout}'")
    if skeleton.ndim != 3 or skeleton.shape[-1] != 3:
        raise MatplotlibRenderError(f"skeleton_data must have shape [T, J, 3], got {skeleton.shape}")
    if skeleton.shape[0] == 0 or skeleton.shape[1] == 0:
        raise MatplotlibRenderError("skeleton_data must contain at least one frame and one joint")
    if predictions is not None and (predictions.ndim != 3 or predictions.shape[-1] != 3):
        raise MatplotlibRenderError(
            f"predictions_data must have shape [T, J, 3], got {predictions.shape}"
        )
    if predictions is not None and predictions.shape[1:] != skeleton.shape[1:]:
        raise MatplotlibRenderError(
            "predictions_data joint shape must match skeleton_data joint shape: "
            f"{predictions.shape[1:]} != {skeleton.shape[1:]}"
        )
    if predictions is not None and predictions.shape[0] > skeleton.shape[0]:
        raise MatplotlibRenderError(
            "predictions_data frame count cannot exceed skeleton_data frame count: "
            f"{predictions.shape[0]} > {skeleton.shape[0]}"
        )
    if not joint_connections:
        raise MatplotlibRenderError("joint_connections must not be empty")
    max_joint = skeleton.shape[1] - 1
    for start, end in joint_connections:
        if start < 0 or end < 0 or start > max_joint or end > max_joint:
            raise MatplotlibRenderError(f"joint connection {(start, end)} is outside skeleton joint range 0..{max_joint}")
    if output_path.suffix.lower() != ".mp4":
        raise MatplotlibRenderError(f"matplotlib backend writes MP4 only, got: {output_path}")
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise MatplotlibRenderError(f"resolution dimensions must be positive, got: {resolution}")
    if fps <= 0:
        raise MatplotlibRenderError(f"fps must be positive, got {fps}")
    if duration_s <= 0:
        raise MatplotlibRenderError(f"duration_s must be positive, got {duration_s}")


def _build_layout(fig, layout: str):
    if layout == "side-by-side":
        grid = fig.add_gridspec(2, 2, height_ratios=[4.5, 1.2], hspace=0.08, wspace=0.02)
        left_ax = fig.add_subplot(grid[0, 0], projection="3d")
        right_ax = fig.add_subplot(grid[0, 1], projection="3d")
        trace_ax = fig.add_subplot(grid[1, :])
        return [left_ax, right_ax], trace_ax
    grid = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.15], hspace=0.08)
    main_ax = fig.add_subplot(grid[0, 0], projection="3d")
    trace_ax = fig.add_subplot(grid[1, 0])
    return [main_ax], trace_ax


def _add_skeleton(ax, frame: np.ndarray, connections: list[tuple[int, int]], color: str, label: str) -> _SkeletonArtists:
    lines = []
    for start, end in connections:
        line = ax.plot(
            [frame[start, 0], frame[end, 0]],
            [frame[start, 1], frame[end, 1]],
            [frame[start, 2], frame[end, 2]],
            color=color,
            linewidth=2.4,
            alpha=0.92,
            solid_capstyle="round",
            label=label if not lines else None,
        )[0]
        lines.append(line)
    joints = ax.scatter(
        frame[:, 0],
        frame[:, 1],
        frame[:, 2],
        s=14,
        c=color,
        alpha=0.90,
        depthshade=False,
    )
    return _SkeletonArtists(lines=lines, joints=joints)


def _update_skeleton(
    artists: _SkeletonArtists,
    frame: np.ndarray,
    connections: list[tuple[int, int]],
) -> None:
    for line, (start, end) in zip(artists.lines, connections):
        line.set_data([frame[start, 0], frame[end, 0]], [frame[start, 1], frame[end, 1]])
        line.set_3d_properties([frame[start, 2], frame[end, 2]])
    artists.joints._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])


def _set_skeleton_visible(artists: _SkeletonArtists, visible: bool) -> None:
    for line in artists.lines:
        line.set_visible(visible)
    artists.joints.set_visible(visible)


def _style_3d_axis(ax, title: str, limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect(
        (
            limits[0][1] - limits[0][0],
            limits[1][1] - limits[1][0],
            limits[2][1] - limits[2][0],
        )
    )
    ax.view_init(elev=15, azim=-72)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_facecolor(BACKGROUND)
    ax.yaxis.pane.set_facecolor(BACKGROUND)
    ax.zaxis.pane.set_facecolor(BACKGROUND)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    ax.xaxis.line.set_color(BACKGROUND)
    ax.yaxis.line.set_color(BACKGROUND)
    ax.zaxis.line.set_color(BACKGROUND)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(MUTED_TEXT_COLOR)


def _axis_limits(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    combined = np.concatenate([arr.reshape(-1, 3) for arr in arrays], axis=0)
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 0.6)
    radius = float(np.max(span) / 2.0) * 1.15
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def _add_motion_trace(trace_ax, skeleton: np.ndarray, predictions: np.ndarray | None, *, fps: int, duration_s: float):
    trace_ax.set_facecolor(BACKGROUND)
    trace_ax.set_xlim(0.0, duration_s)
    trace_ax.set_title("Representative joint motion", color=MUTED_TEXT_COLOR, fontsize=10, pad=2)
    time = np.arange(skeleton.shape[0], dtype=np.float32) / fps
    for trace in _representative_traces(skeleton):
        trace_ax.plot(time, trace, color=INPUT_COLOR, linewidth=1.3, alpha=0.62)
    if predictions is not None:
        prediction_time = np.arange(predictions.shape[0], dtype=np.float32) / fps
        for trace in _representative_traces(predictions):
            trace_ax.plot(
                prediction_time,
                trace,
                color=PREDICTION_COLOR,
                linewidth=1.3,
                alpha=0.60,
                linestyle="--",
            )
    trace_ax.tick_params(axis="x", colors=MUTED_TEXT_COLOR, labelsize=8)
    trace_ax.tick_params(axis="y", colors=MUTED_TEXT_COLOR, labelsize=8)
    for spine in trace_ax.spines.values():
        spine.set_color("#333333")
    trace_ax.grid(False)
    trace_ax.set_yticks([])
    trace_ax.set_xlabel("seconds", color=MUTED_TEXT_COLOR, fontsize=8)
    return trace_ax.axvline(0.0, color=TEXT_COLOR, linewidth=1.0, alpha=0.85)


def _representative_traces(skeleton: np.ndarray) -> list[np.ndarray]:
    joint_count = skeleton.shape[1]
    sample_joints = [min(idx, joint_count - 1) for idx in (6, 7, 18, 32)]
    traces = []
    for joint in sample_joints:
        values = skeleton[:, joint, 2]
        spread = float(values.max() - values.min())
        if spread > 1e-6:
            values = (values - values.min()) / spread
        else:
            values = values * 0.0
        traces.append(values)
    return traces
