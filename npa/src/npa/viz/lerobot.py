"""Shared LeRobotDataset loading and Unitree G1 visualization data helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM, G1_STATE_NAMES_43


DEFAULT_DURATION_CAP_S = 10.0
DEFAULT_FPS = 30
SUPPORTED_LAYOUTS = {"single", "side-by-side", "overlay"}
PREDICTION_LAYOUTS = {"side-by-side", "overlay"}

G1_JOINT_NAMES = list(G1_STATE_NAMES_43)
_G1_INDEX = {name: idx for idx, name in enumerate(G1_JOINT_NAMES)}


def _idx(name: str) -> int:
    return _G1_INDEX[name]


G1_JOINT_CONNECTIONS: list[tuple[int, int]] = [
    (_idx("waist_yaw_joint"), _idx("waist_roll_joint")),
    (_idx("waist_roll_joint"), _idx("waist_pitch_joint")),
    (_idx("waist_yaw_joint"), _idx("left_hip_yaw_joint")),
    (_idx("left_hip_yaw_joint"), _idx("left_hip_roll_joint")),
    (_idx("left_hip_roll_joint"), _idx("left_hip_pitch_joint")),
    (_idx("left_hip_pitch_joint"), _idx("left_knee_joint")),
    (_idx("left_knee_joint"), _idx("left_ankle_pitch_joint")),
    (_idx("left_ankle_pitch_joint"), _idx("left_ankle_roll_joint")),
    (_idx("waist_yaw_joint"), _idx("right_hip_yaw_joint")),
    (_idx("right_hip_yaw_joint"), _idx("right_hip_roll_joint")),
    (_idx("right_hip_roll_joint"), _idx("right_hip_pitch_joint")),
    (_idx("right_hip_pitch_joint"), _idx("right_knee_joint")),
    (_idx("right_knee_joint"), _idx("right_ankle_pitch_joint")),
    (_idx("right_ankle_pitch_joint"), _idx("right_ankle_roll_joint")),
    (_idx("waist_pitch_joint"), _idx("left_shoulder_pitch_joint")),
    (_idx("left_shoulder_pitch_joint"), _idx("left_shoulder_roll_joint")),
    (_idx("left_shoulder_roll_joint"), _idx("left_shoulder_yaw_joint")),
    (_idx("left_shoulder_yaw_joint"), _idx("left_elbow_pitch_joint")),
    (_idx("left_elbow_pitch_joint"), _idx("left_elbow_roll_joint")),
    (_idx("left_elbow_roll_joint"), _idx("left_wrist_pitch_joint")),
    (_idx("left_wrist_pitch_joint"), _idx("left_wrist_yaw_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_pinky_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_ring_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_middle_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_index_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_thumb_bend_joint")),
    (_idx("left_hand_thumb_bend_joint"), _idx("left_hand_thumb_rotation_joint")),
    (_idx("left_wrist_yaw_joint"), _idx("left_hand_aux_joint")),
    (_idx("waist_pitch_joint"), _idx("right_shoulder_pitch_joint")),
    (_idx("right_shoulder_pitch_joint"), _idx("right_shoulder_roll_joint")),
    (_idx("right_shoulder_roll_joint"), _idx("right_shoulder_yaw_joint")),
    (_idx("right_shoulder_yaw_joint"), _idx("right_elbow_pitch_joint")),
    (_idx("right_elbow_pitch_joint"), _idx("right_elbow_roll_joint")),
    (_idx("right_elbow_roll_joint"), _idx("right_wrist_pitch_joint")),
    (_idx("right_wrist_pitch_joint"), _idx("right_wrist_yaw_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_pinky_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_ring_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_middle_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_index_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_thumb_bend_joint")),
    (_idx("right_hand_thumb_bend_joint"), _idx("right_hand_thumb_rotation_joint")),
    (_idx("right_wrist_yaw_joint"), _idx("right_hand_aux_joint")),
]


class VizDataError(Exception):
    """Raised when visualization inputs cannot be loaded or normalized."""


@dataclass(frozen=True)
class RenderInputs:
    skeleton_data: np.ndarray
    predictions_data: np.ndarray | None
    duration_s: float
    source_fps: int
    title: str
    frame_indices: np.ndarray


def validate_layout_predictions(layout: str, predictions_data: np.ndarray | None) -> None:
    if layout not in SUPPORTED_LAYOUTS:
        raise VizDataError(f"Unsupported layout '{layout}'. Expected one of: {', '.join(sorted(SUPPORTED_LAYOUTS))}")
    if layout in PREDICTION_LAYOUTS and predictions_data is None:
        raise VizDataError(f"--predictions-path is required when --layout={layout}")


def parse_resolution(value: str) -> tuple[int, int]:
    raw = value.lower().strip()
    if "x" not in raw:
        raise VizDataError(f"Resolution must use WIDTHxHEIGHT format, got: {value}")
    width_s, height_s = raw.split("x", 1)
    try:
        width = int(width_s)
        height = int(height_s)
    except ValueError as exc:
        raise VizDataError(f"Resolution must use integer WIDTHxHEIGHT format, got: {value}") from exc
    if width <= 0 or height <= 0:
        raise VizDataError(f"Resolution dimensions must be positive, got: {value}")
    return width, height


def load_render_inputs(
    input_path: Path,
    *,
    predictions_path: Path | None = None,
    layout: str = "single",
    duration_s: float | None = None,
    output_fps: int = DEFAULT_FPS,
) -> RenderInputs:
    """Load LeRobot and optional GR00T predictions into backend-ready arrays."""
    if output_fps <= 0:
        raise VizDataError(f"fps must be positive, got {output_fps}")
    state_vectors, source_fps, title = load_lerobot_state_vectors(input_path)
    resolved_duration = resolve_duration_s(
        frame_count=state_vectors.shape[0],
        source_fps=source_fps,
        requested_duration_s=duration_s,
    )
    selected_states, frame_indices = select_frames(
        state_vectors,
        source_fps=source_fps,
        output_fps=output_fps,
        duration_s=resolved_duration,
    )
    skeleton_data = g1_state_vectors_to_skeleton(selected_states)

    predictions_data = None
    if predictions_path is not None:
        predictions_data = load_predictions_skeleton(
            predictions_path,
            source_fps=source_fps,
            output_fps=output_fps,
            duration_s=resolved_duration,
            target_joint_count=skeleton_data.shape[1],
        )

    validate_layout_predictions(layout, predictions_data)
    return RenderInputs(
        skeleton_data=skeleton_data,
        predictions_data=predictions_data,
        duration_s=resolved_duration,
        source_fps=source_fps,
        title=title,
        frame_indices=frame_indices,
    )


def load_lerobot_state_vectors(root: Path) -> tuple[np.ndarray, int, str]:
    """Read ``observation.state`` rows from a standard LeRobotDataset directory."""
    root = Path(root)
    if not root.exists():
        raise VizDataError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise VizDataError(f"Input path must be a LeRobotDataset directory: {root}")

    data_dir = root / "data"
    if not data_dir.exists():
        raise VizDataError(f"LeRobotDataset data directory is missing: {data_dir}")
    parquet_paths = sorted(
        path for path in data_dir.rglob("*.parquet") if not path.name.startswith("._")
    )
    if not parquet_paths:
        raise VizDataError(f"No parquet data files found under {data_dir}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise VizDataError("pyarrow is required to read LeRobotDataset parquet files") from exc

    rows: list[list[float]] = []
    indices: list[int] = []
    saw_index = True
    for path in parquet_paths:
        table = pq.read_table(path)
        if "observation.state" not in table.column_names:
            raise VizDataError(f"{path} has no observation.state column")
        rows.extend(table["observation.state"].to_pylist())
        if "index" in table.column_names:
            indices.extend(int(v) for v in table["index"].to_pylist())
        else:
            saw_index = False

    if not rows:
        raise VizDataError(f"No observation.state rows found under {data_dir}")
    if saw_index and len(indices) == len(rows):
        rows = [row for _idx_value, row in sorted(zip(indices, rows), key=lambda item: item[0])]

    state = np.asarray(rows, dtype=np.float32)
    if state.ndim != 2:
        raise VizDataError(f"observation.state must be 2D after loading, got shape {state.shape}")
    if state.shape[1] != G1_STATE_DIM:
        raise VizDataError(f"Expected {G1_STATE_DIM}D Unitree G1 state vectors, got {state.shape[1]}D")

    info = _read_info(root)
    fps = int(info.get("fps") or DEFAULT_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS
    title = _read_task_title(root) or str(info.get("task") or info.get("robot_type") or root.name)
    return state, fps, title


def resolve_duration_s(
    *,
    frame_count: int,
    source_fps: int,
    requested_duration_s: float | None,
    duration_cap_s: float = DEFAULT_DURATION_CAP_S,
) -> float:
    if frame_count <= 0:
        raise VizDataError("Cannot render an empty trajectory")
    if source_fps <= 0:
        raise VizDataError(f"source_fps must be positive, got {source_fps}")
    if requested_duration_s is not None:
        if requested_duration_s <= 0:
            raise VizDataError(f"duration must be positive, got {requested_duration_s}")
        return float(requested_duration_s)
    return min(frame_count / float(source_fps), duration_cap_s)


def select_frames(
    data: np.ndarray,
    *,
    source_fps: int,
    output_fps: int,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[0] <= 0:
        raise VizDataError("Cannot select frames from an empty array")
    if source_fps <= 0 or output_fps <= 0:
        raise VizDataError(f"fps values must be positive, got source={source_fps} output={output_fps}")
    if duration_s <= 0:
        raise VizDataError(f"duration must be positive, got {duration_s}")

    target_frames = max(1, int(round(duration_s * output_fps)))
    source_duration = data.shape[0] / float(source_fps)
    if source_duration >= duration_s:
        indices = np.rint(np.linspace(0, data.shape[0] - 1, target_frames)).astype(np.int64)
    else:
        indices = np.floor(np.arange(target_frames, dtype=np.float64) * source_fps / output_fps).astype(np.int64)
        indices = np.clip(indices, 0, data.shape[0] - 1)
    return data[indices], indices


def load_predictions_skeleton(
    path: Path,
    *,
    source_fps: int,
    output_fps: int,
    duration_s: float,
    target_joint_count: int,
) -> np.ndarray:
    predictions = _load_prediction_array(Path(path))
    if predictions.ndim == 3 and predictions.shape[-1] == 3:
        skeleton = predictions.astype(np.float32, copy=False)
    elif predictions.ndim >= 2 and predictions.shape[-1] == G1_STATE_DIM:
        state_vectors = predictions.reshape(-1, G1_STATE_DIM).astype(np.float32, copy=False)
        skeleton = g1_state_vectors_to_skeleton(state_vectors)
    else:
        raise VizDataError(
            "Predictions must be either G1 state/action vectors with last dimension "
            f"{G1_STATE_DIM} or skeleton positions shaped [T, J, 3]; got {predictions.shape}"
        )

    selected, _indices = select_frames(
        skeleton,
        source_fps=source_fps,
        output_fps=output_fps,
        duration_s=duration_s,
    )
    if selected.shape[1] != target_joint_count:
        raise VizDataError(
            f"Prediction joint count {selected.shape[1]} does not match input joint count {target_joint_count}"
        )
    return selected


def g1_state_vectors_to_skeleton(state_vectors: np.ndarray) -> np.ndarray:
    state = np.asarray(state_vectors, dtype=np.float32)
    if state.ndim != 2:
        raise VizDataError(f"G1 state vectors must be 2D [T, {G1_STATE_DIM}], got {state.shape}")
    if state.shape[1] != G1_STATE_DIM:
        raise VizDataError(f"Expected {G1_STATE_DIM}D G1 state vectors, got {state.shape[1]}D")
    poses = np.empty((state.shape[0], G1_STATE_DIM, 3), dtype=np.float32)
    for frame, row in enumerate(state):
        poses[frame] = _g1_pose_from_state(row)
    return poses


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise VizDataError(f"Invalid LeRobotDataset info.json: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _read_task_title(root: Path) -> str:
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return ""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    except Exception:
        return ""
    if "task" not in table.column_names or table.num_rows == 0:
        return ""
    value = table["task"].to_pylist()[0]
    return str(value) if value else ""


def _load_prediction_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise VizDataError(f"Predictions path does not exist: {path}")
    if path.is_dir():
        preferred = [
            path / "predicted_actions.npz",
            path / "predictions.npz",
            path / "predictions.json",
            path / "npa_groot_infer_results.json",
        ]
        for candidate in preferred:
            if candidate.exists():
                return _load_prediction_array(candidate)
        for candidate in sorted(path.rglob("*.npz")):
            return _load_prediction_array(candidate)
        for candidate in sorted(path.rglob("*.json")):
            return _load_prediction_array(candidate)
        raise VizDataError(f"No .npz or .json prediction artifacts found under {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise VizDataError(f"No arrays found in {path}")
        key = "trajectory_0" if "trajectory_0" in data.files else data.files[0]
        return np.asarray(data[key], dtype=np.float32)
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise VizDataError(f"Invalid prediction JSON {path}: {exc}") from exc
        arr = _first_numeric_array(payload)
        if arr is None:
            raise VizDataError(f"No numeric prediction array found in {path}")
        return arr.astype(np.float32, copy=False)
    raise VizDataError(f"Unsupported predictions file type: {path}")


def _first_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, list):
        try:
            arr = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.ndim >= 2 and arr.size:
            return arr
        for item in value:
            found = _first_numeric_array(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for key in ("predictions", "predicted_actions", "actions", "trajectory_0", "data"):
            if key in value:
                found = _first_numeric_array(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _first_numeric_array(item)
            if found is not None:
                return found
    return None


def _g1_pose_from_state(state: np.ndarray) -> np.ndarray:
    angles = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pose = np.zeros((G1_STATE_DIM, 3), dtype=np.float32)

    pelvis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    waist_rot = _rot_z(_angle(angles[_idx("waist_yaw_joint")]) * 0.35) @ _rot_y(
        _angle(angles[_idx("waist_pitch_joint")]) * 0.35
    ) @ _rot_x(_angle(angles[_idx("waist_roll_joint")]) * 0.35)
    waist_roll = pelvis + waist_rot @ np.array([0.0, 0.0, 0.20], dtype=np.float32)
    chest = pelvis + waist_rot @ np.array([0.0, 0.0, 0.47], dtype=np.float32)
    pose[_idx("waist_yaw_joint")] = pelvis
    pose[_idx("waist_roll_joint")] = waist_roll
    pose[_idx("waist_pitch_joint")] = chest

    _fill_leg(
        pose,
        angles,
        side="left",
        sign=1.0,
        pelvis=pelvis,
    )
    _fill_leg(
        pose,
        angles,
        side="right",
        sign=-1.0,
        pelvis=pelvis,
    )
    _fill_arm(
        pose,
        angles,
        side="left",
        sign=1.0,
        chest=chest,
        trunk_rotation=waist_rot,
    )
    _fill_arm(
        pose,
        angles,
        side="right",
        sign=-1.0,
        chest=chest,
        trunk_rotation=waist_rot,
    )
    return pose


def _fill_leg(
    pose: np.ndarray,
    angles: np.ndarray,
    *,
    side: str,
    sign: float,
    pelvis: np.ndarray,
) -> None:
    hip_yaw_i = _idx(f"{side}_hip_yaw_joint")
    hip_roll_i = _idx(f"{side}_hip_roll_joint")
    hip_pitch_i = _idx(f"{side}_hip_pitch_joint")
    knee_i = _idx(f"{side}_knee_joint")
    ankle_pitch_i = _idx(f"{side}_ankle_pitch_joint")
    ankle_roll_i = _idx(f"{side}_ankle_roll_joint")

    hip = pelvis + np.array([0.13 * sign, 0.0, -0.03], dtype=np.float32)
    hip_yaw = _angle(angles[hip_yaw_i])
    hip_roll = _angle(angles[hip_roll_i]) * sign
    hip_pitch = _angle(angles[hip_pitch_i])
    knee_bend = abs(_angle(angles[knee_i]))
    ankle_pitch = _angle(angles[ankle_pitch_i])
    ankle_roll = _angle(angles[ankle_roll_i]) * sign

    hip_rot = _rot_z(hip_yaw * 0.45) @ _rot_x(hip_roll * 0.55) @ _rot_y(hip_pitch * 0.70)
    knee = hip + hip_rot @ np.array([0.02 * sign, 0.02, -0.42], dtype=np.float32)
    lower_rot = _rot_z(hip_yaw * 0.35) @ _rot_x(hip_roll * 0.30) @ _rot_y((hip_pitch + knee_bend) * 0.55)
    ankle = knee + lower_rot @ np.array([0.0, 0.02, -0.41], dtype=np.float32)
    foot = ankle + (_rot_z(hip_yaw * 0.25) @ _rot_x(ankle_roll * 0.40) @ _rot_y(ankle_pitch * 0.50)) @ np.array(
        [0.03 * sign, 0.20, -0.03], dtype=np.float32
    )

    pose[hip_yaw_i] = hip
    pose[hip_roll_i] = hip + (knee - hip) * 0.20
    pose[hip_pitch_i] = hip + (knee - hip) * 0.38
    pose[knee_i] = knee
    pose[ankle_pitch_i] = ankle
    pose[ankle_roll_i] = foot


def _fill_arm(
    pose: np.ndarray,
    angles: np.ndarray,
    *,
    side: str,
    sign: float,
    chest: np.ndarray,
    trunk_rotation: np.ndarray,
) -> None:
    shoulder_pitch_i = _idx(f"{side}_shoulder_pitch_joint")
    shoulder_roll_i = _idx(f"{side}_shoulder_roll_joint")
    shoulder_yaw_i = _idx(f"{side}_shoulder_yaw_joint")
    elbow_pitch_i = _idx(f"{side}_elbow_pitch_joint")
    elbow_roll_i = _idx(f"{side}_elbow_roll_joint")
    wrist_pitch_i = _idx(f"{side}_wrist_pitch_joint")
    wrist_yaw_i = _idx(f"{side}_wrist_yaw_joint")

    shoulder = chest + trunk_rotation @ np.array([0.23 * sign, 0.0, -0.03], dtype=np.float32)
    shoulder_pitch = _angle(angles[shoulder_pitch_i])
    shoulder_roll = _angle(angles[shoulder_roll_i]) * sign
    shoulder_yaw = _angle(angles[shoulder_yaw_i])
    elbow_pitch = abs(_angle(angles[elbow_pitch_i]))
    elbow_roll = _angle(angles[elbow_roll_i]) * sign
    wrist_pitch = _angle(angles[wrist_pitch_i])
    wrist_yaw = _angle(angles[wrist_yaw_i])

    shoulder_rot = trunk_rotation @ _rot_z(shoulder_yaw * 0.45) @ _rot_x(shoulder_roll * 0.50) @ _rot_y(
        shoulder_pitch * 0.65
    )
    elbow = shoulder + shoulder_rot @ np.array([0.08 * sign, 0.02, -0.31], dtype=np.float32)
    forearm_rot = shoulder_rot @ _rot_x(elbow_roll * 0.40) @ _rot_y(elbow_pitch * 0.60)
    wrist = elbow + forearm_rot @ np.array([0.07 * sign, 0.01, -0.28], dtype=np.float32)
    palm = wrist + forearm_rot @ _rot_z(wrist_yaw * 0.35) @ _rot_y(wrist_pitch * 0.35) @ np.array(
        [0.05 * sign, 0.04, -0.03], dtype=np.float32
    )

    pose[shoulder_pitch_i] = shoulder
    pose[shoulder_roll_i] = shoulder + (elbow - shoulder) * 0.22
    pose[shoulder_yaw_i] = shoulder + (elbow - shoulder) * 0.45
    pose[elbow_pitch_i] = elbow
    pose[elbow_roll_i] = elbow + (wrist - elbow) * 0.25
    pose[wrist_pitch_i] = wrist
    pose[wrist_yaw_i] = palm

    spread = [
        ("pinky", -0.045),
        ("ring", -0.022),
        ("middle", 0.0),
        ("index", 0.024),
    ]
    for name, offset in spread:
        pose[_idx(f"{side}_hand_{name}_joint")] = palm + forearm_rot @ np.array(
            [0.04 * sign, 0.06, offset * sign], dtype=np.float32
        )
    thumb = palm + forearm_rot @ np.array([0.07 * sign, 0.02, 0.05], dtype=np.float32)
    pose[_idx(f"{side}_hand_thumb_bend_joint")] = thumb
    pose[_idx(f"{side}_hand_thumb_rotation_joint")] = thumb + forearm_rot @ np.array(
        [0.035 * sign, 0.02, 0.03], dtype=np.float32
    )
    pose[_idx(f"{side}_hand_aux_joint")] = palm + forearm_rot @ np.array(
        [0.02 * sign, 0.02, -0.03], dtype=np.float32
    )


def _angle(value: float) -> float:
    return float(np.tanh(float(value)) * 1.15)


def _rot_x(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
