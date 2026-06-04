"""Convert GR00T prediction artifacts to overlay-compatible Rerun recordings."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from npa.adapter.isaac_lab_lerobot import G1_STATE_DIM
from npa.viz.adapters.lerobot_to_rerun import (
    APPLICATION_ID,
    RerunAdapterError,
    _build_blueprint,
    _import_rerun,
    _is_s3_uri,
    _log_angle_series_styles,
    _log_frame,
    _materialize_dataset,
    _materialize_output,
    _normalize_color,
    _select_adapter_frames,
    _set_time_seconds,
    _storage_client,
)
from npa.viz.lerobot import (
    REAL_G1_ACTION_DIM,
    VizDataError,
    _load_prediction_array,
    g1_state_vectors_to_skeleton,
    load_lerobot_state_vectors,
    real_g1_action_vectors_to_g1_state_vectors,
)


INPUT_ENTITY_ROOT = "world/skeleton"
PREDICTIONS_ENTITY_ROOT = "world/predictions"


def groot_predictions_to_rerun(
    predictions_path: str | Path,
    input_dataset_path: str | Path,
    output_rrd_path: Path,
    input_color: tuple[int, int, int] = (0, 217, 255),
    predictions_color: tuple[int, int, int] = (255, 136, 0),
    duration_s: float | None = None,
) -> None:
    """Write one Rerun ``.rrd`` with LeRobot input and GR00T predictions overlaid."""
    output_ref = str(output_rrd_path)
    with ExitStack() as stack:
        local_predictions = _materialize_predictions(predictions_path, stack)
        local_dataset = _materialize_dataset(input_dataset_path, stack)
        local_output = _materialize_output(output_ref, stack)
        _write_groot_overlay_recording(
            local_predictions,
            local_dataset,
            local_output,
            input_color=input_color,
            predictions_color=predictions_color,
            duration_s=duration_s,
        )
        if _is_s3_uri(output_ref):
            _storage_client(output_ref).upload_file(str(local_output), output_ref)


def _write_groot_overlay_recording(
    predictions_path: Path,
    input_dataset_path: Path,
    output_rrd_path: Path,
    *,
    input_color: tuple[int, int, int],
    predictions_color: tuple[int, int, int],
    duration_s: float | None,
) -> None:
    rr, rrb = _import_rerun()
    input_rgb = _normalize_color(input_color)
    predictions_rgb = _normalize_color(predictions_color)
    input_states, source_fps, _title = load_lerobot_state_vectors(input_dataset_path)
    selected_input_states, _input_indices, _resolved_duration_s = _select_adapter_frames(
        input_states,
        fps=source_fps,
        duration_s=duration_s,
    )
    input_skeleton = g1_state_vectors_to_skeleton(selected_input_states)
    prediction_skeleton, prediction_states = _load_prediction_frames(
        predictions_path,
    )
    if prediction_skeleton.shape[1] != input_skeleton.shape[1]:
        raise RerunAdapterError(
            "Prediction joint count must match input joint count: "
            f"{prediction_skeleton.shape[1]} != {input_skeleton.shape[1]}"
        )
    if prediction_states.shape[1] != selected_input_states.shape[1]:
        raise RerunAdapterError(
            "Prediction angle state width must match input state width: "
            f"{prediction_states.shape[1]} != {selected_input_states.shape[1]}"
        )
    if prediction_skeleton.shape[0] != prediction_states.shape[0]:
        raise RerunAdapterError(
            "Prediction skeleton and angle state frame counts must match: "
            f"{prediction_skeleton.shape[0]} != {prediction_states.shape[0]}"
        )
    if prediction_skeleton.shape[0] > input_skeleton.shape[0]:
        raise RerunAdapterError(
            "Prediction frame count cannot exceed input frame count after sampling: "
            f"{prediction_skeleton.shape[0]} > {input_skeleton.shape[0]}"
        )

    output_rrd_path = Path(output_rrd_path)
    if output_rrd_path.suffix.lower() != ".rrd":
        raise RerunAdapterError(f"Rerun output path must end in .rrd, got: {output_rrd_path}")
    output_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = _build_blueprint(rrb)
    recording = rr.RecordingStream(APPLICATION_ID)
    rr.save(output_rrd_path, default_blueprint=blueprint, recording=recording)
    rr.send_blueprint(blueprint, recording=recording)
    _log_angle_series_styles(rr, recording, INPUT_ENTITY_ROOT, input_rgb)
    _log_angle_series_styles(rr, recording, PREDICTIONS_ENTITY_ROOT, predictions_rgb)
    prediction_frame_count = int(prediction_skeleton.shape[0])
    for frame_idx in range(input_skeleton.shape[0]):
        _set_time_seconds(rr, recording, frame_idx / float(source_fps))
        _log_frame(
            rr,
            recording,
            INPUT_ENTITY_ROOT,
            input_skeleton[frame_idx],
            selected_input_states[frame_idx],
            input_rgb,
        )
        if frame_idx < prediction_frame_count:
            _log_frame(
                rr,
                recording,
                PREDICTIONS_ENTITY_ROOT,
                prediction_skeleton[frame_idx],
                prediction_states[frame_idx],
                predictions_rgb,
            )
    rr.disconnect(recording=recording)

    if not output_rrd_path.exists() or output_rrd_path.stat().st_size == 0:
        raise RerunAdapterError(f"Rerun recording was not written: {output_rrd_path}")


def _load_prediction_frames(
    predictions_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        predictions = _load_prediction_array(Path(predictions_path))
    except VizDataError as exc:
        raise RerunAdapterError(str(exc)) from exc
    if predictions.ndim == 3 and predictions.shape[-1] == 3:
        skeleton = predictions.astype(np.float32, copy=False)
        states = np.zeros((skeleton.shape[0], G1_STATE_DIM), dtype=np.float32)
    elif predictions.ndim >= 2 and predictions.shape[-1] == G1_STATE_DIM:
        states = predictions.reshape(-1, G1_STATE_DIM).astype(np.float32, copy=False)
        skeleton = g1_state_vectors_to_skeleton(states)
    elif predictions.ndim >= 2 and predictions.shape[-1] == REAL_G1_ACTION_DIM:
        action_vectors = predictions.reshape(-1, REAL_G1_ACTION_DIM).astype(np.float32, copy=False)
        states = real_g1_action_vectors_to_g1_state_vectors(action_vectors)
        skeleton = g1_state_vectors_to_skeleton(states)
    else:
        raise RerunAdapterError(
            "Predictions must be either G1 state vectors with last dimension "
            f"{G1_STATE_DIM}, REAL_G1 action vectors with last dimension {REAL_G1_ACTION_DIM}, "
            f"or skeleton positions shaped [T, J, 3]; got {predictions.shape}"
        )
    if skeleton.shape[0] <= 0:
        raise RerunAdapterError("Predictions must contain at least one frame")
    return skeleton, states


def _materialize_predictions(predictions_path: str | Path, stack: ExitStack) -> Path:
    predictions_ref = str(predictions_path)
    if not _is_s3_uri(predictions_ref):
        return Path(predictions_path)
    temp_dir = stack.enter_context(TemporaryDirectory(prefix="npa-rerun-predictions-"))
    return Path(_storage_client(predictions_ref).download_path(predictions_ref, temp_dir))
