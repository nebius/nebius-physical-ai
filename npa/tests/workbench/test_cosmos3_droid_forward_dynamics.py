"""Contract tests for the Cosmos3 DROID forward-dynamics stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from npa.workbench.cosmos.droid_forward_dynamics import (
    CHECKPOINT_REVISION,
    CONTROLS_SCHEMA,
    EVALUATION_SCHEMA,
    PREDICTION_SCHEMA,
    PREPARED_SCHEMA,
    SELECTION_SCHEMA,
    VISUALIZATION_SCHEMA,
    DroidForwardDynamicsError,
    _native_inference_argv,
    _native_raw_actions,
    _write_video,
    evaluate_droid_forward_dynamics,
    prepare_droid_forward_dynamics,
    visualize_droid_forward_dynamics,
)
from npa.workbench.cosmos.policy_artifacts import file_digest, publish_bundle


def _image_entry(path: Path) -> dict[str, object]:
    return {"uri": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}


def _selection(tmp_path: Path) -> Path:
    frames = []
    for index in range(17):
        views = {}
        for name, color in (
            ("wrist_image_left", (index * 7, 30, 80)),
            ("exterior_image_1_left", (40, index * 8, 120)),
            ("exterior_image_2_left", (70, 100, index * 9)),
        ):
            image_path = tmp_path / f"{index:02d}_{name}.png"
            Image.new("RGB", (640, 360), color).save(image_path)
            views[name] = _image_entry(image_path)
        frames.append({"views": views})
    poses = [
        {
            "position_m": [index * 0.01, 0.0, 0.1],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "gripper_position": index / 32.0,
        }
        for index in range(17)
    ]
    selection = {
        "schema": SELECTION_SCHEMA,
        "source": {
            "dataset_repository": "nvidia/Cosmos3-DROID",
            "dataset_version": "droid_plus_lerobot_640x360_20260412",
            "revision": "test-revision",
        },
        "episode_id": "scene-test/episode-001",
        "heldout_split": {
            "method": "scene_or_building",
            "group_key": "scene_id",
            "group_id": "scene-test",
        },
        "frames": frames,
        "poses_abs": poses,
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    return path


def _prediction_artifacts(prepared_dir: Path, output: Path) -> tuple[Path, Path]:
    reference = prepared_dir / "reference_composite.mp4"
    frames = []
    # The test source uses solid fields. These real, newly encoded MP4s differ
    # from each other and exercise the decoded-metric path instead of manifests.
    for index in range(16):
        frames.append(Image.new("RGB", (640, 540), (20 + index, 30, 40)))
    true_video = output / "true-source.mp4"
    perm_video = output / "perm-source.mp4"
    zero_video = output / "zero-source.mp4"
    _write_video(frames, true_video)
    _write_video([Image.new("RGB", (640, 540), (70, index, 10)) for index in range(16)], perm_video)
    _write_video([Image.new("RGB", (640, 540), (10, 70, index)) for index in range(16)], zero_video)

    true_root = output / "true"
    true_root.mkdir()
    (true_root / "prediction_true.mp4").write_bytes(true_video.read_bytes())
    true = publish_bundle(
        true_root,
        str(output / "true-published"),
        {
            "schema": PREDICTION_SCHEMA,
            "status": "succeeded",
            "native": {"checkpoint_identity": {"revision": CHECKPOINT_REVISION}},
        },
        "prediction.json",
    )
    controls_root = output / "controls"
    controls_root.mkdir()
    (controls_root / "prediction_permuted.mp4").write_bytes(perm_video.read_bytes())
    (controls_root / "prediction_zero.mp4").write_bytes(zero_video.read_bytes())
    controls = publish_bundle(
        controls_root,
        str(output / "controls-published"),
        {"schema": CONTROLS_SCHEMA, "status": "succeeded", "controls": {}},
        "controls.json",
    )
    assert reference.is_file()
    return Path(true["artifacts"][0]["uri"]).parent / "prediction.json", Path(
        controls["artifacts"][0]["uri"]
    ).parent / "controls.json"


def test_prepare_evaluate_and_visualize_use_real_media_bytes(tmp_path: Path) -> None:
    selection_path = _selection(tmp_path)
    prepared_output = tmp_path / "prepared-published"
    prepared = prepare_droid_forward_dynamics(
        input_path=str(selection_path), output_path=str(prepared_output)
    )
    assert prepared["schema"] == PREPARED_SCHEMA
    actions = json.loads((prepared_output / "actions_true.json").read_text())
    assert actions["domain_name"] == "droid_lerobot"
    assert len(actions["actions"]) == 16
    assert all(len(action) == 64 and action[9] != 0.0 for action in actions["actions"])

    prediction_path, controls_path = _prediction_artifacts(prepared_output, tmp_path / "predictions")
    evaluation_output = tmp_path / "evaluation-published"
    evaluation = evaluate_droid_forward_dynamics(
        prepared_path=str(prepared_output / "prepared.json"),
        prediction_path=str(prediction_path),
        controls_path=str(controls_path),
        output_path=str(evaluation_output),
    )
    assert evaluation["schema"] == EVALUATION_SCHEMA
    assert len(evaluation["visual_error"]["frame_metrics"]) == 16
    assert evaluation["action_sensitivity"]["true_vs_temporal_permutation_mse"] > 0.0
    assert evaluation["claims"]["heldout_benchmark"] is False

    visualization_output = tmp_path / "visualization-published"
    visualization = visualize_droid_forward_dynamics(
        prepared_path=str(prepared_output / "prepared.json"),
        prediction_path=str(prediction_path),
        controls_path=str(controls_path),
        evaluation_path=str(evaluation_output / "evaluation.json"),
        output_path=str(visualization_output),
    )
    assert visualization["schema"] == VISUALIZATION_SCHEMA
    assert (visualization_output / "droid_forward_dynamics.rrd").stat().st_size > 64
    provenance = visualization["provenance"]
    assert provenance["upstream"]["derivative_checkpoint"]["publisher"] == "jere-mybao"
    assert provenance["upstream"]["framework"]["notice_url"].endswith("/NOTICE")
    assert len(provenance["npa_modifications"]) == 3


def test_prepare_rejects_an_unrecognized_heldout_split(tmp_path: Path) -> None:
    path = _selection(tmp_path)
    selection = json.loads(path.read_text())
    selection["heldout_split"] = {"method": "random"}
    path.write_text(json.dumps(selection))
    with pytest.raises(DroidForwardDynamicsError, match="heldout_split"):
        prepare_droid_forward_dynamics(input_path=str(path), output_path=str(tmp_path / "out"))


def test_native_argv_uses_the_card_contract(tmp_path: Path) -> None:
    repo = tmp_path / "framework"
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").touch()
    argv = _native_inference_argv(
        repo=repo,
        checkpoint=tmp_path / "checkpoint",
        input_json=tmp_path / "input.json",
        action_path=tmp_path / "actions.json",
        output_dir=tmp_path / "out",
        seed=7,
    )
    assert argv[:3] == [str(repo / ".venv/bin/python"), "-m", "cosmos_framework.scripts.inference"]
    assert argv[argv.index("--domain-name") + 1] == "droid_lerobot"
    assert argv[argv.index("--action-chunk-size") + 1] == "16"
    assert argv[argv.index("-i") + 1].endswith("input.json")
    assert argv[argv.index("--seed") + 1] == "7"


def test_relative_action_contract_is_not_accidentally_normalized(tmp_path: Path) -> None:
    path = _selection(tmp_path)
    prepared = prepare_droid_forward_dynamics(input_path=str(path), output_path=str(tmp_path / "out"))
    action_data = json.loads((tmp_path / "out/actions_true.json").read_text())
    first = np.asarray(action_data["actions"][0])
    assert first.shape == (64,)
    assert first[0] == pytest.approx(0.01)
    assert np.allclose(first[10:], 0.0)
    assert prepared["action_contract"]["normalization"] == "none"


def test_native_action_file_uses_raw_width_and_framework_does_the_padding(tmp_path: Path) -> None:
    prepare_droid_forward_dynamics(input_path=str(_selection(tmp_path)), output_path=str(tmp_path / "out"))
    action_data = json.loads((tmp_path / "out/actions_true.json").read_text())
    raw = _native_raw_actions(action_data["actions"])
    assert len(raw) == 16
    assert all(len(action) == 10 for action in raw)
    assert raw[0] == action_data["actions"][0][:10]
