"""Contract tests for the pinned OpenDM DM05/LIBERO workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows import dm05_opendm


ROOT = Path(__file__).parents[3]
SPEC = ROOT / "workflows" / "testing" / "dm05-opendm.yaml"


def test_normalization_contract_is_explicit_and_fixed() -> None:
    assert dm05_opendm.normalization_contract() == {
        "dataset": "libero_pi0_all",
        "robot_type": "Franka",
        "camera_keys": ["images_1", "images_2"],
        "camera_prompts_in_order": ["Head", "Left wrist"],
        "state": {"dimension": 8, "order": "six_joint_then_two_gripper"},
        "action": {"dimension": 7, "mode": "absolute"},
        "action_chunk": 10,
    }
    dm05_opendm.assert_normalization_contract(
        state=[0.0] * 8, image_count=2, action_dim=7
    )
    assert dm05_opendm.evaluator_observation_contract() == {
        "source": "Dexbotic LIBERO evaluator",
        "dimension": 8,
        "order": "eef_position_3_then_axis_angle_3_then_gripper_2",
        "camera_order": ["agentview_image", "robot0_eye_in_hand_image"],
        "http_api": "v1",
        "training_state_equivalent": False,
        "model_add_state": False,
    }


@pytest.mark.parametrize(
    ("state", "image_count", "action_dim", "message"),
    [
        ([0.0] * 7, 2, 7, "8-value Franka state"),
        ([0.0] * 8, 1, 7, "camera order"),
        ([0.0] * 8, 2, 8, "requires 7 action values"),
    ],
)
def test_normalization_contract_rejects_shape_drift(
    state: list[float], image_count: int, action_dim: int, message: str
) -> None:
    with pytest.raises(dm05_opendm.DM05WorkflowError, match=message):
        dm05_opendm.assert_normalization_contract(
            state=state, image_count=image_count, action_dim=action_dim
        )


def test_first_libero_observation_is_safe_and_uses_ordered_cameras(
    tmp_path: Path,
) -> None:
    data = tmp_path / "libero"
    image_root = data / "libero_pi0_all" / "image"
    jsonl = data / "libero_pi0_all" / "jsonl" / "episode.jsonl"
    jsonl.parent.mkdir(parents=True)
    (image_root / "episode").mkdir(parents=True)
    (image_root / "episode" / "head.jpg").write_bytes(b"head")
    (image_root / "episode" / "wrist.jpg").write_bytes(b"wrist")
    jsonl.write_text(
        json.dumps(
            {
                "state": [0.0] * 8,
                "prompt": "Pick up the object",
                "images_1": {"type": "image", "url": "episode/head.jpg"},
                "images_2": {"type": "image", "url": "episode/wrist.jpg"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frame, images = dm05_opendm._first_libero_observation(data)
    payload = dm05_opendm._request_payload(frame, images)

    assert [path.name for path in images] == ["head.jpg", "wrist.jpg"]
    assert payload["observation"]["robot_type"] == "Franka"
    assert list(payload["observation"]["images"]) == ["1", "2"]


def test_workflow_is_a_connected_five_stage_real_component_path() -> None:
    document = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    states = document["states"]
    assert list(states) == [
        "prepare_libero",
        "train_dm05",
        "serve_rollout",
        "evaluate_libero",
        "emit_reviewable_artifacts",
    ]
    assert all(
        "python3 -m npa.workflows.dm05_opendm" in state["run"]["shell"]
        for state in states.values()
    )
    assert (
        states["train_dm05"]["inputs"][0]["uri"] == "{{config.prepared_manifest_uri}}"
    )
    assert (
        states["serve_rollout"]["inputs"][1]["uri"]
        == "{{config.checkpoint_manifest_uri}}"
    )
    assert (
        states["evaluate_libero"]["inputs"][2]["uri"]
        == "{{config.rollout_manifest_uri}}"
    )
    assert (
        states["emit_reviewable_artifacts"]["inputs"][3]["uri"]
        == "{{config.evaluation_manifest_uri}}"
    )
    assert (
        states["emit_reviewable_artifacts"]["outputs"][1]["uri"] == "{{config.rrd_uri}}"
    )
    adapter_source = (ROOT / "npa/src/npa/workflows/dm05_opendm.py").read_text(
        encoding="utf-8"
    )
    assert "example_dm05_libero.yaml" in adapter_source
    assert '"api_style",\n                    "v1"' in adapter_source

    plan = build_plan(load_spec(SPEC), run_id="dm05-contract")
    assert [step.state for step in plan.steps] == list(states)


def test_workflow_does_not_invent_an_openpi_or_generic_terms_gate() -> None:
    contents = SPEC.read_text(encoding="utf-8") + (
        ROOT / "npa/src/npa/workflows/dm05_opendm.py"
    ).read_text(encoding="utf-8")
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in contents
    assert "ACCEPT_" not in contents
    assert "Dexmal/DM05" in contents
    assert "Dexmal/libero" in contents
    assert "dexbotic-benchmark" in contents
