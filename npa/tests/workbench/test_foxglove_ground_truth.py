"""Reject malformed rollout evidence before writing and decode valid MCAP status."""

import json
from pathlib import Path

import pytest

from npa.workbench.foxglove.mcap_writer import (
    McapWriteError,
    MetricsInput,
    write_run_mcap,
)

pytest.importorskip("mcap")
FLAGS = ("contact", "stable_grasp", "gripper_closed", "placement_stable")
INVALID = (None, "false", "true", "", 0, 1, 0.0, 1.0, [], [True], {}, {"value": True})


def _rollout(path: Path, records: list[dict]) -> MetricsInput:
    path.write_text(
        json.dumps({"schema": "npa.sim2real.action_rollout.v1", "actions": records})
    )
    return MetricsInput(path=path, name=path.stem)


@pytest.mark.parametrize("field", FLAGS)
@pytest.mark.parametrize("invalid", INVALID)
@pytest.mark.parametrize(
    "mask", [{}, {"placement_stable": True}, {"termination_reason": "success"}]
)
def test_malformed_flags_reject_before_output(tmp_path, field, invalid, mask):
    ground_truth = {**mask, field: invalid}
    valid = {"action": [0.1], "simulator_ground_truth": {"contact": True}}
    first = _rollout(tmp_path / "first.json", [valid])
    second = _rollout(
        tmp_path / "second.json",
        [valid, {"action": [0.2], "simulator_ground_truth": ground_truth}],
    )
    output = tmp_path / "new" / "out.mcap"
    with pytest.raises(McapWriteError, match=rf"second.json: actions\[1\].*{field}"):
        write_run_mcap(output=output, metrics=[first, second])
    assert not output.parent.exists()
    output.parent.mkdir()
    output.write_bytes(b"existing recording")
    with pytest.raises(McapWriteError, match=field):
        write_run_mcap(output=output, metrics=[first, second])
    assert output.read_bytes() == b"existing recording"


@pytest.mark.parametrize("invalid", [None, False, True, "", "false", 0, 1, [], [True]])
@pytest.mark.parametrize("action", [[0.1], None, ["invalid"]])
def test_malformed_container_is_not_masked_by_skipped_action(tmp_path, invalid, action):
    metric = _rollout(
        tmp_path / "rollout.json",
        [{"action": action, "simulator_ground_truth": invalid}],
    )
    output = tmp_path / "out.mcap"
    with pytest.raises(
        McapWriteError, match="simulator_ground_truth must be an object"
    ):
        write_run_mcap(output=output, metrics=[metric])
    assert not output.exists()


@pytest.mark.parametrize(
    ("source", "phase"),
    [
        ({}, "tracking"),
        ({"simulator_ground_truth": {}}, "tracking"),
        ({"simulator_ground_truth": dict.fromkeys(FLAGS, False)}, "tracking"),
        *(
            ({"simulator_ground_truth": {flag: True}}, phase)
            for flag, phase in zip(FLAGS, ("contact", "lift", "grasp", "complete"))
        ),
        ({"simulator_ground_truth": {"termination_reason": "success"}}, "complete"),
        ({"simulator_ground_truth": {"termination_reason": "complete"}}, "complete"),
    ],
)
def test_real_mcap_preserves_flags_defaults_and_completion_reasons(
    tmp_path, source, phase
):
    from mcap.reader import make_reader

    records = [{"action": [0.1], **source}, {"action": [0.2]}]
    metric = _rollout(tmp_path / "rollout.json", records)
    output = tmp_path / "out.mcap"
    write_run_mcap(output=output, metrics=[metric])
    with output.open("rb") as handle:
        states = [
            json.loads(message.data)
            for _, channel, message in make_reader(handle).iter_messages()
            if channel.topic == "/run/state"
        ]
    assert [state["phase"] for state in states] == [phase, "finished"]
    truth = source.get("simulator_ground_truth", {})
    for flag in FLAGS:
        assert states[0][flag] is truth.get(flag, False)
        assert states[1][flag] is False
    # Success remains measured placement evidence, not an inferred reason.
    assert states[0]["success"] is truth.get("placement_stable", False)
    assert states[0]["termination_reason"] == truth.get("termination_reason", "running")
