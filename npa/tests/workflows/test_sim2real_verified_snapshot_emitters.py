"""Real CPU encoders/decoders for verified inputs; not GPU policy evidence."""

import base64
import copy
import hashlib
import io
import json

import pytest
from mcap.reader import make_reader
from PIL import Image
from rerun.chunk import RrdReader

import npa.workflows.sim2real.stage14_finalize as stage14
import npa.workflows.sim2real_rerun_regen as regen
import npa.workflows.sim2real_viz as viz
from test_sim2real_stage14_fifteenth_review_controls import _materialized_state
from test_sim2real_stage14_thirteenth_review_controls import _config, _decision

pytestmark = pytest.mark.usefixtures("operator_sim2real_image_defaults")


def _state(tmp_path):
    state = _materialized_state(tmp_path)
    relative = "eval/gold-heldout/outer-01/renders"
    state.gold["local_renders_dir"] = relative
    frame = state.local / relative / "env-1/camera-000.png"
    Image.new("RGB", (2, 2), (31, 71, 113)).save(frame)
    payload = frame.read_bytes()
    state.gold["render_manifest"]["frame_artifacts"]["env-1/camera-000.png"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    candidate = state.evidence["checkpoint_candidates"][0]
    candidate["validation_report"] = {
        "policy_checkpoint": candidate["checkpoint_uri"],
        "policy_checkpoint_sha256": candidate["checkpoint_sha256"],
        "evaluation_split": "validation",
        "success_rate": 0.0,
    }
    state.evidence["checkpoint_selection"] = copy.deepcopy(candidate)
    state.evidence["iterations"][0]["update"] = {
        "checkpoint_path": candidate["checkpoint_uri"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
    }
    actions = state.local / "actions/train/outer-01/iter-01"
    rollout = actions / "rollout-unit"
    rollout.mkdir(parents=True)
    Image.new("RGB", (2, 2), (17, 18, 19)).save(rollout / "camera-000.png")
    state.evidence["iterations"][0]["actions_dir"] = str(actions)
    dataset = state.local / "envs/train/envs.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"env_id":"unit-descriptor"}\n')
    return state, frame, payload


def _assert_native_pixels(rrd_path, mcap_path):
    reader = RrdReader(rrd_path)
    store = reader.store(store=reader.recordings()[0]).stream().collect()
    table = store.reader(index="frame_time", contents="/camera").to_arrow_table()
    buffers = [
        value
        for field in table.schema
        if (field.metadata or {}).get(b"rerun:component_type")
        == b"rerun.components.ImageBuffer"
        for value in table[field.name].to_pylist()
        if value is not None
    ]
    assert buffers, table.schema
    expected = bytes((31, 71, 113)) * 4
    assert all(bytes(buffer[0]) == expected for buffer in buffers)
    with mcap_path.open("rb") as handle:
        images = [
            json.loads(message.data)
            for _, channel, message in make_reader(
                handle, validate_crcs=True
            ).iter_messages()
            if channel.topic == viz.MCAP_PRIMARY_CAMERA_TOPIC
        ]
    assert images
    for image in images:
        with Image.open(io.BytesIO(base64.b64decode(image["data"]))) as decoded:
            assert decoded.convert("RGB").tobytes() == expected


@pytest.mark.parametrize("absolute_alias", [False, True])
def test_stage14_real_emitters_consume_snapshot_not_replaced_original(
    tmp_path, absolute_alias
):
    state, frame, _ = _state(tmp_path)
    if absolute_alias:
        state.gold["local_renders_dir"] = str(frame.parent.parent)
    original_gold = copy.deepcopy(state.gold)
    snapshot = stage14._validate_stage14_materialized_frames(state)
    Image.new("RGB", (2, 2), (201, 202, 203)).save(frame)
    reports = state.local / "reports"
    reports.mkdir()
    report = {
        "checkpoint_selection": state.evidence["checkpoint_selection"],
        "outer_loop": {"decision": _decision(state.gold)},
    }
    rrd, mcap = stage14._emit_stage14_outputs(state, [], report, {}, reports, snapshot)
    assert rrd.heldout_frame_count == mcap.heldout_frame_count == 1
    assert rrd.rollout_count == 1
    assert rrd.synthetic_frame_count > 0
    _assert_native_pixels(reports / "sim2real.rrd", reports / "sim2real.mcap")
    assert state.gold == original_gold


def test_regeneration_mcap_independently_uses_verified_snapshot(tmp_path):
    state, frame, _ = _state(tmp_path)
    inputs = regen._RegenInputs(
        state.evidence,
        state.gold,
        tmp_path / "heldout.json",
        state.local / "reports/sim2real-report.json",
        {},
        _decision(state.gold),
    )
    snapshot = regen._validate_regen_materialized_frames(
        _config(), state.local, inputs, snapshot_dir=state.local / ".verified-regen"
    )
    Image.new("RGB", (2, 2), (201, 202, 203)).save(frame)
    regen_state = regen._RegenState(
        state.evidence, state.gold, inputs.report_path, {}, {}, None, snapshot
    )
    result = regen._emit_regen_mcap(state.local, regen_state)
    assert result["status"] == "written"
    with (state.local / "reports/sim2real.mcap").open("rb") as handle:
        messages = [
            json.loads(message.data)
            for _, channel, message in make_reader(
                handle, validate_crcs=True
            ).iter_messages()
            if channel.topic == viz.MCAP_PRIMARY_CAMERA_TOPIC
        ]
    assert len(messages) == 1
    with Image.open(io.BytesIO(base64.b64decode(messages[0]["data"]))) as image:
        assert image.convert("RGB").tobytes() == bytes((31, 71, 113)) * 4


@pytest.mark.parametrize("absolute_alias", [False, True])
def test_regeneration_real_emitters_consume_canonical_snapshot(
    tmp_path, absolute_alias
):
    state, frame, _ = _state(tmp_path)
    if absolute_alias:
        state.gold["local_renders_dir"] = str(frame.parent.parent)
    original_gold = copy.deepcopy(state.gold)
    inputs = regen._RegenInputs(
        state.evidence,
        state.gold,
        tmp_path / "heldout.json",
        state.local / "reports/sim2real-report.json",
        {},
        _decision(state.gold),
    )
    snapshot = regen._validate_regen_materialized_frames(
        _config(), state.local, inputs, snapshot_dir=state.local / ".verified-regen"
    )
    Image.new("RGB", (2, 2), (201, 202, 203)).save(frame)
    regen_state = regen._RegenState(
        state.evidence, state.gold, inputs.report_path, {}, {}, None, snapshot
    )
    rrd_path = state.local / "reports/sim2real.rrd"
    regen._emit_regen_rrd(_config(), state.local, rrd_path, regen_state, [], "")
    mcap = regen._emit_regen_mcap(state.local, regen_state)
    assert mcap["status"] == "written"
    _assert_native_pixels(rrd_path, rrd_path.with_suffix(".mcap"))
    assert state.gold == original_gold


@pytest.mark.parametrize("absolute_alias", [False, True])
def test_regeneration_snapshot_with_symlinked_work_directory_ancestor(
    tmp_path, absolute_alias
):
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    state, frame, _ = _state(alias)
    if absolute_alias:
        state.gold["local_renders_dir"] = str(frame.parent.parent)
    original_gold = copy.deepcopy(state.gold)
    inputs = regen._RegenInputs(
        state.evidence,
        state.gold,
        alias / "heldout.json",
        state.local / "reports/sim2real-report.json",
        {},
        _decision(state.gold),
    )
    snapshot = regen._validate_regen_materialized_frames(
        _config(),
        state.local,
        inputs,
        snapshot_dir=state.local.resolve() / ".verified-regen",
    )
    Image.new("RGB", (2, 2), (201, 202, 203)).save(frame)
    regen_state = regen._RegenState(
        state.evidence, state.gold, inputs.report_path, {}, {}, None, snapshot
    )
    rrd_path = state.local / "reports/sim2real.rrd"
    regen._emit_regen_rrd(_config(), state.local, rrd_path, regen_state, [], "")
    with pytest.raises(regen.Sim2RealRerunRegenError, match="symlinked ancestor"):
        regen._emit_regen_mcap(state.local, regen_state)
    assert (
        regen._emit_regen_mcap(state.local.resolve(), regen_state)["status"]
        == "written"
    )
    _assert_native_pixels(rrd_path, rrd_path.with_suffix(".mcap"))
    assert state.gold == original_gold


def _corrupt_snapshot_provenance(state, frame, payload, tmp_path, mutation):
    if mutation == "alias":
        state.gold["local_renders_dir"] = "eval/other/renders"
    elif mutation == "outer":
        state.gold["outer_iteration"] = 2
    elif mutation == "split":
        state.gold["evaluation_split"] = "validation"
    elif mutation == "checkpoint":
        state.gold["render_manifest"]["policy_checkpoint"]["sha256"] = "b" * 64
    elif mutation == "digest":
        Image.new("RGB", (2, 2), (201, 202, 203)).save(frame)
    else:
        target = tmp_path / "other.png"
        target.write_bytes(payload)
        frame.unlink()
        frame.symlink_to(target)


@pytest.mark.parametrize(
    "mutation", ["alias", "outer", "split", "checkpoint", "digest", "symlink"]
)
@pytest.mark.parametrize("route", ["stage14", "regen"])
def test_snapshot_admission_preserves_provenance_guards(tmp_path, mutation, route):
    state, frame, payload = _state(tmp_path)
    _corrupt_snapshot_provenance(state, frame, payload, tmp_path, mutation)
    messages = {
        "alias": "render path sources disagree",
        "outer": "outer iteration",
        "split": "evaluation split",
        "checkpoint": "checkpoint disagrees",
        "digest": "frame bytes disagree",
        "symlink": "frame bytes.*safe",
    }
    with pytest.raises(
        (RuntimeError, ValueError, viz.Sim2RealVizError), match=messages[mutation]
    ):
        if route == "stage14":
            stage14._validate_stage14_materialized_frames(state)
        else:
            inputs = regen._RegenInputs(
                state.evidence,
                state.gold,
                tmp_path / "heldout.json",
                state.local / "reports/sim2real-report.json",
                {},
                _decision(state.gold),
            )
            regen._validate_regen_materialized_frames(
                _config(),
                state.local,
                inputs,
                snapshot_dir=state.local / ".verified-regen",
            )


@pytest.mark.parametrize("kind", ["outside", "symlink", "wrong-layout"])
def test_snapshot_input_root_cannot_override_canonical_authority(tmp_path, kind):
    state, _, _ = _state(tmp_path)
    snapshot = stage14._validate_stage14_materialized_frames(state)
    if kind == "outside":
        snapshot = tmp_path
    elif kind == "symlink":
        link = state.local / "linked-snapshot"
        link.symlink_to(snapshot, target_is_directory=True)
        snapshot = link
    else:
        state.gold["local_renders_dir"] = "eval/gold-heldout/outer-02/renders"
        state.gold["render_lineage"]["local_relative_dir"] = state.gold[
            "local_renders_dir"
        ]
    with pytest.raises(viz.Sim2RealVizError):
        viz.emit_sim2real_mcap(
            local_dir=state.local,
            inner_evidence=state.evidence,
            heldout_report=state.gold,
            heldout_local_dir=snapshot,
        )
