"""Verify native contact selection, identity association and immutable evidence copies."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import contact_evidence, measure, reference_contacts


def _path(robot, body=""):
    return f"/World/envs/env_{robot}/Robot" + (f"/{body}" if body else "")


@pytest.fixture
def native(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(reference_contacts, "_tensor", lambda value: value)
    sensors = [_path(0, "foot"), _path(1, "base"), _path(0, "base"), _path(1, "foot")]
    forces = torch.zeros((12, 1))
    points = torch.arange(36, dtype=torch.float32).reshape(12, 3)
    normals = torch.zeros((12, 3))
    normals[:, 2] = 1
    separation = -torch.arange(12, dtype=torch.float32).reshape(12, 1) / 100
    counts = torch.zeros((4, 2), dtype=torch.int32)
    starts = counts.clone()
    data = (forces, points, normals, separation, counts, starts)
    view = SimpleNamespace(
        sensor_paths=sensors,
        sensor_count=4,
        filter_count=2,
        filter_paths=[["/Scene/floor", "/Scene/wall"] for _ in sensors],
        get_contact_data=lambda dt: data,
    )
    body_poses = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    root_poses = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    body_view = SimpleNamespace(
        prim_paths=sensors[::-1], get_transforms=lambda: body_poses
    )
    root_view = SimpleNamespace(
        prim_paths=[_path(1), _path(0)], get_root_transforms=lambda: root_poses
    )
    clock = SimpleNamespace(
        tick=100,
        get_num_physics_steps=lambda: clock.tick,
        get_simulation_time=lambda: clock.tick * 0.005,
    )
    env = SimpleNamespace(num_envs=2, physics_dt=0.005)
    measurements = reference_contacts.ContactMeasurements.__new__(
        reference_contacts.ContactMeasurements
    )
    measurements.env = env
    measurements.view = view
    measurements.body_count = 2
    measurements.body_names = ["base", "foot"]
    measurements.base_index = 0
    measurements.filter_paths = ["/Scene/floor", "/Scene/wall"]
    measurements.obstacle = torch.zeros(2)
    measurements.peer = torch.zeros(2)
    measurements.evidence = None
    measurements.surface = SimpleNamespace(
        metadata={"test_surface": True},
        recognize=lambda data, pairs, indices, original, retain: (
            torch.zeros_like(original),
            None,
        ),
    )
    monkeypatch.setattr(
        contact_evidence, "_native_handles", lambda *_: (body_view, root_view, clock)
    )
    evidence = contact_evidence.ContactEvidence(env, measurements, tmp_path, "solo")
    measurements.evidence = evidence
    return SimpleNamespace(**locals())


def _tick(native):
    result = native.measurements._obstacle_forces()
    native.measurements.obstacle = native.torch.maximum(
        native.measurements.obstacle, result
    )
    native.clock.tick += 1
    return result


def test_actual_paths_bind_poses_independently_from_classified_indices(native):
    n = native
    n.counts[1, 1] = 1
    n.starts[1, 1] = 7
    n.forces[7] = 2
    n.normals[7] = n.torch.tensor([1.0, 0, 0])
    expected = [array.clone() for array in n.data]
    with n.evidence.interval():
        _tick(n)
    row = n.evidence.rows[0]
    assert row["classified_robot_index"] == 0 and row["actual_robot_index"] == 1
    assert row["classified_body_name"] == "foot" and row["actual_body_name"] == "base"
    assert row["sensor_order_matches_classification"] is False
    assert (
        row["native_filter_path"] == "/Scene/wall" and row["contact_buffer_index"] == 7
    )
    np.testing.assert_array_equal(row["body_pose_world_xyzw"], n.body_poses[2])
    np.testing.assert_array_equal(row["root_pose_world_xyzw"], n.root_poses[0])
    np.testing.assert_array_equal(row["point_world_m"], n.points[7])
    assert row["separation_m"] == float(n.separation[7, 0])
    assert row["physics_event_count"] == 100 and row["physics_substep"] == 1
    for before, after in zip(expected, n.data):
        assert n.torch.equal(before, after)


def test_small_individual_contacts_and_distinct_aggregate_peak_are_retained(native):
    n = native
    n.counts[0, 0] = 2
    n.starts[0, 0] = 3
    n.forces[3] = -0.015
    n.forces[4] = 0.012
    with n.evidence.interval():
        first = _tick(n)
        n.counts[0, 0] = 3
        n.forces[3:6] = 0.014
        n.body_poses += 100
        n.root_poses += 100
        second = _tick(n)
    with np.load(n.evidence.output / "000001.npz", allow_pickle=False) as saved:
        assert saved["force_n"][0] == pytest.approx(-0.015)
        assert saved["sample_tick_original_classified_sum_n"][0] == pytest.approx(0.027)
        assert saved["sample_tick_effective_classified_sum_n"][0] == pytest.approx(
            0.027
        )
        assert saved["control_interval_peak_effective_sum_n"][0] == pytest.approx(0.042)
        assert saved["sample_tick_original_contributing_contacts"][0] == 2
        assert saved["sample_tick_effective_contributing_contacts"][0] == 2
        assert (
            saved["physics_substeps_observed"] == 2
            and saved["physics_event_count"][0] == 100
        )
        assert (
            saved["classified_by_base"][0]
            and not saved["classified_by_steep_normal"][0]
        )
        assert saved["actual_body_name"][0] == "foot"
        assert saved["body_pose_world_xyzw"][0, 0] < 100
    assert first[0] == pytest.approx(0.027) and second[0] == pytest.approx(0.042)


def test_contacts_outside_probe_interval_are_not_recorded(native):
    native.counts[0, 0] = 1
    native.forces[0] = 100
    _tick(native)
    assert native.evidence.rows == {} and not list(native.evidence.output.glob("*.npz"))


def test_native_step_exception_retains_partial_contacts_before_propagating(native):
    native.counts[0, 0] = 1
    native.forces[0] = 100
    with pytest.raises(RuntimeError, match="native failure"):
        with native.evidence.interval():
            _tick(native)
            raise RuntimeError("native failure")
    with np.load(native.evidence.output / "000001.npz", allow_pickle=False) as saved:
        assert not saved["step_completed"] and saved["force_n"][0] == 100


def test_empty_interval_retains_zero_population_aggregate(native):
    with native.evidence.interval():
        _tick(native)
    with np.load(native.evidence.output / "000001.npz", allow_pickle=False) as saved:
        assert saved["physics_substeps_observed"] == 1
        np.testing.assert_array_equal(
            saved["control_interval_peak_effective_sum_n"], [0, 0]
        )


def test_recorded_probe_exits_recorder_before_physical_gate(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def recorder(env, output, name):
        yield
        (output / "raw-contact.json").write_text("{}")
        events.append("retained")

    adapter = SimpleNamespace(record_probe_contacts=recorder)
    recipe = SimpleNamespace(probe=SimpleNamespace(actions=[[0]], tolerance=0.001))
    monkeypatch.setattr(
        measure,
        "_probe_trace",
        lambda *_: [{"state": {"obstacle_contact": np.array([1])}}],
    )
    monkeypatch.setattr(measure, "save_trace", lambda *_: events.append("trace"))
    trace = measure._recorded_probe(adapter, None, None, recipe, None, tmp_path, "solo")
    with pytest.raises(ValueError, match="free-space baseline"):
        measure._probe_controls(None, None, None, recipe, None, trace, tmp_path)
    assert (tmp_path / "raw-contact.json").exists() and events == ["retained", "trace"]


def test_body_mapping_rejects_ambiguous_native_paths(native):
    native.body_view.prim_paths[0] = native.body_view.prim_paths[1]
    with pytest.raises(ValueError, match="ambiguous"):
        contact_evidence._mapping(
            native.view, ["base", "foot"], native.body_view, native.root_view
        )


def test_recognized_original_contact_is_retained_with_effective_counts_and_geometry(
    native,
):
    n = native
    n.counts[0, 0] = 2
    n.forces[0:2] = n.torch.tensor([[100.0], [10.0]])
    fields = {
        "support_reason": n.torch.tensor([1, 8]),
        "support_radius_m": n.torch.tensor([0.02, 0.02]),
        "source_face_index": n.torch.tensor([15, 16]),
        "source_distance_m": n.torch.tensor([0.01, 0.01]),
        "terrain_witness_world_m": n.torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "support_witness_valid": n.torch.tensor([True, True]),
    }
    n.measurements.surface.recognize = lambda *args, **kwargs: (
        n.torch.tensor([True, False]),
        fields,
    )
    with n.evidence.interval():
        result = _tick(n)
        fields["source_face_index"][:] = 999
        fields["terrain_witness_world_m"][:] = 999
    row = n.evidence.rows[0]
    assert row["force_magnitude_n"] == 100.0 and result[0] == 10.0
    assert row["original_candidate"] and not row["effective_candidate"]
    assert row["source_face_index"] == 15
    assert row["terrain_witness_world_m"].tolist() == [1, 2, 3]
    assert row["support_witness_valid"]
    assert row["sample_tick_original_contributing_contacts"] == 2
    assert row["sample_tick_effective_contributing_contacts"] == 1
    assert row["sample_tick_original_classified_sum_n"] == 110.0
    assert row["sample_tick_effective_classified_sum_n"] == 10.0


def test_v3_nonfoot_geometry_has_false_witness_sentinel(native):
    import json

    index = json.loads((native.evidence.output / "index.json").read_text())
    assert index["schema"] == "npa.navigation.probe-contact-samples.v3"
    row = contact_evidence._surface_row(None, 0)
    assert row["support_reason"] == 2 and row["support_witness_valid"] is False
    assert row["terrain_witness_world_m"].tolist() == [-1, -1, -1]
