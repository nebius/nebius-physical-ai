"""Guard physical robot import, upstream policy inputs, and truthful GPU reports."""

import json
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from npa.workbench.marble.api import MarbleError
from npa.workbench.marble.quadruped_assets import _description, acquire_robot
from npa.workbench.marble.quadruped_control import DEFAULT_ANGLES, observe
from npa.workbench.marble.quadruped_report import _data
from npa.workbench.marble.schemas import QuadrupedRequest


def test_shadow_mesh_matches_physics_after_provider_transform(monkeypatch, tmp_path):
    import trimesh
    from npa.workbench.marble import quadruped_render
    from npa.workbench.marble.rover_physics import _warehouse

    mesh = trimesh.creation.box(extents=[2, 3, 5])
    mesh.export(tmp_path / "collider.glb")
    transform = {"scale": [2, 0.5, 3], "translation": [7, -4, 9]}
    world = {
        "source_kind": "world-api",
        "mesh_transform": transform,
        "splat_transform": transform,
    }
    physics = Mock(GEOM_MESH=5, GEOM_FORCE_CONCAVE_TRIMESH=1)
    _warehouse(physics, tmp_path, world)
    collider = physics.createCollisionShape.call_args.kwargs
    monkeypatch.setattr(
        quadruped_render,
        "_render_actor",
        lambda root: {
            "device_type": "CUDA",
            "cpu_render_fallback": False,
            "frame_wall_seconds": [1],
        },
    )
    quadruped_render._actor(
        tmp_path, world, SimpleNamespace(width=32, height=32, samples=1, frames=1)
    )
    shadow = json.loads((tmp_path / "render-warehouse.json").read_text())
    np.testing.assert_allclose(shadow["vertices"], collider["vertices"])
    np.testing.assert_array_equal(
        np.asarray(shadow["faces"]).reshape(-1), collider["indices"]
    )


def test_robot_import_preserves_dynamic_trunk_and_massless_optical_frames(tmp_path):
    source = tmp_path / "robot-assets"
    (source / "urdf").mkdir(parents=True)
    (source / "urdf/go1.urdf").write_text("""<robot name="go1">
      <link name="base"/><joint name="floating_base" type="fixed"><parent link="base"/><child link="trunk"/></joint>
      <link name="trunk"><inertial><mass value="5.204"/></inertial>
      <visual><geometry><mesh filename="package://go1_description/meshes/trunk.dae"/></geometry></visual></link>
      <link name="camera_optical_face"/>
      <joint name="optical" type="fixed"><parent link="trunk"/><child link="camera_optical_face"/></joint></robot>""")
    path = _description(source, tmp_path / "go1.urdf")
    tree = ET.parse(path)
    assert tree.find("link[@name='base']") is None
    assert tree.find("joint[@name='floating_base']") is None
    assert tree.find("link[@name='trunk']/inertial/mass").get("value") == "5.204"
    assert (
        tree.find("link[@name='camera_optical_face']/inertial/mass").get("value") == "0"
    )
    assert tree.find(".//mesh").get("filename") == "robot-assets/meshes/trunk.dae"


def test_upstream_policy_observation_protocol_uses_measured_local_state():
    class Physics:
        def getBasePositionAndOrientation(self, robot):
            return [0, 0, 0.3], [0, 0, 0, 1]

        def getBaseVelocity(self, robot):
            return [1, 2, 3], [4, 5, 6]

        def getMatrixFromQuaternion(self, quaternion):
            return [0, -1, 0, 1, 0, 0, 0, 0, 1]

        def getJointStates(self, robot, joints):
            return [
                (float(angle) + 0.2, index, (), 0)
                for index, angle in enumerate(DEFAULT_ANGLES)
            ]

    observation, _, _ = observe(
        Physics(), 1, list(range(12)), np.arange(12) / 10, [0.35, 0, 0.1]
    )
    np.testing.assert_allclose(observation[:9], [2, -1, 3, 5, -4, 6, 0, 0, -1])
    np.testing.assert_allclose(observation[9:21], 0.2)
    np.testing.assert_allclose(observation[21:33], np.arange(12))
    np.testing.assert_allclose(observation[33:45], np.arange(12) / 10)
    np.testing.assert_allclose(observation[45:], [0.35, 0, 0.1])
    assert observation.shape == (48,) and observation.dtype == np.float32


def test_unverified_policy_bytes_are_rejected_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "npa.workbench.marble.quadruped_assets._download",
        lambda url: b"changed-upstream",
    )
    with pytest.raises(MarbleError, match="hash mismatch"):
        acquire_robot(tmp_path)


def test_cpu_actor_render_cannot_be_presented_as_gpu_evidence(tmp_path):
    result = {
        "metrics": {
            "physics": {},
            "observer": {
                "actor": {
                    "frame_wall_seconds": [1],
                    "device_type": "CPU",
                    "cpu_render_fallback": True,
                    "devices": [],
                }
            },
        }
    }
    with pytest.raises(MarbleError, match="CUDA"):
        _data(tmp_path, result)


def test_sensor_clock_requires_exact_physics_alignment():
    settings = {
        "input_path": "s3://example-bucket/world",
        "output_path": "s3://example-bucket/capture",
        "run_id": "go1-test",
    }
    assert QuadrupedRequest(**settings).sensor_hz == 25
    with pytest.raises(ValueError, match="250 Hz"):
        QuadrupedRequest(**settings, sensor_hz=24)


def test_blender_script_failure_surfaces_diagnostic_even_with_zero_exit(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace
    from npa.workbench.marble import quadruped_render

    monkeypatch.setattr(quadruped_render, "_blender", lambda root: root / "blender")

    def failed_render(command, stdout, stderr):
        assert command[command.index("--python-exit-code") + 1] == "1"
        stdout.write("RuntimeError: missing renderer device")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(quadruped_render.subprocess, "run", failed_render)
    with pytest.raises(MarbleError, match="missing renderer device"):
        quadruped_render._render_part(tmp_path / "blender", tmp_path, 0, 1)


def test_renderer_batches_reject_gaps_and_cpu_fallback():
    from npa.workbench.marble.quadruped_render import _merge_parts

    first = {
        "frame_start": 0,
        "frame_stop": 2,
        "frame_wall_seconds": [1.2, 1.1],
        "device_type": "CUDA",
        "cpu_render_fallback": False,
        "devices": ["test-device"],
    }
    second = dict(first, frame_start=2, frame_stop=4)
    assert _merge_parts([first, second], 4)["frame_wall_seconds"] == [1.2, 1.1] * 2
    with pytest.raises(MarbleError, match="incomplete"):
        _merge_parts([first, dict(second, frame_start=3)], 4)
    with pytest.raises(MarbleError, match="incomplete"):
        _merge_parts([first, dict(second, cpu_render_fallback=True)], 4)


def test_patrol_brakes_turns_and_resumes_from_measured_heading():
    from npa.workbench.marble.quadruped_patrol import PatrolController

    class Physics:
        position = [0, 0, 0.35]
        yaw = 0.0

        def getBasePositionAndOrientation(self, robot):
            return self.position, [0, 0, 0, 1]

        def getEulerFromQuaternion(self, orientation):
            return [0, 0, self.yaw]

    physics = Physics()
    patrol = PatrolController(
        {"start": [0, 0, 0.35], "heading": 0, "distance": 8}, 1.2, "turnaround"
    )
    assert patrol.command(physics, 1, False)[0] == 0
    speeds = [patrol.command(physics, 1, True)[0] for _ in range(100)]
    assert max(np.diff(speeds)) <= 0.016001 and speeds[-1] == pytest.approx(1.2)
    physics.position = [6.8, 0, 0.35]
    for _ in range(60):
        command = patrol.command(physics, 1, True)
    assert patrol.phase == "turning" and command[0] == 0
    assert abs(command[2]) == pytest.approx(0.9)
    physics.yaw = np.pi - 0.18
    command = patrol.command(physics, 1, True)
    assert patrol.phase == "inbound" and patrol.turns == 1 and command[0] > 0


def test_turnaround_report_rejects_rotation_without_return_travel():
    from types import SimpleNamespace
    from npa.workbench.marble.quadruped_physics import _motion_metrics

    rows = [
        {
            "linear_velocity_bullet": [0, 0, 0],
            "heading_radians": yaw,
            "position_bullet": [0, 0, 0.35],
            "motion_phase": "turning",
        }
        for yaw in np.linspace(0, np.pi, 100)
    ]
    request = SimpleNamespace(motion_profile="turnaround", speed_mps=1.2)
    with pytest.raises(MarbleError, match="two-meter return"):
        _motion_metrics(rows, request)


def test_return_distance_does_not_bridge_separate_patrol_legs():
    from types import SimpleNamespace
    from npa.workbench.marble.quadruped_physics import _motion_metrics

    phases = ["inbound", "inbound", "outbound", "inbound", "inbound"]
    rows = [
        {
            "linear_velocity_bullet": [1, 0, 0],
            "heading_radians": np.pi * index / 4,
            "position_bullet": [position, 0, 0.35],
            "motion_phase": phase,
        }
        for index, (position, phase) in enumerate(zip([0, 1, 5, 10, 11], phases))
    ]
    metrics = _motion_metrics(
        rows, SimpleNamespace(motion_profile="turnaround", speed_mps=1.2)
    )
    assert metrics["return_distance_m"] == pytest.approx(2.0)
