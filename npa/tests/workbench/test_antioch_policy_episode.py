"""Regression coverage for model visibility, physical verdicts and input evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import zipfile
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[3] / "npa/examples/antioch-openpi-live/src"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    antioch = types.ModuleType("antioch")
    antioch.Logger = lambda _name: SimpleNamespace()
    antioch.param = lambda default, **_kwargs: default
    antioch.scenario = lambda **_kwargs: lambda body: body
    monkeypatch.setitem(sys.modules, "antioch", antioch)
    loaded = []
    for name in ("scenario_v2", "policy_episode"):
        spec = importlib.util.spec_from_file_location(name, SOURCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def _rgb(*, target_size=16, shifted=False):
    rows, columns = np.indices((224, 224))
    image = np.repeat(((rows + columns * 2) % 180 + 30)[..., None], 3, axis=2).astype(
        np.uint8
    )
    image[100 : 100 + target_size, 100 : 100 + target_size] = [220, 12, 8]
    return np.roll(image, 31, axis=1) if shifted else image


def test_washed_out_but_nonblack_policy_frame_is_rejected(modules):
    scenario, _episode = modules
    image = np.full((224, 224, 3), 247, dtype=np.uint8)
    image[:, :10] = 220
    image[111:115, 113:118] = [215, 30, 20]
    frame = scenario._camera_frame_from_buffer(image, view="exterior")
    assert frame.luminance_variance > 25  # This satisfied the former gate.
    assert frame.reason == "overexposed"
    assert frame.near_white_fraction > 0.90
    np.testing.assert_array_equal(frame.rgb, image)


def test_tiny_target_or_wrong_initial_wrist_aim_blocks_inference(modules):
    scenario, _episode = modules
    good = scenario._camera_frame_from_buffer(_rgb(), view="exterior")
    tiny = scenario._camera_frame_from_buffer(
        _rgb(target_size=4, shifted=True), view="wrist"
    )
    pair = scenario._validate_camera_pair(
        good,
        tiny,
        render_sequence=2,
        last_accepted_render_sequence=1,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
        initial_alignment=True,
    )
    assert pair.reason == "target_unresolved"
    assert pair.rejected_view == "wrist"
    missing = scenario._camera_frame_from_buffer(_rgb(target_size=0), view="exterior")
    pair = scenario._validate_camera_pair(
        missing,
        tiny,
        render_sequence=3,
        last_accepted_render_sequence=2,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
    )
    assert not pair.accepted


def test_runtime_occlusion_requires_other_view_or_measured_contact(modules):
    scenario, _episode = modules
    missing = scenario._camera_frame_from_buffer(_rgb(target_size=0), view="exterior")
    wrist = scenario._camera_frame_from_buffer(_rgb(shifted=True), view="wrist")
    kwargs = dict(
        render_sequence=2,
        last_accepted_render_sequence=1,
        exterior_cube_in_frame=True,
        wrist_cube_in_frame=True,
    )
    assert scenario._validate_camera_pair(missing, wrist, **kwargs).accepted
    wrist = scenario._camera_frame_from_buffer(
        _rgb(target_size=0, shifted=True), view="wrist"
    )
    assert not scenario._validate_camera_pair(missing, wrist, **kwargs).accepted
    assert scenario._validate_camera_pair(
        missing, wrist, gripper_contact=True, **kwargs
    ).accepted


def test_scene_optics_cover_cube_and_approach_region(modules):
    scenario, _episode = modules
    pose = (scenario.EXTERIOR_CAMERA_EYE, scenario.EXTERIOR_CAMERA_TARGET, (0, 0, 1))
    optics = scenario._camera_optical_config("exterior")
    for point in (scenario.CUBE_INITIAL_POSITION, (0.36, 0.0, 0.49), (0.48, 0, 0.15)):
        assert scenario._point_in_camera_frame(point, pose, optics)
    # At the cube depth the new horizontal projection is well above four pixels.
    distance = np.linalg.norm(
        np.subtract(scenario.EXTERIOR_CAMERA_EYE, scenario.CUBE_INITIAL_POSITION)
    )
    projected = (
        224 * optics["focal_length"] / optics["horizontal_aperture"] * 0.07 / distance
    )
    assert projected > 12


def test_fixed_renderer_exposure_is_applied_to_actual_settings(modules):
    scenario, _episode = modules
    settings = {}
    values = scenario._configure_policy_rendering(
        SimpleNamespace(set=settings.__setitem__)
    )
    assert settings == values
    assert settings["/rtx/post/histogram/enabled"] is False
    assert settings["/rtx/post/tonemap/filmIso"] == 100
    assert settings["/rtx/post/dof/enabled"] is False


def test_pickup_requires_simulation_time_and_continuous_contact(modules):
    _scenario, episode = modules
    progress = episode._PickupProgress()
    progress.observe(sim_seconds=0, distance=0.4, lift=0, contact_force=0, closed=False)
    sample = dict(distance=0.03, lift=0.06, contact_force=1, closed=True)
    progress.observe(sim_seconds=1, **sample)
    for _ in range(100):
        progress.observe(sim_seconds=1, **sample)
    assert not progress.success
    progress.observe(sim_seconds=1.9, **sample)
    assert not progress.success
    progress.observe(sim_seconds=2, **(sample | {"contact_force": 0}))
    progress.observe(sim_seconds=2.1, **sample)
    assert progress.hold_seconds == 0
    progress.observe(sim_seconds=3.1, **sample)
    assert progress.success
    progress.observe(sim_seconds=3.2, **(sample | {"lift": 0}))
    assert not progress.success


def test_contact_requires_both_fingers_and_large_cube_can_close(modules):
    scenario, _episode = modules
    view = SimpleNamespace(
        get_contact_force_matrix=lambda **_kwargs: np.array([[[1, 0, 0], [0, 0, 0]]])
    )
    assert scenario._contact_force_magnitude(view, 1 / 60) == 0
    view.get_contact_force_matrix = lambda **_kwargs: np.array(
        [[[1, 0, 0], [-2, 0, 0]]]
    )
    assert scenario._contact_force_magnitude(view, 1 / 60) == 1
    state = np.array([*scenario.DROID_RESET_JOINTS, 0.035, 0.035])
    assert scenario._droid_gripper_observation(state) >= 0.05
    assert scenario._droid_gripper_observation(state) < 0.5


def test_second_reply_does_not_end_unexecuted_chunk_or_pass_pickup(modules):
    _scenario, episode = modules
    state = dict(
        round_trips=2,
        completed_chunks=1,
        pickup=episode._PickupProgress(),
        applied=5,
        control_steps=15,
        sim_seconds=10,
        last_apply_sim_seconds=9,
    )
    assert episode._termination_reason(objective="communication", **state) == ""
    assert episode._termination_reason(objective="pickup", **state) == ""
    state.update(completed_chunks=2, applied=10)
    assert (
        episode._termination_reason(objective="communication", **state)
        == "communication_complete"
    )
    assert episode._termination_reason(objective="pickup", **state) == ""
    state.update(applied=15, last_apply_sim_seconds=10)
    assert episode._termination_reason(objective="pickup", **state) == ""
    state["sim_seconds"] = 10.1
    assert (
        episode._termination_reason(objective="pickup", **state)
        == "control_steps_exhausted"
    )


def test_archive_retains_exact_request_raw_actions_and_applied_targets(
    modules, tmp_path
):
    import openpi_protocol

    scenario, episode = modules
    evidence = episode._PolicyEvidence(tmp_path)
    joints = np.array([*scenario.DROID_RESET_JOINTS, 0.04, 0.04], dtype=np.float32)
    observation = scenario._build_policy_observation(
        _rgb(), _rgb(shifted=True), joints, "pick up the red cube"
    )
    request = scenario.PolicyRequest(observation, 1, 20)
    payload = evidence.request(
        request,
        sim_seconds=1,
        producer_markers={"exterior": (20, 60), "wrist": (20, 60)},
    )
    raw = np.tile(joints[:8], (15, 1))
    raw[0, 0] = 10.0  # The unprojected value must survive for diagnosis.
    evidence.response(1, {"actions": raw}, sim_seconds=2)
    projected, _ = scenario._validated_actions({"actions": raw}, joints)
    evidence.target(
        camera_pair_id=1, row=0, target=projected[0], before=joints, sim_seconds=2
    )
    artifacts = []
    digest = evidence.finish(
        SimpleNamespace(add_artifact=lambda path, **_kwargs: artifacts.append(path))
    )
    assert digest == hashlib.sha256(artifacts[0].read_bytes()).hexdigest()
    with zipfile.ZipFile(artifacts[0]) as archive:
        assert archive.read("request-000001.msgpack") == payload
        decoded = openpi_protocol.unpackb(payload)
        np.testing.assert_array_equal(
            decoded["observation/wrist_image_left"],
            observation["observation/wrist_image_left"],
        )
        with archive.open("response-000001.npy") as saved:
            np.testing.assert_array_equal(np.load(saved, allow_pickle=False), raw)
        manifest = json.loads(archive.read("manifest.json"))
        for name, sha256 in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha256
        events = [
            json.loads(line) for line in archive.read("control.jsonl").splitlines()
        ]
        assert events[-1]["target"][0] != float(raw[0, 0])
        assert events[0]["payload_sha256"] == hashlib.sha256(payload).hexdigest()


class _Body:
    def __init__(self, **values):
        self.position = np.asarray(values.get("position", [0.36, 0, 0.49]))
        self.joints = np.zeros(9)
        self.end_effector = self
        self.actions = []

    def get_world_pose(self):
        return self.position.copy(), [1, 0, 0, 0]

    def set_joint_positions(self, value):
        self.joints = value.copy()

    def get_joint_positions(self):
        return self.joints.copy()

    def apply_action(self, action):
        self.actions.append(action.joint_positions.copy())
        self.joints = action.joint_positions.copy()

    def get_contact_force_matrix(self, **_kwargs):
        return np.zeros((1, 2, 3))


class _World:
    def __init__(self):
        self.current_time = 0.0
        self.stage = object()
        self.scene = SimpleNamespace(
            add=lambda body: body, add_ground_plane=lambda **_kwargs: None
        )

    def reset(self):
        self.current_time = 0

    def step(self, **_kwargs):
        self.current_time += 1 / 60

    def get_physics_dt(self):
        return 1 / 60


class _GraspingBody(_Body):
    clock = None

    def __init__(self, **values):
        super().__init__(**values)
        self.name = values.get("name")

    def get_world_pose(self):
        if self.clock.current_time >= 2:
            if self.name == "cube":
                return np.array([0.48, 0, 0.095]), [1, 0, 0, 0]
            if self.name == "franka":
                return np.array([0.48, 0, 0.15]), [1, 0, 0, 0]
        return super().get_world_pose()

    def apply_action(self, action):
        super().apply_action(action)
        # Model readback at the cube width, despite the commanded zero gap.
        if self.joints[7] == 0:
            self.joints[7:] = 0.035

    def get_contact_force_matrix(self, **_kwargs):
        if self.clock.current_time >= 2:
            return np.array([[[1, 0, 0], [-1, 0, 0]]])
        return np.zeros((1, 2, 3))


class _Run:
    def __init__(self):
        self.results = {}
        self.checks = {}
        self.artifacts = []

    def add_result(self, name, value):
        self.results[name] = value

    def check(self, name, passed, **_kwargs):
        self.checks[name] = bool(passed)

    def add_artifact(self, path, **_kwargs):
        self.artifacts.append(path)


def _install_fake_isaac(monkeypatch):
    surfaces = {
        "carb": {
            "settings": SimpleNamespace(
                get_settings=lambda: SimpleNamespace(set=lambda *_args: None)
            )
        },
        "isaacsim.core.experimental.utils.app": {},
        "isaacsim.core.api.objects": {"DynamicCuboid": _Body, "FixedCuboid": _Body},
        "isaacsim.core.prims": {"RigidPrim": _Body},
        "isaacsim.core.utils.types": {"ArticulationAction": SimpleNamespace},
        "isaacsim.core.utils.viewports": {"set_camera_view": lambda **_kwargs: None},
        "isaacsim.core.utils.extensions": {"enable_extension": lambda *_args: None},
        "isaacsim.robot.manipulators.examples.franka": {"Franka": _Body},
        "isaacsim.sensors.experimental.rtx": {
            "CameraSensor": object,
            "RtxCamera": object,
        },
    }
    for name, values in surfaces.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            parent = ".".join(parts[:end])
            if parent not in sys.modules:
                module = types.ModuleType(parent)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, parent, module)
        for key, value in values.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)


def _install_fake_camera_scene(scenario, monkeypatch, world):
    camera = SimpleNamespace(close=lambda: None)
    telemetry = SimpleNamespace(
        publish_camera_pair=lambda *_args: (),
        publish_display=lambda *_args: None,
        snapshot=lambda: {"logger": {"dropped": 0}},
        close=lambda: {"exterior": True, "wrist": True, "display": True},
    )
    for name in (
        "_configure_lighting",
        "_look_at",
        "_configure_camera_optics",
        "_start_camera_timeline",
    ):
        monkeypatch.setattr(scenario, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scenario, "_stock_franka_camera_points", lambda *_args: (0, 0, 0)
    )
    monkeypatch.setattr(scenario, "_world_transform", lambda *_args: None)
    monkeypatch.setattr(
        scenario, "_calibrate_wrist_camera_mount", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(scenario, "_aim_wrist_camera", lambda *_args: (0, 0, 1))
    monkeypatch.setattr(
        scenario, "_point_in_camera_frame", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        scenario, "_build_rtx_rgb_camera", lambda *_args, **_kwargs: camera
    )
    monkeypatch.setattr(
        scenario, "_initialize_live_capture", lambda *_args: (telemetry, {}, {})
    )
    monkeypatch.setattr(
        scenario,
        "_install_overlay",
        lambda: [SimpleNamespace(text="") for _ in range(5)],
    )
    monkeypatch.setattr(scenario, "_franka_link_points", lambda *_args: [])
    frames = {
        view: scenario._camera_frame_from_buffer(
            _rgb(shifted=view == "wrist"), view=view
        )
        for view in ("exterior", "wrist")
    }
    occluded = {
        view: scenario._camera_frame_from_buffer(
            _rgb(target_size=0, shifted=view == "wrist"), view=view
        )
        for view in ("exterior", "wrist")
    }

    def capture(*_args):
        active = (
            occluded
            if getattr(world, "grasp", False) and world.current_time >= 2
            else frames
        )
        return {
            view: scenario.CameraSample(frame, (round(world.current_time * 60), 60))
            for view, frame in active.items()
        }

    monkeypatch.setattr(scenario, "_capture_camera_samples", capture)


class _ImmediateExecutor:
    def __init__(self, **_kwargs):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    def shutdown(self, **_kwargs):
        pass


@pytest.mark.parametrize(
    "objective,steps,replies,grasp",
    [
        ("communication", 10, 2, False),
        ("pickup", 15, 3, False),
        ("pickup", 450, None, True),
    ],
)
def test_executing_loop_runs_second_chunk_and_requires_task_evidence(
    modules,
    monkeypatch,
    tmp_path,
    objective,
    steps,
    replies,
    grasp,
):
    rr = pytest.importorskip("rerun")
    scenario, episode = modules
    _install_fake_isaac(monkeypatch)
    world = _World()
    world.grasp = grasp
    if grasp:
        monkeypatch.setattr(_GraspingBody, "clock", world)
        monkeypatch.setattr(
            sys.modules["isaacsim.core.api.objects"], "DynamicCuboid", _GraspingBody
        )
        monkeypatch.setattr(
            sys.modules["isaacsim.core.prims"], "RigidPrim", _GraspingBody
        )
        monkeypatch.setattr(
            sys.modules["isaacsim.robot.manipulators.examples.franka"],
            "Franka",
            _GraspingBody,
        )
    monkeypatch.setattr(sys.modules["antioch"], "world", lambda: world, raising=False)
    _install_fake_camera_scene(scenario, monkeypatch, world)
    monkeypatch.setattr(rr, "send_blueprint", lambda *_args: None)
    client = SimpleNamespace(reconnects=0, shutdown=lambda: None)
    actions = np.tile([*scenario.DROID_RESET_JOINTS, float(grasp)], (15, 1))
    client.infer = lambda request: (
        {"actions": actions},
        0.01,
        request.camera_pair_id,
        request.render_sequence,
    )
    monkeypatch.setattr(scenario, "SafePolicyClient", lambda: client)
    monkeypatch.setattr(scenario, "ThreadPoolExecutor", _ImmediateExecutor)
    recorder = episode._PolicyEvidence(tmp_path)
    monkeypatch.setattr(episode, "_PolicyEvidence", lambda: recorder)
    run = _Run()
    scenario._run_openpi_episode(
        run, "pick up the red cube", objective=objective, control_steps=steps
    )
    if grasp:
        assert 10 <= run.results["safe_targets_applied"] < steps
        assert run.results["termination_reason"] == "pickup_complete"
        assert run.checks["cube_lift_held"]
        assert run.results["pickup_hold_seconds"] >= 1
    else:
        assert run.results["safe_targets_applied"] == steps
        assert run.results["policy_round_trips"] == replies
    assert run.checks["policy_evidence_complete"]
    assert run.checks["episode_completed"] == (objective == "communication" or grasp)
    if objective == "pickup" and not grasp:
        assert run.results["termination_reason"] == "control_steps_exhausted"
        assert not run.checks["cube_lift_held"]
    events = [
        json.loads(line)
        for line in (tmp_path / "control.jsonl").read_text().splitlines()
    ]
    applied = [event for event in events if event["kind"] == "applied"]
    assert {event["row"] for event in applied if event["camera_pair_id"] == 2} == set(
        range(5)
    )
    if not grasp:
        assert events[-1]["sim_seconds"] - applied[-1]["sim_seconds"] >= 1 / 15 - 1e-9


@pytest.mark.parametrize(
    "camera_ready,last_control_at,now,expected",
    [
        (False, None, 90, "camera_unavailable"),
        (True, 0, 89, ""),
        (True, 0, 90, "control_stalled"),
        (True, 85, 90, ""),
        (True, None, 90, ""),
    ],
)
def test_readiness_deadline_does_not_require_an_advancing_simulation_clock(
    modules,
    camera_ready,
    last_control_at,
    now,
    expected,
):
    _scenario, episode = modules
    assert (
        episode._readiness_failure(
            now=now,
            camera_ready=camera_ready,
            camera_unavailable_since=0,
            last_control_at=last_control_at,
            deadline=90,
        )
        == expected
    )
