"""Check first-interval tracing preserves native calls and retains partial failures."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import reference, reference_contacts, reset_evidence

torch = pytest.importorskip("torch")


class Actuator:
    def __init__(self):
        self.num_joints = 2
        self.sea_input = torch.zeros(4, 1, 2)
        self.sea_hidden_state = torch.zeros(1, 4, 3)
        self.sea_cell_state = torch.zeros(1, 4, 3)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(1, 2, 2, 3)
        self.sea_cell_state_per_env = self.sea_cell_state.view(1, 2, 2, 3)
        self.computed_effort = self.applied_effort = torch.zeros(2, 2)
        self.calls = []

    def network(self, value, memory):
        self.calls.append((value, memory))
        return value[:, :, 0], (memory[0] + 1, memory[1] + 2)

    def compute(self, control_action, *, joint_pos, joint_vel):
        self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
        self.sea_input[:, 0, 1] = joint_vel.flatten()
        effort, state = self.network(
            self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
        )
        self.sea_hidden_state[:] = state[0]
        self.sea_cell_state[:] = state[1]
        self.computed_effort = effort.reshape(2, 2)
        self.applied_effort = self.computed_effort.clamp(-1, 1)
        control_action.joint_efforts = self.applied_effort
        return control_action


class LowObservations:
    def __init__(self):
        self.value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.results = []

    def compute_group(self, name, *, update_history=False):
        assert name == "ll_policy" and update_history is False
        self.results.append(self.value)
        return self.value


class Action:
    def __init__(self, robot):
        self._low_level_obs_manager = LowObservations()
        self.low_level_actions = torch.zeros(2, 2)
        self.raw_actions = torch.zeros(2, 3)
        self._low_level_action_term = SimpleNamespace(
            processed_actions=torch.zeros(2, 2)
        )
        self.robot = robot
        self._counter = 0

    def apply_actions(self):
        if self._counter % 2 == 0:
            value = self._low_level_obs_manager.compute_group(
                "ll_policy", update_history=False
            )
            self.low_level_actions[:] = value[:, :2]
        self._low_level_action_term.processed_actions[:] = self.low_level_actions + 1
        self.robot.data.joint_pos_target.torch[:] = (
            self._low_level_action_term.processed_actions
        )
        self._counter += 1
        return self.low_level_actions


class Scene:
    def __init__(self, robot):
        self.robot = robot
        self.updates = []

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot

    def update(self, dt):
        self.updates.append(dt)
        return self.robot


class Environment:
    def step(self, batch, *, marker=None):
        self.action.raw_actions[:] = batch
        for _ in range(self.cfg.decimation):
            self.action.apply_actions()
            control = SimpleNamespace(
                joint_positions=self.robot.data.joint_pos_target.torch,
                joint_velocities=torch.zeros(2, 2),
                joint_efforts=torch.zeros(2, 2),
            )
            returned = self.actuator.compute(
                control, joint_pos=self.joint_position, joint_vel=self.joint_velocity
            )
            assert returned is control
            self.joint_velocity[:] = returned.joint_efforts
            self.joint_position += self.joint_velocity * 0.005
            self.clock.tick += 1
            self.scene.update(0.005)
        return marker


@pytest.fixture
def native(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    monkeypatch.setattr(reference_contacts, "_tensor", lambda value: value)
    monkeypatch.setattr(reset_evidence, "_runtime", lambda _: {"fixture": True})
    env = Environment()
    env.num_envs, env.cfg = 2, SimpleNamespace(decimation=4)
    env.joint_position, env.joint_velocity = torch.zeros(2, 2), torch.zeros(2, 2)
    env.clock = SimpleNamespace(tick=0)
    env.clock.get_num_physics_steps = lambda: env.clock.tick
    env.clock.get_simulation_time = lambda: env.clock.tick * 0.005
    stage = Usd.Stage.CreateInMemory()
    for index in range(2):
        UsdPhysics.ArticulationRootAPI.Apply(
            stage.DefinePrim(f"/World/envs/env_{index}/Robot", "Xform")
        )
    env.sim = SimpleNamespace(stage=stage)
    view = SimpleNamespace(
        prim_paths=[f"/World/envs/env_{index}/Robot" for index in range(2)],
        get_root_transforms=lambda: torch.zeros(2, 7),
        get_root_velocities=lambda: torch.zeros(2, 6),
        get_dof_positions=lambda: env.joint_position,
        get_dof_velocities=lambda: env.joint_velocity,
    )
    env.actuator = Actuator()
    env.robot = SimpleNamespace(
        root_view=view,
        actuators={"legs": env.actuator},
        data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=torch.zeros(2, 2))),
    )
    env.action = Action(env.robot)
    env.action_manager = SimpleNamespace(get_term=lambda _: env.action)
    env.scene = Scene(env.robot)
    env.npa_contacts = SimpleNamespace(evidence=SimpleNamespace(clock=env.clock))
    return env


def _read(tmp_path, name="solo"):
    root = tmp_path / f"probe-{name}-reset"
    record = json.loads((root / "index.json").read_text())
    assert (
        record["arrays_sha256"]
        == hashlib.sha256((root / "first-interval.npz").read_bytes()).hexdigest()
    )
    with np.load(root / "first-interval.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return record, arrays


def test_original_calls_return_identity_order_and_complete_scope(native, tmp_path):
    marker = object()
    recorder = reset_evidence.ResetEvidence(native, tmp_path, "solo")
    with recorder.record():
        assert native.step(torch.zeros(2, 3), marker=marker) is marker
        assert len(native.actuator.calls) == 4
        assert len(native.action._low_level_obs_manager.results) == 2
        assert all(
            value is native.action._low_level_obs_manager.value
            for value in native.action._low_level_obs_manager.results
        )
    record, arrays = _read(tmp_path)
    assert record["complete"] is True and record["observed_substeps"] == 4
    first = [event["phase"] for event in record["events"] if event["substep"] == 1]
    assert first == [
        "control_begin",
        "low_level_observation",
        "control_applied",
        "actuator_before",
        "network_before",
        "network_after",
        "actuator_after",
        "post_physics",
    ]
    post = [event for event in record["events"] if event["phase"] == "post_physics"]
    assert [event["native_step"] for event in post] == [1, 2, 3, 4]
    assert all(
        arrays[event["arrays"]["joint_position"]].shape == (2,) for event in post
    )


def test_copies_survive_source_mutation_and_only_first_interval(native, tmp_path):
    with reset_evidence.ResetEvidence(native, tmp_path, "solo").record():
        native.step(torch.zeros(2, 3))
        before = (tmp_path / "probe-solo-reset" / "first-interval.npz").read_bytes()
        native.step(torch.zeros(2, 3))
        assert (
            tmp_path / "probe-solo-reset" / "first-interval.npz"
        ).read_bytes() == before
    native.actuator.sea_hidden_state.fill_(999)
    native.action._low_level_obs_manager.value.fill_(999)
    record, arrays = _read(tmp_path)
    network = [
        event for event in record["events"] if event["phase"] == "network_before"
    ]
    assert [float(arrays[event["arrays"]["hidden"]].max()) for event in network] == [
        0,
        1,
        2,
        3,
    ]
    obs = next(
        event for event in record["events"] if event["phase"] == "low_level_observation"
    )
    assert arrays[obs["arrays"]["value"]].tolist() == [0, 1, 2, 3]


def test_interval_begin_precedes_original_without_extra_steps_or_observations(
    native, tmp_path
):
    recorder = reset_evidence.ResetEvidence(native, tmp_path, "solo")
    original = native.step
    calls = []

    def step(*args, **kwargs):
        calls.append((args, kwargs))
        assert recorder.events[0]["phase"] == "interval_begin"
        assert recorder.events[0]["substep"] == recorder.events[0]["native_step"] == 0
        assert not native.action._low_level_obs_manager.results
        return original(*args, **kwargs)

    native.step = step
    with recorder.record():
        native.step(torch.zeros(2, 3))
    assert len(calls) == 1 and native.clock.tick == 4
    assert len(native.action._low_level_obs_manager.results) == 2
    record, arrays = _read(tmp_path)
    begin = record["events"][0]
    assert begin["native_time_s"] == 0.0
    assert np.array_equal(arrays[begin["arrays"]["joint_position"]], np.zeros(2))
    assert (
        len([event for event in record["events"] if event["phase"] == "interval_begin"])
        == 1
    )


def test_runtime_failure_leaves_inactive_and_does_not_step(
    native, tmp_path, monkeypatch
):
    failure = RuntimeError("fixture fingerprint failure")

    def runtime(_):
        raise failure

    monkeypatch.setattr(reset_evidence, "_runtime", runtime)
    recorder = reset_evidence.ResetEvidence(native, tmp_path, "solo")
    with pytest.raises(RuntimeError) as caught:
        with recorder.record():
            native.step(torch.zeros(2, 3))
    assert caught.value is failure and recorder.active is False
    assert native.clock.tick == 0 and not native.action._low_level_obs_manager.results
    assert "step" not in vars(native)
    assert not list((tmp_path / "probe-solo-reset").iterdir())


def test_failure_preserves_original_exception_partial_events_and_attributes(
    native, tmp_path
):
    failure = RuntimeError("native fixture network failure")

    def network(*args):
        raise failure

    native.actuator.network = network
    owned = {
        id(obj): set(vars(obj))
        for obj in (
            native,
            native.action,
            native.scene,
            native.actuator,
            native.action._low_level_obs_manager,
        )
    }
    with pytest.raises(RuntimeError) as caught:
        with reset_evidence.ResetEvidence(native, tmp_path, "repeat").record():
            native.step(torch.zeros(2, 3))
    assert caught.value is failure and native.actuator.network is network
    for obj in (
        native,
        native.action,
        native.scene,
        native.actuator,
        native.action._low_level_obs_manager,
    ):
        assert set(vars(obj)) == owned[id(obj)]
    record, _ = _read(tmp_path, "repeat")
    assert record["complete"] is False
    assert record["events"][-1]["phase"] == "network_before"


def test_no_interval_does_not_fabricate_native_evidence(native, tmp_path):
    with pytest.raises(ValueError, match="reset fixture"):
        with reset_evidence.ResetEvidence(native, tmp_path, "solo").record():
            raise ValueError("reset fixture")
    assert not list((tmp_path / "probe-solo-reset").iterdir())
    assert "step" not in vars(native)


def test_unsupported_focal_order_rejected(native, tmp_path):
    native.robot.root_view.prim_paths[0] = "/World/envs/env_0/Robot/wrong"
    with pytest.raises(ValueError, match="resolved articulation"):
        reset_evidence.ResetEvidence(native, tmp_path, "solo")


@pytest.mark.parametrize("suffix", ["/base", "/assembly/floating_link"])
def test_actual_usd_nested_roots_and_reordered_view_bind_focal_values(
    native, tmp_path, suffix
):
    from pxr import UsdPhysics

    stage = native.sim.stage
    for index in range(2):
        path = f"/World/envs/env_{index}/Robot"
        stage.GetPrimAtPath(path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
        UsdPhysics.ArticulationRootAPI.Apply(stage.DefinePrim(path + suffix, "Xform"))
    native.robot.root_view.prim_paths = [
        f"/World/envs/env_{index}/Robot{suffix}" for index in (1, 0)
    ]
    poses = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    native.robot.root_view.get_root_transforms = lambda: poses
    before = stage.GetRootLayer().ExportToString()
    with reset_evidence.ResetEvidence(native, tmp_path, "solo").record():
        native.step(torch.zeros(2, 3))
    record, arrays = _read(tmp_path)
    assert record["focal_native_index"] == 1
    assert record["focal_native_root_path"] == f"/World/envs/env_0/Robot{suffix}"
    post = next(event for event in record["events"] if event["phase"] == "post_physics")
    assert arrays[post["arrays"]["root_transform_xyzw"]].tolist() == poses[1].tolist()
    observation = next(
        event for event in record["events"] if event["phase"] == "low_level_observation"
    )
    assert arrays[observation["arrays"]["value"]].tolist() == [4, 5, 6, 7]
    assert stage.GetRootLayer().ExportToString() == before


def test_partial_installation_restores_preceding_patches(native, tmp_path):
    native.robot.actuators["second"] = SimpleNamespace()
    with pytest.raises(AttributeError, match="compute"):
        with reset_evidence.ResetEvidence(native, tmp_path, "solo").record():
            pytest.fail("unsupported setup entered native probe")
    assert "step" not in vars(native)
    assert "update" not in vars(native.scene)
    assert "compute" not in vars(native.actuator)
    assert "network" not in vars(native.actuator)


def test_real_torchscript_callable_is_forwarded_and_restored(native, tmp_path):
    class Network(torch.nn.Module):
        def forward(self, value, memory):
            return value[:, :, 0], (memory[0] + 1, memory[1] + 2)

    original = torch.jit.trace(
        Network(),
        (
            native.actuator.sea_input,
            (native.actuator.sea_hidden_state, native.actuator.sea_cell_state),
        ),
    )
    native.actuator.network = original
    with reset_evidence.ResetEvidence(native, tmp_path, "solo").record():
        native.step(torch.zeros(2, 3))
    assert native.actuator.network is original
    record, arrays = _read(tmp_path)
    before = next(
        event for event in record["events"] if event["phase"] == "network_before"
    )
    after = next(
        event for event in record["events"] if event["phase"] == "network_after"
    )
    assert arrays[before["arrays"]["hidden"]].max() == 0
    assert arrays[after["arrays"]["hidden"]].min() == 1


def test_network_wrapper_preserves_argument_and_return_objects(native, tmp_path):
    recorder = reset_evidence.ResetEvidence(native, tmp_path, "solo")
    recorder.active = True
    memory = (native.actuator.sea_hidden_state, native.actuator.sea_cell_state)
    expected = (native.actuator.sea_input[:, :, 0], memory)

    def original(value, state):
        assert value is native.actuator.sea_input and state is memory
        return expected

    wrapped = recorder._network_factory("legs", native.actuator)(original)
    assert wrapped(native.actuator.sea_input, memory) is expected


def test_runtime_fingerprints_read_flags_without_changing_them(monkeypatch):
    visited = []

    def distribution(name):
        visited.append(name)
        return SimpleNamespace(version="fixture", read_text=lambda _: "record fixture")

    monkeypatch.setattr(reset_evidence.metadata, "distribution", distribution)
    monkeypatch.setattr(
        reset_evidence, "_source_files", lambda _: {"fixture": "a" * 64}
    )
    before = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    result = reset_evidence._runtime(None)
    assert before == (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    assert result["cudnn_benchmark"] == before[0]
    assert result["cudnn_deterministic"] == before[1]
    assert visited == ["torch", "isaaclab", "isaacsim", "warp-lang"]
    assert (
        result["packages"]["isaaclab"]["record_sha256"]
        == hashlib.sha256(b"record fixture").hexdigest()
    )


def test_source_fingerprint_includes_inherited_python_methods(native):
    class EnvironmentChild(Environment):
        pass

    native.__class__ = EnvironmentChild
    sources = reset_evidence._source_files(native)
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert sources[f"{Environment.__module__}.{Environment.__qualname__}"] == expected
    assert (
        sources[f"{EnvironmentChild.__module__}.{EnvironmentChild.__qualname__}"]
        == expected
    )
    assert not any(name.startswith("builtins.") for name in sources)


def test_missing_distribution_metadata_is_explicit_not_inferred(monkeypatch):
    def absent(name):
        raise reset_evidence.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(reset_evidence.metadata, "distribution", absent)
    assert reset_evidence._package_record("isaacsim") == {
        "status": "unavailable",
        "version": None,
        "record_sha256": None,
    }


@pytest.mark.parametrize("name", ["solo", "repeat", "overlap", "obstacle"])
def test_reference_context_wires_each_probe_before_gates(
    native, tmp_path, monkeypatch, name
):
    from npa.workflows.navigation import contact_evidence

    previous = native.npa_contacts.evidence
    monkeypatch.setattr(contact_evidence, "ContactEvidence", lambda *args: previous)
    with reference.record_probe_contacts(
        SimpleNamespace(unwrapped=native), tmp_path, name
    ):
        native.step(torch.zeros(2, 3))
    assert native.npa_contacts.evidence is previous
    record, _ = _read(tmp_path, name)
    assert record["schema"] == "npa.navigation.reset-first-interval.v1"
