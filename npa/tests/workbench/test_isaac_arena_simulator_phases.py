"""Prove phase diagnostics expose blocked boundaries without leaking or changing work."""

import json
import inspect
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from npa.workbench.isaac_arena import simulator_phases as phases


def _rows(directory: Path, rank: int = 0) -> list[dict]:
    path = directory / f"simulator-phases-rank{rank}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_begin_is_readable_while_native_operation_has_not_returned(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def operation():
        with phases.phase_scope(env, "env_step", 5):
            entered.set()
            release.wait()

    thread = threading.Thread(target=operation)
    thread.start()
    entered.wait()
    try:
        assert [(row["phase"], row["event"]) for row in _rows(tmp_path)] == [
            ("env_step", "begin")
        ]
    finally:
        release.set()
        thread.join()
    assert [row["event"] for row in _rows(tmp_path)] == ["begin", "end"]


def test_native_exception_is_identical_and_its_contents_never_enter_journal(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path, rank=2)
    error = RuntimeError("private replay payload and private input location")
    with pytest.raises(RuntimeError) as caught:
        with phases.phase_scope(env, "policy_action", 5):
            raise error
    assert caught.value is error
    rows = _rows(tmp_path, 2)
    assert [row["event"] for row in rows] == ["begin", "failed"]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert all(row["rank"] == 2 and row["action_step"] == 5 for row in rows)
    assert rows[1]["monotonic_ns"] >= rows[0]["monotonic_ns"]
    assert "private" not in json.dumps(rows)
    assert (tmp_path / "simulator-phases-rank2.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fail_operation", [False, True])
def test_unwritable_journal_never_changes_result_or_masks_exception(
    tmp_path, fail_operation
):
    target = tmp_path / "unrelated.txt"
    target.write_text("preserve")
    (tmp_path / "simulator-phases-rank0.jsonl").symlink_to(target)
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    error = RuntimeError("native error")
    result = object()

    def operation():
        with phases.phase_scope(env, "env_step", 1):
            if fail_operation:
                raise error
            return result

    if fail_operation:
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is error
    else:
        assert operation() is result
    assert env._npa_phase_journal.write_failed
    assert target.read_text() == "preserve"


def _pink_environment(controller):
    term_type = type("PinkIKAction", (), {})
    term_type.__module__ = "isaaclab.envs.mdp.actions.pink_task_space_actions"
    term = term_type()
    term._ik_controllers = [controller]
    return SimpleNamespace(
        action_manager=SimpleNamespace(
            active_terms=["arms"],
            get_term=lambda name: term,
        )
    )


def test_pink_wrapper_preserves_each_call_arguments_result_and_exception(tmp_path):
    calls = []
    result, joint_state = object(), object()
    error = RuntimeError("private solver details")

    class Controller:
        def compute(self, positions, dt, *, fail=False):
            calls.append((positions, dt, fail))
            if fail:
                raise error
            return result

    Controller.__module__ = "isaaclab.controllers.pink_ik.pink_ik"
    controller = Controller()
    other = Controller()
    env = _pink_environment(controller)
    phases.configure_phase_journal(env, tmp_path)
    phases.configure_phase_journal(env, tmp_path)
    with phases.phase_scope(env, "env_step", 5):
        assert controller.compute(joint_state, 0.005) is result
        with pytest.raises(RuntimeError) as caught:
            controller.compute(joint_state, 0.005, fail=True)
    assert caught.value is error
    assert calls == [(joint_state, 0.005, False), (joint_state, 0.005, True)]
    assert "compute" not in other.__dict__
    rows = _rows(tmp_path)
    assert [row["event"] for row in rows if row["phase"] == "pink_ik"] == [
        "begin",
        "end",
        "begin",
        "failed",
    ]
    assert all(row["action_step"] == 5 for row in rows)
    assert "private" not in json.dumps(rows)


def test_other_controller_contracts_are_not_wrapped(tmp_path):
    controller = SimpleNamespace(compute=lambda *args: None)
    original = controller.compute
    env = _pink_environment(controller)
    phases.configure_phase_journal(env, tmp_path)
    assert controller.compute is original


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "private input location"),
        ("action_step", True),
        ("render_call", -1),
        ("render_call", 0),
    ],
)
def test_phase_rejects_arbitrary_text_and_invalid_counters(tmp_path, field, value):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    arguments = {"phase": "capture", "action_step": 0, "render_call": 1}
    arguments[field] = value
    with pytest.raises(ValueError, match="invalid simulator"):
        with phases.phase_scope(env, **arguments):
            pytest.fail("invalid diagnostic arguments reached operation")
    assert not list(tmp_path.iterdir())


def test_readiness_records_only_observed_boolean_or_null_flags(tmp_path):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    phases.record_readiness(
        env,
        1,
        stage_ready=False,
        annotator_ready=None,
        nonblack_rgb=None,
    )
    row = _rows(tmp_path)[0]
    assert row["stage_ready"] is False and row["annotator_ready"] is None
    with pytest.raises(ValueError, match="invalid simulator readiness"):
        phases.record_readiness(env, 2, private_payload="never publish")
    assert len(_rows(tmp_path)) == 1


@pytest.mark.parametrize(
    "phase,event,fields",
    [
        ("private phase", "begin", {}),
        ("env_step", "private error", {}),
        ("env_step", "failed", {"exception": "private exception content"}),
        ("env_step", "begin", {"render_call": "private value"}),
        ("capture", "unavailable", {}),
        ("physics_step", "unavailable", {"reason": "private binding details"}),
        (
            "capture_readiness",
            "observed",
            {
                "render_call": 1,
                "stage_ready": True,
                "annotator_ready": True,
                "nonblack_rgb": "private image contents",
            },
        ),
        (
            "capture_readiness",
            "observed",
            {
                "render_call": 1,
                "stage_ready": True,
                "annotator_ready": True,
                "nonblack_rgb": True,
                "asset_path": "private location",
            },
        ),
    ],
)
def test_write_sink_rejects_unapproved_fields_even_from_internal_callers(
    tmp_path, phase, event, fields
):
    env = SimpleNamespace()
    phases.configure_phase_journal(env, tmp_path)
    with pytest.raises(ValueError) as caught:
        env._npa_phase_journal.emit(phase, event, 5, **fields)
    assert "private" not in str(caught.value)
    assert not list(tmp_path.iterdir())


def _stepping_environment(block):
    calls = []

    class Physics:
        @classmethod
        def wait_for_playing(cls):
            calls.append("wait")

        @classmethod
        def step(cls):
            calls.append("physics")
            block()

    def step(*, render):
        calls.append(("simulation", render))
        Physics.wait_for_playing()
        Physics.step()

    env = SimpleNamespace(
        action_manager=SimpleNamespace(apply_action=lambda: calls.append("action")),
        scene=SimpleNamespace(
            write_data_to_sim=lambda: calls.append("write"),
            update=lambda dt: calls.append(("update", dt)),
        ),
        sim=SimpleNamespace(
            physics_manager=Physics,
            step=step,
            render=lambda: calls.append("render"),
        ),
    )
    return env, calls


def _step_environment(env):
    with phases.phase_scope(env, "env_step", 5):
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.sim.render()
        env.scene.update(0.005)


def test_native_step_boundary_is_visible_before_the_blocked_call_returns(tmp_path):
    entered, release = threading.Event(), threading.Event()
    env, calls = _stepping_environment(lambda: (entered.set(), release.wait()))
    phases.configure_phase_journal(env, tmp_path)
    thread = threading.Thread(target=_step_environment, args=(env,))
    thread.start()
    entered.wait()
    try:
        rows = _rows(tmp_path)
        assert (rows[-1]["phase"], rows[-1]["event"]) == ("physics_step", "begin")
        assert all(row["action_step"] == 5 for row in rows)
        assert "render" not in calls
    finally:
        release.set()
        thread.join()
        phases.finalize_phase_journal(env)
    assert calls == [
        "action",
        "write",
        ("simulation", False),
        "wait",
        "physics",
        "render",
        ("update", 0.005),
    ]
    rows = _rows(tmp_path)
    entered_phases = [row["phase"] for row in rows if row["event"] == "begin"]
    assert entered_phases == [
        "env_step",
        "action_apply",
        "scene_write",
        "simulation_step",
        "physics_wait",
        "physics_step",
        "simulation_render",
        "scene_update",
    ]
    assert len(rows) == 2 * len(entered_phases)


def _physics_hierarchy(calls, result, error):
    class Base:
        @classmethod
        def step(cls, value, *, fail=False):
            calls.append((cls, value, fail))
            if fail:
                raise error
            return result

    class Concrete(Base):
        pass

    class Derived(Concrete):
        pass

    return Base, Concrete, Derived


@pytest.mark.parametrize("inherited", [False, True])
def test_native_classmethod_binding_and_exception_survive_observation(
    tmp_path, inherited
):
    calls, result, argument = [], object(), object()
    error = RuntimeError("private native state")
    base, concrete, derived = _physics_hierarchy(calls, result, error)
    original = inspect.getattr_static(base, "step")
    env = SimpleNamespace(
        sim=SimpleNamespace(physics_manager=concrete if inherited else base)
    )
    phases.configure_phase_journal(env, tmp_path)
    try:
        with phases.phase_scope(env, "env_step", 3):
            for caller in (concrete, concrete(), derived, derived()):
                assert caller.step(argument) is result
            with pytest.raises(RuntimeError) as caught:
                derived.step(argument, fail=True)
        assert caught.value is error
    finally:
        phases.finalize_phase_journal(env)
    assert calls == [(concrete, argument, False)] * 2 + [
        (derived, argument, False),
        (derived, argument, False),
        (derived, argument, True),
    ]
    assert "step" not in vars(concrete)
    assert inspect.getattr_static(base, "step") is original
    rows = _rows(tmp_path)
    assert sum(row["phase"] == "physics_step" for row in rows) == 10
    assert "private" not in json.dumps(rows)


@pytest.mark.parametrize("relationship", ["same", "ancestor", "descendant"])
def test_overlapping_class_owner_is_rejected_and_released_on_teardown(
    tmp_path, relationship
):
    base, concrete, derived = _physics_hierarchy([], object(), RuntimeError())
    first = SimpleNamespace(sim=SimpleNamespace(physics_manager=concrete))
    target = {"same": concrete, "ancestor": base, "descendant": derived}[relationship]
    second = SimpleNamespace(sim=SimpleNamespace(physics_manager=target))
    phases.configure_phase_journal(first, tmp_path)
    installed = inspect.getattr_static(concrete, "step")
    try:
        with pytest.raises(RuntimeError, match="active owner"):
            phases.configure_phase_journal(second, tmp_path)
        assert inspect.getattr_static(concrete, "step") is installed
        assert not hasattr(second, "_npa_phase_journal")
    finally:
        phases.finalize_phase_journal(first)
    phases.configure_phase_journal(second, tmp_path)
    phases.finalize_phase_journal(second)
    assert "step" not in vars(concrete)


def test_retained_foreign_wrapper_chain_survives_teardown_and_same_env_reuse(tmp_path):
    env, calls = _stepping_environment(lambda: None)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    original = env.sim.step
    phases.configure_phase_journal(env, first)
    retained = env.sim.step

    def foreign(*args, **kwargs):
        return retained(*args, **kwargs)

    env.sim.step = foreign
    _step_environment(env)
    phases.finalize_phase_journal(env)
    assert env.sim.step is foreign
    previous = (first / "simulator-phases-rank0.jsonl").read_bytes()
    env.sim.step(render=False)
    phases.configure_phase_journal(env, second)
    try:
        _step_environment(env)
    finally:
        phases.finalize_phase_journal(env)
    assert (first / "simulator-phases-rank0.jsonl").read_bytes() == previous
    assert sum(row["phase"] == "simulation_step" for row in _rows(second)) == 2
    assert env.sim.step is foreign and original is not foreign
    assert calls.count(("simulation", False)) == 3


def test_restore_failure_preserves_native_exception_and_ownership_until_retry(
    tmp_path, monkeypatch
):
    env, _calls = _stepping_environment(lambda: None)
    phases.configure_phase_journal(env, tmp_path)
    journal = env._npa_phase_journal
    restore = phases._ObservedMethod.restore
    error = RuntimeError("native rollout failed")

    def unavailable(_observer):
        raise RuntimeError("temporary restoration failure")

    monkeypatch.setattr(phases._ObservedMethod, "restore", unavailable)
    with pytest.raises(RuntimeError) as caught:
        try:
            raise error
        finally:
            phases.finalize_phase_journal(env)
    assert caught.value is error
    assert journal.observers and env._npa_phase_journal is journal
    assert phases._CLASS_OWNERS[env.sim.physics_manager] is journal
    monkeypatch.setattr(phases._ObservedMethod, "restore", restore)
    phases.finalize_phase_journal(env)
    assert not journal.observers and not hasattr(env, "_npa_phase_journal")
    assert env.sim.physics_manager not in phases._CLASS_OWNERS


def test_retained_foreign_classmethod_keeps_subclass_binding_after_reuse(tmp_path):
    calls, result, argument = [], object(), object()
    _base, concrete, derived = _physics_hierarchy(calls, result, RuntimeError())
    env = SimpleNamespace(sim=SimpleNamespace(physics_manager=concrete))
    phases.configure_phase_journal(env, tmp_path)
    retained = inspect.getattr_static(concrete, "step")

    @classmethod
    def foreign(cls, *args, **kwargs):
        return retained.__get__(None, cls)(*args, **kwargs)

    concrete.step = foreign
    phases.finalize_phase_journal(env)
    assert derived().step(argument) is result
    assert not list(tmp_path.iterdir())
    phases.configure_phase_journal(env, tmp_path)
    try:
        assert derived.step(argument) is result
    finally:
        phases.finalize_phase_journal(env)
    assert inspect.getattr_static(concrete, "step") is foreign
    assert calls == [(derived, argument, False)] * 2
    assert len(_rows(tmp_path)) == 2


def test_unsupported_native_binding_rolls_back_previously_installed_hooks(tmp_path):
    env, _calls = _stepping_environment(lambda: None)

    class Unsupported:
        @classmethod
        def wait_for_playing(cls):
            pass

        @staticmethod
        def step():
            pass

    original_wait = inspect.getattr_static(Unsupported, "wait_for_playing")
    original_step, original_update = env.sim.step, env.scene.update
    env.sim.physics_manager = Unsupported
    with pytest.raises(RuntimeError, match="native classmethod"):
        phases.configure_phase_journal(env, tmp_path)
    assert env.sim.step is original_step and env.scene.update is original_update
    assert inspect.getattr_static(Unsupported, "wait_for_playing") is original_wait
    assert not hasattr(env, "_npa_phase_journal")
    assert Unsupported not in phases._CLASS_OWNERS


def _immutable_physics(binding, calls, result, error):
    class SlottedPhysics:
        __slots__ = ()

        def wait_for_playing(self):
            calls.append("wait")

        def step(self, value, *, fail=False):
            calls.append((value, fail))
            if fail:
                raise error
            return result

    class ReadOnlyPhysics(SlottedPhysics):
        def __setattr__(self, name, value):
            raise AttributeError("native method is read-only")

    return SlottedPhysics() if binding == "slotted" else ReadOnlyPhysics()


@pytest.mark.parametrize("binding", ["slotted", "read_only"])
@pytest.mark.parametrize("fail", [False, True])
def test_immutable_native_binding_never_blocks_or_changes_rollout(
    tmp_path, binding, fail
):
    calls, result, argument = [], object(), object()
    error = RuntimeError("private native failure")
    physics = _immutable_physics(binding, calls, result, error)
    original = inspect.getattr_static(type(physics), "step")

    def step(value, *, fail=False):
        physics.wait_for_playing()
        return physics.step(value, fail=fail)

    env = SimpleNamespace(sim=SimpleNamespace(physics_manager=physics, step=step))
    phases.configure_phase_journal(env, tmp_path)
    try:
        with phases.phase_scope(env, "env_step", 1):
            if fail:
                with pytest.raises(RuntimeError) as caught:
                    env.sim.step(argument, fail=True)
                assert caught.value is error
            else:
                assert env.sim.step(argument) is result
    finally:
        phases.finalize_phase_journal(env)
    _assert_immutable_rollout(tmp_path, env, physics, original, step, fail)
    assert calls == ["wait", (argument, fail)]


def _assert_immutable_rollout(directory, env, physics, original, step, fail):
    assert env.sim.step is step
    assert inspect.getattr_static(type(physics), "step") is original
    assert type(physics) not in phases._CLASS_OWNERS
    assert not hasattr(env, "_npa_phase_journal")
    rows = _rows(directory)
    assert [(row["phase"], row["event"]) for row in rows[:2]] == [
        ("physics_wait", "unavailable"),
        ("physics_step", "unavailable"),
    ]
    assert all(row["action_step"] == 0 for row in rows[:2])
    assert [(row["phase"], row["event"]) for row in rows[2:]] == [
        ("env_step", "begin"),
        ("simulation_step", "begin"),
        ("simulation_step", "failed" if fail else "end"),
        ("env_step", "end"),
    ]
    assert "private" not in json.dumps(rows)
