"""Verify simulator video captures terminal state before the base environment resets."""

from __future__ import annotations

import hashlib
import json
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from npa.workbench.isaac_arena import simulator_video
from npa.workbench.isaac_arena.simulator_phases import (
    configure_phase_journal,
    phase_scope,
)


class _RecorderTerm:
    def __init__(self, cfg, env):
        self._env = env


class _Wrapper:
    def __init__(self, env):
        self.env = env

    @property
    def unwrapped(self):
        return self.env

    def step(self, action):
        return self.env.step(action)


class _Settings:
    def __init__(self):
        self.values = {
            simulator_video._PLAY_SIMULATIONS: True,
            **simulator_video._STARTUP_RENDER_SETTINGS,
        }
        self.set_calls = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value


class _Subscription:
    def __init__(self, callback):
        self.callback = callback


def _subscribe(runtime, callback):
    subscription = _Subscription(callback)
    runtime.subscriptions.append(weakref.ref(subscription))
    return subscription


def _emit_native_step(runtime, delta):
    for reference in runtime.subscriptions:
        subscription = reference()
        if subscription is not None:
            subscription.callback(delta)


def _install_physx(monkeypatch, runtime):
    omni = ModuleType("omni")
    physx = ModuleType("omni.physx")
    physx.get_physx_interface = lambda: SimpleNamespace(
        subscribe_physics_step_events=lambda callback: _subscribe(runtime, callback)
    )
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)


@pytest.fixture
def simulator_modules(monkeypatch):
    managers = ModuleType("isaaclab.managers.recorder_manager")
    managers.RecorderTerm = _RecorderTerm
    managers.RecorderTermCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(
        sys.modules, "isaaclab.managers", ModuleType("isaaclab.managers")
    )
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    gym = ModuleType("gymnasium")
    gym.Wrapper = _Wrapper
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setattr(
        simulator_video, "_step_tensor", lambda step, env: np.array([[step]])
    )
    runtime = SimpleNamespace(
        settings=_Settings(), env=None, resets=0, subscriptions=[]
    )
    context = SimpleNamespace(
        get_stage_streaming_status=lambda: (
            runtime.env.renders < runtime.env.streaming_renders
        ),
        get_stage_loading_status=lambda: (
            "",
            int(runtime.env.renders >= runtime.env.loading_renders),
            1,
        ),
        reset_renderer_accumulation=lambda: setattr(
            runtime, "resets", runtime.resets + 1
        ),
    )
    monkeypatch.setattr(
        simulator_video, "_kit_interfaces", lambda: (runtime.settings, context)
    )
    _install_physx(monkeypatch, runtime)
    return runtime


class _AutoResetEnvironment:
    def __init__(self, directory: Path, runtime, *, warmup_renders: int = 0):
        self.runtime = runtime
        runtime.env = self
        self.cfg = SimpleNamespace(
            recorders=SimpleNamespace(dataset_export_dir_path=str(directory)),
            sim=SimpleNamespace(render=SimpleNamespace(carb_settings={})),
        )
        self.num_envs = 1
        self.device = "cpu"
        self.state = 0
        self.physics_time = 0.0
        self.physics_steps = 0
        self.renders = 0
        self.env_renders = 0
        self.streaming_renders = 0
        self.loading_renders = 0
        self.stopped = False
        self.action_steps = []
        self.warmup_renders = warmup_renders
        self.sim = SimpleNamespace(
            is_stopped=lambda: self.stopped,
            get_physics_step_count=lambda: self.physics_steps,
            physics_manager=SimpleNamespace(forward=lambda: None),
        )
        self.scene = self._physics_scene()
        annotator = SimpleNamespace(get_data=self._annotator_data)
        self.video_recorder = SimpleNamespace(
            _capture=SimpleNamespace(_rgb_annotator=annotator),
            render_rgb_array=self._native_render,
        )
        simulator_video.configure_video_capture(self.cfg)
        runtime.settings.values.update(self.cfg.sim.render.carb_settings)
        recorder_cfg = self.cfg.recorders.npa_video
        self.recorder = recorder_cfg.class_type(recorder_cfg, self)

    def _physics_scene(self):
        view = SimpleNamespace(
            get_root_transforms=lambda: np.array([[self.state, 0.0, 0.0]]),
            get_root_velocities=lambda: np.zeros((1, 6)),
            get_dof_positions=lambda: np.array([[self.state * 0.1]]),
            get_dof_velocities=lambda: np.zeros((1, 1)),
        )
        return SimpleNamespace(
            articulations={"robot": SimpleNamespace(root_view=view)},
            rigid_objects={},
            rigid_object_collections={},
            deformable_objects={},
        )

    def _pixels(self):
        return np.full((24, 32, 3), 40 + self.state * 40, dtype=np.uint8)

    def _annotator_data(self):
        # An initialized annotator can still return a nonempty black buffer.
        return (
            self._pixels()
            if self.renders >= self.warmup_renders
            else np.zeros((24, 32, 3), dtype=np.uint8)
        )

    def render(self):
        self.env_renders += 1
        # Mirror upstream KitVisualizer restoring True before native capture.
        self.runtime.settings.set(simulator_video._PLAY_SIMULATIONS, True)
        return self._native_render()

    def _native_render(self):
        if self.runtime.settings.get(simulator_video._PLAY_SIMULATIONS):
            self.state += 1
            self.physics_time += 0.005
            _emit_native_step(self.runtime, 0.005)
        self.renders += 1
        return (
            self._pixels()
            if self.renders >= self.warmup_renders
            else np.zeros((24, 32, 3), dtype=np.uint8)
        )

    def reset(self):
        self.recorder.record_pre_reset([0])
        self.state = 0
        self.recorder.record_post_reset([0])

    def step(self, action):
        self.state += action
        self.physics_steps += 4
        self.physics_time += 0.02
        for _ in range(4):
            _emit_native_step(self.runtime, 0.005)
        key, value = self.recorder.record_post_step()
        assert key == "npa_video/action_step"
        self.action_steps.append(int(value[0, 0]))
        terminal = self.state >= 3
        if terminal:
            self.recorder.record_pre_reset([0])
            self.state = 0
            self.recorder.record_post_reset([0])
        return self.state, terminal


def _assert_freeze_trace(evidence, expected_steps):
    assert [
        check["action_step"] for check in evidence["physics_freeze_checks"]
    ] == expected_steps
    for check in evidence["physics_freeze_checks"]:
        assert check["state_sha256_before"] == check["state_sha256_after"]
        assert check["physics_time_before"] == check["physics_time_after"]
        assert check["physics_step_before"] == check["physics_step_after"]
        assert check["native_physics_step_before"] == check["native_physics_step_after"]
        assert check["pre_settling_render_calls"] >= 1
        assert (
            check["settling_render_calls"] >= simulator_video._MINIMUM_SETTLING_RENDERS
        )
        assert check["render_calls"] == (
            check["pre_settling_render_calls"] + check["settling_render_calls"]
        )
        assert check["consecutive_ready_render_calls"] == (
            check["settling_render_calls"] + 1
        )


def test_interrupted_readiness_restarts_the_full_settling_window(monkeypatch) -> None:
    readiness = iter([True, True, False, *([True] * 9)])
    frame = np.ones((24, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(simulator_video, "_render_frame", lambda _env: frame)
    monkeypatch.setattr(
        simulator_video,
        "_ready_frame",
        lambda _env, _context, _frame: next(readiness),
    )
    captured, renders, settling = simulator_video._ready_capture_frame(
        object(), object()
    )
    np.testing.assert_array_equal(captured, frame)
    assert renders == 12
    assert settling == simulator_video._MINIMUM_SETTLING_RENDERS


def test_phase_journal_distinguishes_capture_from_remaining_environment_step(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules, warmup_renders=2)
    configure_phase_journal(env, tmp_path)
    journal = tmp_path / "simulator-phases-rank0.jsonl"
    env.reset()
    with phase_scope(env, "env_step", 1):
        env.step(1)
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        assert rows[-1]["phase"] == "capture" and rows[-1]["event"] == "end"
        assert rows[-1]["action_step"] == 1
        assert not any(
            row["phase"] == "env_step" and row["event"] == "end" for row in rows
        )
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["phase"] == "env_step" and rows[-1]["event"] == "end"
    readiness = [row for row in rows if row["phase"] == "capture_readiness"]
    assert readiness[0]["nonblack_rgb"] is False
    assert readiness[-1]["nonblack_rgb"] is True
    renders = [row for row in rows if row["phase"] == "render_call"]
    assert len(renders) == 2 * env.renders
    assert [row["event"] for row in renders] == ["begin", "end"] * env.renders
    assert env.physics_steps == 4 and env.state == 1


def test_render_failure_retains_begin_and_preserves_native_exception(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    configure_phase_journal(env, tmp_path)
    error = RuntimeError("private renderer context")

    def fail_render():
        rows = [
            json.loads(line)
            for line in (tmp_path / "simulator-phases-rank0.jsonl")
            .read_text()
            .splitlines()
        ]
        assert rows[-1]["phase"] == "render_call" and rows[-1]["event"] == "begin"
        raise error

    env.video_recorder.render_rgb_array = fail_render
    with pytest.raises(RuntimeError) as caught:
        env.reset()
    assert caught.value is error
    text = (tmp_path / "simulator-phases-rank0.jsonl").read_text()
    assert "private renderer" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert [(row["phase"], row["event"]) for row in rows[-2:]] == [
        ("render_call", "failed"),
        ("capture", "failed"),
    ]
    assert env.runtime.settings.get(simulator_video._PLAY_SIMULATIONS) is True
    assert env.physics_steps == 0 and env.state == 0


def test_video_records_terminal_frame_before_autoreset(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    frames = []
    for _ in range(3):
        wrapper.step(1)
        frames.append(wrapper.render())
    assert env.state == 0
    assert [int(frame[0, 0, 0]) for frame in frames] == [80, 120, 160]
    assert env.action_steps == [1, 2, 3]
    assert env.renders == 4 * (simulator_video._MINIMUM_SETTLING_RENDERS + 1)
    assert env.env_renders == 0
    assert env.physics_time == 0.06
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    assert evidence["captured_action_steps"] == 3
    assert evidence["schema"] == "npa.isaac-arena.video-capture.v2"
    assert evidence["physics_clock"] == "native_physx_step_events_since_capture_setup"
    assert evidence["capture_phase"] == "recorder_post_step_before_autoreset"
    assert evidence["initial"]["action_step"] == 0
    assert evidence["terminals"][0]["action_step"] == 3
    assert evidence["terminals"][0]["source_frame_index"] == 2
    final = np.asarray(Image.open(tmp_path / evidence["terminals"][0]["path"]))
    np.testing.assert_array_equal(final, frames[-1])
    assert (
        evidence["terminals"][0]["decoded_rgb_sha256"]
        == hashlib.sha256(final.tobytes()).hexdigest()
    )
    _assert_freeze_trace(evidence, [0, 1, 2, 3])


def test_initial_renderer_warmup_adds_no_action_or_video_frames(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules, warmup_renders=3)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    assert env.renders == env.warmup_renders + simulator_video._MINIMUM_SETTLING_RENDERS
    assert env.physics_time == 0.0
    assert env.state == 0
    assert env.action_steps == []
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    initial = np.asarray(Image.open(tmp_path / evidence["initial"]["path"]))
    assert int(initial[0, 0, 0]) == 40
    assert evidence["captured_action_steps"] == 0
    with pytest.raises(RuntimeError, match="before its first action"):
        wrapper.render()
    wrapper.step(1)
    assert int(wrapper.render()[0, 0, 0]) == 80
    assert env.action_steps == [1]


def test_outer_video_reads_copy_without_rerendering(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    wrapper = simulator_video.cached_frame_env(env)
    env.reset()
    wrapper.step(1)
    rendered = wrapper.render()
    rendered[:] = 0
    assert int(wrapper.render()[0, 0, 0]) == 80
    assert env.renders == 2 * (simulator_video._MINIMUM_SETTLING_RENDERS + 1)


def test_capture_requires_real_renderer_frame(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    env.video_recorder.render_rgb_array = lambda: None
    with pytest.raises(RuntimeError, match="no RGB uint8 frame"):
        env.reset()
    assert not (tmp_path / "simulator-initial.png").exists()


def test_capture_setup_preserves_existing_metric_configuration(
    simulator_modules, tmp_path: Path
) -> None:
    success_recorder = object()
    cfg = SimpleNamespace(
        recorders=SimpleNamespace(
            success=success_recorder, dataset_export_dir_path=str(tmp_path)
        ),
        sim=SimpleNamespace(
            render=SimpleNamespace(
                carb_settings={"/rtx/sceneDb/ambientLightIntensity": 0.0}
            )
        ),
    )
    simulator_video.configure_video_capture(cfg)
    assert cfg.recorders.success is success_recorder
    assert cfg.sim.render.carb_settings["/rtx/sceneDb/ambientLightIntensity"] == 0.0
    assert cfg.sim.render.carb_settings == {
        "/rtx/rendermode": "RaytracedLighting",
        "/rtx/post/aa/op": 1,
        "/rtx/post/dlss/execMode": 2,
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx/ecoMode/enabled": False,
        "/rtx/directLighting/sampledLighting/samplesPerPixel": 32,
        "/rtx/indirectDiffuse/fetchSampleCount": 32,
        "/rtx/reflections/sampledLighting/samplesPerPixel": 16,
        "/rtx/sceneDb/ambientLightIntensity": 0.0,
    }
    assert cfg.sim.render.antialiasing_mode == "TAA"
    assert cfg.sim.render.dlss_mode == 2
    assert cfg.sim.render.enable_dl_denoiser is True
    assert cfg.sim.render.samples_per_pixel == 32
    assert (
        cfg.recorders.npa_video.class_type.__mro__[1]
        is simulator_video._VideoCaptureMethods
    )
    with pytest.raises(RuntimeError, match="requires the task's metric recorder"):
        simulator_video.configure_video_capture(SimpleNamespace(recorders=None))


def test_texture_streaming_and_asset_loading_finish_before_capture(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules, warmup_renders=3)
    env.loading_renders = 5
    env.streaming_renders = 7
    env.reset()
    assert env.renders == (
        env.streaming_renders + simulator_video._MINIMUM_SETTLING_RENDERS
    )
    assert simulator_modules.resets == 0
    assert env.state == 0
    assert env.physics_time == 0.0
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    check = evidence["physics_freeze_checks"][0]
    assert check["stage_streaming_idle"] is True
    assert check["stage_assets_loaded"] is True
    assert check["pre_settling_render_calls"] == env.streaming_renders
    assert check["settling_render_calls"] == simulator_video._MINIMUM_SETTLING_RENDERS
    assert check["consecutive_ready_render_calls"] == (
        simulator_video._MINIMUM_SETTLING_RENDERS + 1
    )


@pytest.mark.parametrize("previous", [True, False])
def test_native_render_freezes_physics_and_restores_original_setting(
    simulator_modules, tmp_path: Path, previous: bool
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    settings = simulator_modules.settings
    settings.set(simulator_video._PLAY_SIMULATIONS, previous)
    env.reset()
    assert settings.get(simulator_video._PLAY_SIMULATIONS) is previous
    assert env.physics_time == 0.0
    assert env.state == 0
    assert env._npa_video_initial_physics_state["state"]["articulations/robot"][
        "get_dof_positions"
    ] == [[0.0]]


@pytest.mark.parametrize("previous", [True, False])
def test_renderer_exception_restores_setting_and_retains_no_initial_frame(
    simulator_modules, tmp_path: Path, previous: bool
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    simulator_modules.settings.set(simulator_video._PLAY_SIMULATIONS, previous)

    def broken_renderer():
        assert (
            simulator_modules.settings.get(simulator_video._PLAY_SIMULATIONS) is False
        )
        raise RuntimeError("renderer failed")

    env.video_recorder.render_rgb_array = broken_renderer
    with pytest.raises(RuntimeError, match="renderer failed"):
        env.reset()
    assert simulator_modules.settings.get(simulator_video._PLAY_SIMULATIONS) is previous
    assert not (tmp_path / "simulator-initial.png").exists()


@pytest.mark.parametrize("changed", ["state", "native_event", "physics_steps"])
def test_capture_rejects_unreported_physics_changes(
    simulator_modules, tmp_path: Path, changed: str
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native = env.video_recorder.render_rgb_array

    def drifting_renderer():
        if changed == "native_event":
            _emit_native_step(simulator_modules, 0.005)
        else:
            setattr(env, changed, getattr(env, changed) + 1)
        return native()

    env.video_recorder.render_rgb_array = drifting_renderer
    with pytest.raises(RuntimeError, match="advanced physics time, steps, or state"):
        env.reset()
    assert simulator_modules.settings.get(simulator_video._PLAY_SIMULATIONS) is True
    assert not (tmp_path / "simulator-initial.png").exists()


def test_zero_duration_native_step_during_render_fails_freeze_check(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native = env.video_recorder.render_rgb_array

    def invisible_step():
        _emit_native_step(simulator_modules, 0.0)
        return native()

    env.video_recorder.render_rgb_array = invisible_step
    with pytest.raises(RuntimeError, match="advanced physics time, steps, or state"):
        env.reset()
    assert env._npa_video_physics_clock.elapsed == 0.0
    assert (
        env._npa_video_physics_clock.steps
        == simulator_video._MINIMUM_SETTLING_RENDERS + 1
    )
    assert not (tmp_path / "simulator-initial.png").exists()


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), -0.005, None, "bad"])
def test_invalid_native_step_duration_cannot_be_silently_ignored(
    simulator_modules, tmp_path: Path, delta
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    _emit_native_step(simulator_modules, delta)
    assert env._npa_video_physics_clock.steps == 1
    with pytest.raises(RuntimeError, match="invalid native PhysX step duration"):
        env.reset()
    assert env.renders == 0


def test_native_subscription_remains_live_and_observes_real_action_steps(
    simulator_modules, tmp_path: Path
) -> None:
    import gc

    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    env.reset()
    gc.collect()
    assert simulator_modules.subscriptions[0]() is not None
    env.step(1)
    env.step(1)
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    checks = evidence["physics_freeze_checks"]
    assert [check["native_physics_step_before"] for check in checks] == [0, 4, 8]
    assert checks[-1]["physics_time_before"] == pytest.approx(0.04)
    _assert_freeze_trace(evidence, [0, 1, 2])


def test_disconnected_native_observer_fails_after_first_actual_action(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    env.reset()
    simulator_modules.subscriptions.clear()
    with pytest.raises(RuntimeError, match="did not observe native physics steps"):
        env.step(1)
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    assert evidence["captured_action_steps"] == 0
    assert len(evidence["physics_freeze_checks"]) == 1


def test_nonempty_black_annotator_does_not_become_evidence(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules, warmup_renders=100)
    native = env.video_recorder.render_rgb_array

    def stops_without_ready_frame():
        frame = native()
        env.stopped = env.renders == 2
        return frame

    env.video_recorder.render_rgb_array = stops_without_ready_frame
    with pytest.raises(
        RuntimeError, match="stopped before its RGB annotator was ready"
    ):
        env.reset()
    assert env.renders == 2
    assert not (tmp_path / "simulator-initial.png").exists()


def test_capture_reasserts_mutable_renderer_settings_after_late_override(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    simulator_modules.settings.set("/rtx/rendermode", "RealTimePathTracing")
    simulator_modules.settings.set("/rtx/post/aa/op", 1)
    simulator_modules.settings.set_calls.clear()
    env.reset()
    assert simulator_modules.settings.set_calls[
        : len(simulator_video._CAPTURE_RENDER_SETTINGS)
    ] == list(simulator_video._CAPTURE_RENDER_SETTINGS.items())
    assert simulator_modules.settings.get("/persistent/rtx/modes/rt2/enabled") is False
    assert simulator_modules.settings.get("/rtx/rendermode") == ("RaytracedLighting")
    assert simulator_modules.settings.get("/rtx/post/aa/op") == 1
    assert simulator_modules.settings.get("/rtx/post/dlss/execMode") == 2
    assert simulator_modules.settings.get("/rtx-transient/dldenoiser/enabled") is True
    assert simulator_modules.settings.get("/rtx-transient/dlssg/enabled") is False
    assert simulator_modules.settings.get("/rtx/ecoMode/enabled") is False
    assert env._npa_video_rendering["stochastic_accumulation"] is False
    assert env._npa_video_rendering["minimum_settling_renders"] == 8


def test_capture_refuses_renderer_that_rejects_required_settings(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native_set = simulator_modules.settings.set

    def reject_taa(key, value):
        native_set(key, 2 if key == "/rtx/post/aa/op" else value)

    simulator_modules.settings.set = reject_taa
    with pytest.raises(
        RuntimeError,
        match=r'required capture settings: .*"/rtx/post/aa/op": 2',
    ):
        env.reset()
    assert env.renders == 0


def test_capture_refuses_wrong_startup_renderer_registration(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    simulator_modules.settings.set("/persistent/rtx/modes/rt2/enabled", True)
    simulator_modules.settings.set("/rtx/rendermode", "RealTimePathTracing")
    with pytest.raises(
        RuntimeError,
        match=r'required capture settings: .*"/persistent/rtx/modes/rt2/enabled": true',
    ):
        env.reset()
    assert env.renders == 0


@pytest.mark.parametrize(
    "setting",
    [
        "/rtx/directLighting/sampledLighting/samplesPerPixel",
        "/rtx/indirectDiffuse/fetchSampleCount",
        "/rtx/reflections/sampledLighting/samplesPerPixel",
    ],
)
def test_capture_refuses_lighting_sample_reduction_after_render(
    simulator_modules, tmp_path: Path, setting: str
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native_render = env.video_recorder.render_rgb_array

    def reduce_samples_during_render():
        frame = native_render()
        simulator_modules.settings.set(setting, 1)
        return frame

    env.video_recorder.render_rgb_array = reduce_samples_during_render
    with pytest.raises(RuntimeError, match="required capture settings"):
        env.reset()
    assert env.renders == simulator_video._MINIMUM_SETTLING_RENDERS + 1
    assert not (tmp_path / "simulator-initial.png").exists()
    assert not (tmp_path / "simulator-video-evidence.json").exists()


def test_capture_refuses_rt2_remap_triggered_by_render(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native_render = env.video_recorder.render_rgb_array
    native_set = simulator_modules.settings.set

    def remap_during_render():
        frame = native_render()
        native_set("/persistent/rtx/modes/rt2/enabled", True)
        native_set("/rtx/rendermode", "RealTimePathTracing")
        return frame

    env.video_recorder.render_rgb_array = remap_during_render
    with pytest.raises(
        RuntimeError,
        match=r'required capture settings: .*"/rtx/rendermode": "RealTimePathTracing"',
    ):
        env.reset()
    assert env.renders == simulator_video._MINIMUM_SETTLING_RENDERS + 1
    assert not (tmp_path / "simulator-initial.png").exists()
    assert not (tmp_path / "simulator-video-evidence.json").exists()


def test_capture_refuses_generated_frames_enabled_during_render(
    simulator_modules, tmp_path: Path
) -> None:
    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    native_render = env.video_recorder.render_rgb_array

    def enable_frame_generation_during_render():
        frame = native_render()
        simulator_modules.settings.set("/rtx-transient/dlssg/enabled", True)
        return frame

    env.video_recorder.render_rgb_array = enable_frame_generation_during_render
    with pytest.raises(
        RuntimeError,
        match=r'required capture settings: .*"/rtx-transient/dlssg/enabled": true',
    ):
        env.reset()
    assert env.renders == simulator_video._MINIMUM_SETTLING_RENDERS + 1
    assert not (tmp_path / "simulator-video-evidence.json").exists()


def test_explicit_finalization_retains_diagnostic_before_simulator_teardown(
    simulator_modules, tmp_path: Path, monkeypatch
) -> None:
    from npa.workbench.isaac_arena import simulator_diagnostics

    env = _AutoResetEnvironment(tmp_path, simulator_modules)
    env.reset()
    env.step(1)
    calls = []

    def retain(actual_env, directory, count):
        calls.append((actual_env, directory, count))
        return {"path": "simulator-unscored-episode.json", "sha256": "a" * 64}

    monkeypatch.setattr(simulator_diagnostics, "retain_unscored_episode", retain)
    wrapper = SimpleNamespace(unwrapped=env)
    simulator_video.finalize_video_capture(wrapper)
    assert calls == [(env, tmp_path, 1)]
    # The actual environment deletes these during close; later idempotent
    # finalization must not try to read stopped physics or removed managers.
    del env.scene
    simulator_video.finalize_video_capture(wrapper)
    assert calls == [(env, tmp_path, 1)]
    evidence = json.loads((tmp_path / "simulator-video-evidence.json").read_text())
    assert evidence["unscored_diagnostic"]["path"] == "simulator-unscored-episode.json"
    assert evidence["terminals"] == []
