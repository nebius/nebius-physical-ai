"""Execute the generated rollout with a hermetic Isaac/RSL control environment."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
from npa.cli.isaac_lab import _build_train_trajectory_export_script


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def any(self):
        return _Tensor(self.value.any())

    def item(self):
        return self.value.item()


class _ControlEnvironment:
    """The first observation coordinate is simulated control time."""

    def __init__(self, step_dt):
        self.step_dt = step_dt
        self.unwrapped = self
        self.scene = {}
        self.index = 0
        self.closed = False
        self.recorded_times = []

    def reset(self):
        self.index = 0
        return [[0.0, 1.0]], {}

    def step(self, actions):
        assert np.asarray(actions).shape == (1, 1)
        self.recorded_times.append(self.index * self.step_dt)
        self.index += 1
        # Model the four-item RSL-RL vector wrapper result.
        return [[self.index * self.step_dt, 1.0]], [1.0], [False], {}

    def close(self):
        self.closed = True


def _install_runtime_stubs(monkeypatch, step_dt):
    env = _ControlEnvironment(step_dt)
    app = SimpleNamespace(closed=False)

    def close_app():
        app.closed = True

    app.close = close_app

    class _Runner:
        def __init__(self, wrapped, cfg, *, log_dir, device):
            assert wrapped is env

        def load(self, checkpoint):
            assert Path(checkpoint).read_bytes() == b"synthetic policy fixture"

        def get_inference_policy(self, *, device):
            return lambda obs: [[0.25]]

    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    install("isaaclab")
    install("isaaclab.app", AppLauncher=lambda **kwargs: SimpleNamespace(app=app))
    install("isaaclab_tasks")
    install(
        "isaaclab_tasks.utils",
        parse_env_cfg=lambda *args, **kwargs: SimpleNamespace(),
        load_cfg_from_registry=lambda *args: {"device": "cpu"},
    )
    install("isaaclab_rl")
    install(
        "isaaclab_rl.rsl_rl",
        RslRlVecEnvWrapper=lambda value, **kwargs: value,
        handle_deprecated_rsl_rl_cfg=lambda cfg, version: cfg,
    )
    install("rsl_rl")
    install("rsl_rl.runners", OnPolicyRunner=_Runner)
    install("gymnasium", make=lambda *args, **kwargs: env)
    install(
        "torch",
        as_tensor=lambda value: value if isinstance(value, _Tensor) else _Tensor(value),
        inference_mode=contextlib.nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "synthetic-runtime")
    # The generated exporter sets this flag; restore the ambient value after each test.
    monkeypatch.setenv("OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA", "1")
    return env, app


def _run_export(monkeypatch, tmp_path, *, step_dt):
    env, app = _install_runtime_stubs(monkeypatch, step_dt)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic policy fixture")
    raw = tmp_path / "trajectories"
    script = _build_train_trajectory_export_script(
        "Isaac-Cartpole-v0", 1, 7, str(checkpoint), str(raw), capture_rgb=False
    )
    exec(compile(script, "<generated-isaac-export>", "exec"), {"__name__": "__main__"})
    assert env.closed and app.closed
    return env, raw


@pytest.mark.parametrize("step_dt", [1 / 60, 1 / 100, 1 / 50], ids=["cartpole-60hz", "custom-100hz", "legacy-50hz"])
def test_export_metadata_and_converted_timeline_follow_control_cadence(
    monkeypatch, tmp_path, step_dt
):
    env, raw = _run_export(monkeypatch, tmp_path, step_dt=step_dt)
    meta = json.loads((raw / "meta.json").read_text())
    states = np.load(raw / "episode_000000" / "state.npy")
    actions = np.load(raw / "episode_000000" / "actions.npy")
    assert meta["policy_loaded"] is True
    assert states.shape == (7, 2)
    assert actions.shape == (7, 1)
    np.testing.assert_allclose(states[:, 0], env.recorded_times)
    dataset = tmp_path / "lerobot"
    convert(
        raw,
        dataset,
        fps=meta["fps"],
        spec=LeRobotFeatureSpec(meta["state_names"], meta["action_names"], "cartpole"),
    )
    rows = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pylist()
    observed = {
        "step_dt": step_dt,
        "expected_fps": 1 / step_dt,
        "actual_fps": meta["fps"],
        "expected_timestamps": env.recorded_times,
        "actual_timestamps": [row["timestamp"] for row in rows],
        "source": "hermetic generated exporter; no Isaac/GPU execution",
    }
    (tmp_path / "observed.json").write_text(json.dumps(observed, indent=2))
    assert meta["fps"] == pytest.approx(1 / step_dt)
    assert observed["actual_timestamps"] == pytest.approx(env.recorded_times)


@pytest.mark.parametrize("step_dt", [0.0, -0.02, float("nan"), float("inf")])
def test_export_rejects_invalid_control_cadence(monkeypatch, tmp_path, step_dt):
    env, app = _install_runtime_stubs(monkeypatch, step_dt)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic policy fixture")
    raw = tmp_path / "trajectories"
    script = _build_train_trajectory_export_script(
        "Isaac-Cartpole-v0", 1, 1, str(checkpoint), str(raw), capture_rgb=False
    )
    with pytest.raises(RuntimeError, match="control timestep must be finite and positive"):
        exec(compile(script, "<generated-isaac-export>", "exec"), {"__name__": "__main__"})
    assert env.closed and app.closed
    assert env.index == 0
    assert not (raw / "meta.json").exists()
