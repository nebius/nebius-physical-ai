"""Control-step alignment with synthetic Isaac boundaries and real local converters."""

from __future__ import annotations

from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
from npa.cli.isaac_lab import _build_train_trajectory_export_script
from npa.viz.adapters.lerobot_to_rerun import lerobot_dataset_logical_to_rerun


class _Tensor(np.ndarray):
    """Only the CPU tensor operations exercised by the generated exporter."""

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def _tensor(value):
    return np.asarray(value).view(_Tensor)


class _ControlEnvironment:
    def __init__(self, step_dt, *, episode_length=7):
        self.step_dt = step_dt
        # Distinguish physics, rendering, and control cadence.
        self.physics_dt = step_dt / 4 if isinstance(step_dt, (float, int)) else 0.01
        self.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(render_interval=8))
        self.unwrapped = self
        self.scene = {}
        self.episode_length = episode_length
        self.closed = False
        self.ticks = []
        self.actions = []
        self.episode = -1
        self._sim_step_counter = 0

    def reset(self):
        self.episode += 1
        self._sim_step_counter = 0
        self.ticks.append([])
        self.actions.append([])
        return self._observation(), {}

    def _observation(self):
        return _tensor([[self._sim_step_counter, self.episode]])

    def step(self, actions):
        self.ticks[-1].append(self._sim_step_counter)
        self.actions[-1].append(np.asarray(actions)[0].tolist())
        self._sim_step_counter += self.cfg.decimation
        done = len(self.ticks[-1]) >= self.episode_length
        return self._observation(), _tensor([1.0]), _tensor([done]), {}

    def close(self):
        self.closed = True


def _install_synthetic_runtime(monkeypatch, step_dt, *, episode_length=7):
    env = _ControlEnvironment(step_dt, episode_length=episode_length)
    app = SimpleNamespace(closed=False)
    loads = []

    def close_app():
        app.closed = True

    app.close = close_app

    class Wrapper:
        def __init__(self, wrapped, **kwargs):
            self.unwrapped = wrapped.unwrapped
            self.reset = wrapped.reset
            self.step = wrapped.step
            self.close = wrapped.close
            # A wrapper-local value must never replace the source cadence.
            self.step_dt = 1.0 / 999.0

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, path):
            loads.append(Path(path).read_bytes())

        def get_inference_policy(self, **kwargs):
            return lambda obs: _tensor([[float(obs[0, 0]) + 100.0]])

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    module("isaaclab")
    module("isaaclab.app", AppLauncher=lambda **kwargs: SimpleNamespace(app=app))
    module("gymnasium", make=lambda *args, **kwargs: env)
    module("isaaclab_tasks")
    module(
        "isaaclab_tasks.utils",
        parse_env_cfg=lambda *args, **kwargs: env.cfg,
        load_cfg_from_registry=lambda *args: {"device": "cpu"},
    )
    module("isaaclab_rl")
    module(
        "isaaclab_rl.rsl_rl",
        RslRlVecEnvWrapper=Wrapper,
        handle_deprecated_rsl_rl_cfg=lambda cfg, version: cfg,
    )
    module("rsl_rl")
    module("rsl_rl.runners", OnPolicyRunner=Runner)
    module(
        "torch",
        as_tensor=_tensor,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (
            "synthetic-runtime"
            if name in {"isaaclab", "rsl-rl-lib"}
            else original_version(name)
        ),
    )
    # The generated script sets this before AppLauncher; restore it after exec.
    monkeypatch.setenv("OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA", "1")
    return env, app, loads


def _export(tmp_path, *, episodes=2, steps=10):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint: runtime loading is stubbed")
    raw = tmp_path / "raw"
    script = _build_train_trajectory_export_script(
        "Isaac-Cartpole-v0",
        episodes,
        steps,
        str(checkpoint),
        str(raw),
        capture_rgb=False,
    )
    exec(compile(script, "<generated-trained-export>", "exec"), {})
    return raw


def _assert_converted_timeline(raw, output, times_by_episode):
    """Use exporter metadata exactly as a downstream metadata-fed conversion does."""
    meta = json.loads((raw / "meta.json").read_text())
    spec = LeRobotFeatureSpec(
        robot_type="cadence_test",
        state_names=meta["state_names"],
        action_names=meta["action_names"],
    )
    dataset = convert(raw, output / "lerobot", fps=meta["fps"], spec=spec)
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["fps"] == meta["fps"]
    rows = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pylist()
    for episode, times in enumerate(times_by_episode):
        selected = [row for row in rows if row["episode_index"] == episode]
        assert [row["frame_index"] for row in selected] == list(range(len(times)))
        np.testing.assert_allclose(
            [row["timestamp"] for row in selected], times, atol=1e-8
        )
        np.testing.assert_array_equal(
            [row["observation.state"] for row in selected],
            np.load(raw / f"episode_{episode:06d}/state.npy"),
        )
        np.testing.assert_array_equal(
            [row["action"] for row in selected],
            np.load(raw / f"episode_{episode:06d}/actions.npy"),
        )

    rrd = output / "control-cadence.rrd"
    lerobot_dataset_logical_to_rerun(
        dataset,
        rrd,
        input_episode_indices=[],
        rollout_episode_indices=list(range(len(times_by_episode))),
        feedback_by_episode={},
        max_frames_per_episode=max(map(len, times_by_episode)),
    )
    chunks = list(load_recording(rrd).chunks())
    for episode, times in enumerate(times_by_episode):
        for kind, filename in (("state", "state.npy"), ("actions", "actions.npy")):
            values = np.load(raw / f"episode_{episode:06d}" / filename)
            for dimension in range(values.shape[1]):
                entity = f"/policy_rollout/episodes/episode_{episode:06d}/{kind}/dim_{dimension:02d}"
                batches = [
                    chunk.to_record_batch()
                    for chunk in chunks
                    if str(chunk.entity_path) == entity and not chunk.is_static
                ]
                assert batches, entity
                table = pa.Table.from_batches(batches)
                decoded_times = table["frame_time"].cast(pa.int64()).to_numpy() / 1e9
                order = np.argsort(decoded_times)
                assert decoded_times[order] == pytest.approx(times, abs=1e-8)
                decoded_values = np.asarray(
                    table["Scalars:scalars"].to_pylist()
                ).reshape(-1)
                np.testing.assert_allclose(decoded_values[order], values[:, dimension])


@pytest.mark.parametrize("fps", [50.0, 60.0, 100.0, 59.94])
def test_trained_export_preserves_source_control_ticks(monkeypatch, tmp_path, fps):
    env, app, loads = _install_synthetic_runtime(monkeypatch, 1.0 / fps)
    raw = _export(tmp_path)
    assert env.closed and app.closed
    assert len(loads) == 1
    assert env.ticks == [list(range(0, 28, 4))] * 2
    meta = json.loads((raw / "meta.json").read_text())
    assert meta["episode_lengths"] == [7, 7]
    assert meta["policy_loaded"] is True  # Stubbed loader, not simulator proof.
    assert meta["rgb_enabled"] is False
    assert meta["rgb_frame_count"] == 0
    assert meta["genuine_simulator_pixels"] is False
    times = [np.asarray(ticks) * env.physics_dt for ticks in env.ticks]
    for episode, ticks in enumerate(env.ticks):
        states = np.load(raw / f"episode_{episode:06d}/state.npy")
        np.testing.assert_array_equal(states[:, 0], ticks)
        np.testing.assert_array_equal(states[:, 1], [episode] * len(ticks))
        np.testing.assert_array_equal(
            np.load(raw / f"episode_{episode:06d}/actions.npy"), env.actions[episode]
        )
    # The independent tick clock is the oracle, not the metadata under test.
    _assert_converted_timeline(raw, tmp_path, times)
    assert meta["fps"] == pytest.approx(fps)
    assert meta["control_dt"] == pytest.approx(env.step_dt)


@pytest.mark.parametrize(
    "step_dt", [0.0, -0.02, float("nan"), float("inf"), None, "bad"]
)
def test_invalid_control_cadence_closes_runtime_without_rollout(
    monkeypatch, tmp_path, step_dt
):
    env, app, loads = _install_synthetic_runtime(monkeypatch, step_dt)
    with pytest.raises(
        RuntimeError, match="control timestep must be finite and positive"
    ):
        _export(tmp_path, episodes=1, steps=1)
    assert env.closed and app.closed
    assert env.ticks == []
    assert loads == []
    assert not (tmp_path / "raw/meta.json").exists()
    assert not list((tmp_path / "raw").glob("episode_*"))


def test_missing_control_cadence_closes_runtime(monkeypatch, tmp_path):
    env, app, loads = _install_synthetic_runtime(monkeypatch, 1.0 / 60.0)
    del env.step_dt
    with pytest.raises(
        RuntimeError, match="control timestep must be finite and positive"
    ):
        _export(tmp_path, episodes=1, steps=1)
    assert env.closed and app.closed
    assert not env.ticks and not loads
    assert not (tmp_path / "raw/meta.json").exists()


def test_unrepresentable_frame_rate_closes_runtime(monkeypatch, tmp_path):
    env, app, loads = _install_synthetic_runtime(monkeypatch, 5e-324)
    with pytest.raises(
        RuntimeError, match="control timestep must yield a finite frame rate"
    ):
        _export(tmp_path, episodes=1, steps=1)
    assert env.closed and app.closed
    assert not env.ticks and not loads
    assert not (tmp_path / "raw/meta.json").exists()


# This observer runs around the unchanged production exporter in an existing
# Isaac interpreter. It observes native env.step calls without wrapping the
# environment, selecting its timestep, or replacing policy/simulator methods.
_LIVE_OBSERVER = r"""
import json
from pathlib import Path
import runpy
import sys

script = str(Path(sys.argv[1]).resolve())
observations = []
active = {}

def observe(frame, event, arg):
    if frame.f_code.co_name != "step" or event not in {"call", "return"}:
        return
    if not frame.f_globals.get("__name__", "").startswith("isaaclab.envs."):
        return
    env = frame.f_locals.get("self")
    if env is None or getattr(env, "unwrapped", None) is not env:
        return
    if event == "return":
        row = active.pop(id(frame), None)
        if row is not None:
            row["physics_tick_after"] = int(env._sim_step_counter)
        return
    caller = frame.f_back
    while caller is not None and caller.f_code.co_filename != script:
        caller = caller.f_back
    if caller is None or "episode_index" not in caller.f_globals:
        return
    row = {
        "episode": int(caller.f_globals["episode_index"]),
        "frame": int(caller.f_globals["step"]),
        "step_dt": float(env.step_dt),
        "physics_dt": float(env.physics_dt),
        "decimation": int(env.cfg.decimation),
        "render_interval": int(env.cfg.sim.render_interval),
        "physics_tick_before": int(env._sim_step_counter),
    }
    observations.append(row)
    active[id(frame)] = row

sys.setprofile(observe)
try:
    runpy.run_path(script, run_name="__main__")
finally:
    sys.setprofile(None)
    Path(sys.argv[2]).write_text(json.dumps(observations, indent=2))
"""


@pytest.mark.gpu
def test_existing_isaac_trained_rollout_cadence(tmp_path):
    """Opt-in real RT-core runtime/checkpoint; creates only test-owned local outputs.

    Set NPA_INTEGRATION_E2E=1, NPA_E2E_ISAAC_CADENCE_PYTHON to the existing
    Isaac interpreter/shim, and NPA_E2E_ISAAC_CADENCE_CHECKPOINT to a trained
    Cartpole checkpoint. Run only inside an explicitly authorized RT-core
    runtime; this test neither provisions nor downloads a checkpoint.
    """
    import hashlib
    import os
    import subprocess

    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("requires explicit existing Isaac runtime validation")
    runtime = os.environ.get("NPA_E2E_ISAAC_CADENCE_PYTHON", "")
    checkpoint = os.environ.get("NPA_E2E_ISAAC_CADENCE_CHECKPOINT", "")
    if not runtime or not checkpoint:
        pytest.skip(
            "requires an existing Isaac interpreter and trained Cartpole checkpoint"
        )
    assert Path(runtime).is_file()
    assert Path(checkpoint).is_file()
    raw = tmp_path / "live-raw"
    exporter = tmp_path / "trained_export.py"
    exporter.write_text(
        _build_train_trajectory_export_script(
            "Isaac-Cartpole-v0",
            2,
            12,
            checkpoint,
            str(raw),
            capture_rgb=True,
        )
    )
    observer = tmp_path / "observe_control.py"
    observer.write_text(_LIVE_OBSERVER)
    ticks_file = tmp_path / "source-control-ticks.json"
    with (tmp_path / "isaac-runtime.log").open("w") as log:
        result = subprocess.run(
            [runtime, str(observer), str(exporter), str(ticks_file)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    assert result.returncode == 0, "Inspect the test-owned isaac-runtime.log"
    source = json.loads(ticks_file.read_text())
    meta = json.loads((raw / "meta.json").read_text())
    assert source, "No native Isaac environment control steps were observed"
    assert meta["policy_loaded"] is True
    assert (
        meta["checkpoint_sha256"]
        == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    )
    assert meta["runtime_version"] != "synthetic-runtime"
    assert meta["genuine_simulator_pixels"] is True
    assert len(source) == meta["total_frames"] == meta["rgb_frame_count"]
    times_by_episode = []
    for episode, length in enumerate(meta["episode_lengths"]):
        rows = [row for row in source if row["episode"] == episode]
        assert [row["frame"] for row in rows] == list(range(length))
        assert length > 1, "Need multiple observed control steps to prove cadence"
        first_tick = rows[0]["physics_tick_before"]
        times = []
        for row in rows:
            assert (
                row["physics_tick_after"] - row["physics_tick_before"]
                == row["decimation"]
            )
            assert row["step_dt"] == pytest.approx(
                row["physics_dt"] * row["decimation"]
            )
            assert meta["control_dt"] == pytest.approx(row["step_dt"])
            assert meta["fps"] == pytest.approx(1.0 / row["step_dt"])
            times.append((row["physics_tick_before"] - first_tick) * row["physics_dt"])
        np.testing.assert_allclose(
            np.diff(times), [row["step_dt"] for row in rows[:-1]]
        )
        rgb = np.load(raw / f"episode_{episode:06d}/rgb.npy")
        assert rgb.shape[0] == length
        times_by_episode.append(times)
    _assert_converted_timeline(raw, tmp_path, times_by_episode)


def test_live_observer_on_synthetic_environment(tmp_path):
    """Check profiler plumbing only; the environment here is explicitly synthetic."""
    import subprocess

    observer = tmp_path / "observer.py"
    observer.write_text(_LIVE_OBSERVER)
    script = tmp_path / "synthetic_export.py"
    script.write_text('''
from types import ModuleType, SimpleNamespace
module = ModuleType("isaaclab.envs.synthetic")
exec("""
class Environment:
    def __init__(self):
        self.unwrapped = self
        self._sim_step_counter = 0
        self.physics_dt = 0.005
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(decimation=4, sim=SimpleNamespace(render_interval=8))
    def step(self):
        self._sim_step_counter += 4
        return self._sim_step_counter
""", {"SimpleNamespace": SimpleNamespace, **module.__dict__}, namespace := {})
env = namespace["Environment"]()
for episode_index in range(2):
    env._sim_step_counter = 0
    for step in range(3):
        assert env.step() == (step + 1) * 4
''')
    ticks = tmp_path / "synthetic-ticks.json"
    subprocess.run([sys.executable, str(observer), str(script), str(ticks)], check=True)
    rows = json.loads(ticks.read_text())
    assert [(row["episode"], row["frame"]) for row in rows] == [
        (episode, frame) for episode in range(2) for frame in range(3)
    ]
    assert [row["physics_tick_before"] for row in rows] == [0, 4, 8] * 2
    assert [row["physics_tick_after"] for row in rows] == [4, 8, 12] * 2
    assert all(row["step_dt"] == 0.02 for row in rows)
    assert all(row["physics_dt"] == 0.005 for row in rows)
    assert all(row["decimation"] == 4 and row["render_interval"] == 8 for row in rows)
