"""Incremental cadence errors, native tick alignment, and real RRD decoding."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rerun.recording import load_recording

from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
from npa.cli.isaac_lab import _build_train_trajectory_export_script
from npa.viz.adapters.lerobot_to_rerun import lerobot_dataset_logical_to_rerun

from .test_isaac_lab_trajectory_cadence import _install_runtime_stubs


def _export(tmp_path, *, episodes=2, steps=10):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic policy fixture")
    raw = tmp_path / "raw"
    script = _build_train_trajectory_export_script(
        "Isaac-Cartpole-v0", episodes, steps, str(checkpoint), str(raw), capture_rgb=False
    )
    exec(compile(script, "<generated-trained-export>", "exec"), {"__name__": "__main__"})
    return raw


def _assert_converted_timeline(raw, output, times_by_episode):
    """Decode every scalar and, when present, video frame against source ticks."""
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
            [row["timestamp"] for row in selected], times, rtol=0, atol=1e-8
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

    def decoded_entity(entity, times):
        batches = [
            chunk.to_record_batch()
            for chunk in chunks
            if str(chunk.entity_path) == entity and not chunk.is_static
        ]
        assert batches, entity
        table = pa.Table.from_batches(batches)
        decoded_times = table["frame_time"].cast(pa.int64()).to_numpy() / 1e9
        order = np.argsort(decoded_times)
        np.testing.assert_allclose(decoded_times[order], times, rtol=0, atol=1e-8)
        return table, order

    for episode, times in enumerate(times_by_episode):
        root = f"/policy_rollout/episodes/episode_{episode:06d}"
        for kind, filename in (("state", "state.npy"), ("actions", "actions.npy")):
            values = np.load(raw / f"episode_{episode:06d}" / filename)
            for dimension in range(values.shape[1]):
                table, order = decoded_entity(f"{root}/{kind}/dim_{dimension:02d}", times)
                decoded_values = np.asarray(table["Scalars:scalars"].to_pylist()).reshape(-1)
                np.testing.assert_allclose(decoded_values[order], values[:, dimension])
        if meta["rgb_enabled"]:
            table, order = decoded_entity(f"{root}/camera/observation_images_workspace", times)
            video_times = np.asarray(
                table["VideoFrameReference:timestamp"].to_pylist()
            ).reshape(-1) / 1e9
            np.testing.assert_allclose(video_times[order], times, rtol=0, atol=1e-8)
            video_entity = f"videos/episode_{episode:06d}/observation_images_workspace"
            assert table["VideoFrameReference:video_reference"].to_pylist() == [
                [video_entity]
            ] * len(times)
            assets = [
                chunk.to_record_batch()
                for chunk in chunks
                if str(chunk.entity_path) == f"/{video_entity}" and chunk.is_static
            ]
            assert assets, video_entity
            assert any(batch.column("AssetVideo:blob").to_pylist()[0] for batch in assets)


def test_fractional_cadence_preserves_physics_ticks_across_episodes(monkeypatch, tmp_path):
    # Source physics ticks form the oracle, independently of exported metadata.
    physics_dt, decimation = 1.0 / 239.76, 4
    env, app = _install_runtime_stubs(monkeypatch, physics_dt * decimation)
    episode = -1
    tick = 0
    ticks_by_episode, actions_by_episode = [], []

    def reset():
        nonlocal episode, tick
        episode += 1
        tick = 0
        ticks_by_episode.append([])
        actions_by_episode.append([])
        return [[tick, episode]], {}

    def step(actions):
        nonlocal tick
        ticks_by_episode[-1].append(tick)
        actions_by_episode[-1].append(actions[0])
        tick += decimation
        done = len(ticks_by_episode[-1]) == 7
        return [[tick, episode]], [1.0], [done], {}

    monkeypatch.setattr(env, "reset", reset)
    monkeypatch.setattr(env, "step", step)
    monkeypatch.setattr(
        sys.modules["rsl_rl.runners"].OnPolicyRunner,
        "get_inference_policy",
        lambda self, **kwargs: lambda obs: [[float(obs[0][0]) + 100.0]],
    )
    raw = _export(tmp_path)
    assert env.closed and app.closed
    assert ticks_by_episode == [list(range(0, 28, 4))] * 2
    meta = json.loads((raw / "meta.json").read_text())
    assert meta["episode_lengths"] == [7, 7]  # Early termination before the ten-step cap.
    assert meta["rgb_enabled"] is False and meta["genuine_simulator_pixels"] is False
    for episode, ticks in enumerate(ticks_by_episode):
        states = np.load(raw / f"episode_{episode:06d}/state.npy")
        np.testing.assert_array_equal(states[:, 0], ticks)
        np.testing.assert_array_equal(states[:, 1], [episode] * len(ticks))
        np.testing.assert_array_equal(
            np.load(raw / f"episode_{episode:06d}/actions.npy"), actions_by_episode[episode]
        )
    times = [np.asarray(ticks) * physics_dt for ticks in ticks_by_episode]
    _assert_converted_timeline(raw, tmp_path, times)
    assert meta["fps"] == pytest.approx(59.94)
    assert meta["control_dt"] == pytest.approx(physics_dt * decimation)


@pytest.mark.parametrize(
    ("step_dt", "cause"),
    [(None, TypeError), ("bad", ValueError), (10**400, OverflowError), ("missing", AttributeError)],
    ids=["none", "nonnumeric", "float-overflow", "missing"],
)
def test_unavailable_cadence_closes_runtime_before_rollout(monkeypatch, tmp_path, step_dt, cause):
    env, app = _install_runtime_stubs(monkeypatch, step_dt)
    if step_dt == "missing":
        del env.step_dt
    loads = []
    monkeypatch.setattr(
        sys.modules["rsl_rl.runners"].OnPolicyRunner, "load", lambda *args: loads.append(args)
    )
    with pytest.raises(RuntimeError, match="control timestep must be finite and positive") as error:
        _export(tmp_path, episodes=1, steps=1)
    assert isinstance(error.value.__cause__, cause)
    assert env.closed and app.closed
    assert env.index == 0 and not loads
    assert not (tmp_path / "raw/meta.json").exists()
    assert not list((tmp_path / "raw").glob("episode_*"))


def test_unrepresentable_frame_rate_closes_runtime_before_rollout(monkeypatch, tmp_path):
    env, app = _install_runtime_stubs(monkeypatch, 5e-324)
    loads = []
    monkeypatch.setattr(
        sys.modules["rsl_rl.runners"].OnPolicyRunner, "load", lambda *args: loads.append(args)
    )
    with pytest.raises(RuntimeError, match="control timestep must yield a finite frame rate"):
        _export(tmp_path, episodes=1, steps=1)
    assert env.closed and app.closed
    assert env.index == 0 and not loads
    assert not (tmp_path / "raw/meta.json").exists()
    assert not list((tmp_path / "raw").glob("episode_*"))


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
