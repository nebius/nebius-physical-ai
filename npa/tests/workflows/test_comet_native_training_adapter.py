from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3]
    / "workflows/implementations/behavior-comet12/comet_openpi_runtime.py"
)


def load():
    spec = importlib.util.spec_from_file_location("held_comet_runtime", IMPLEMENTATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_module_import_does_not_import_openpi():
    before = {
        name for name in sys.modules if name == "openpi" or name.startswith("openpi.")
    }
    load()
    after = {
        name for name in sys.modules if name == "openpi" or name.startswith("openpi.")
    }
    assert after == before


def test_direct_step_binds_config_and_reads_step_before_donation(monkeypatch):
    module = load()
    fake_jax = types.SimpleNamespace(device_get=lambda value: value)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    runtime = module.CometOpenPIRuntime.__new__(module.CometOpenPIRuntime)
    runtime.config = types.SimpleNamespace(
        lr_schedule=types.SimpleNamespace(create=lambda: lambda step: 0.5)
    )
    runtime.train_rng = "rng"

    class State:
        def __init__(self, step):
            self.step = step

    calls = []

    def step_fn(rng, state, batch):
        calls.append((rng, state.step, batch))
        state.step = 999
        return State(2), {"loss": 1, "grad_norm": 2, "param_norm": 3}

    runtime.step_fn = step_fn
    state = State(1)
    new, metrics = runtime.train_step(state, "batch")
    assert calls == [("rng", 1, "batch")]
    assert new.step == 2
    assert metrics["learning_rate"] == 0.5


def test_preloaded_openpi_fails_closed(monkeypatch):
    module = load()
    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    with pytest.raises(ValueError, match="before source-overlay"):
        module._require_no_openpi_import()


def _fake_observation_modules(monkeypatch):
    model = types.ModuleType("openpi.models.model")
    model.Observation = types.SimpleNamespace(from_dict=lambda value: value)
    models = types.ModuleType("openpi.models")
    models.model = model
    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setitem(sys.modules, "openpi.models.model", model)


def test_delivered_cursor_crosses_epoch_without_reindexing(monkeypatch):
    module = load()
    monkeypatch.setattr(
        module,
        "native_epoch_indices",
        lambda size, seed, epoch: [2, 0, 3, 1] if epoch == 0 else [1, 3, 0, 2],
    )
    runtime = module.CometOpenPIRuntime.__new__(module.CometOpenPIRuntime)
    runtime.iterator = iter(
        (
            {"_npa_sample_index": [2, 0], "actions": "a"},
            {"_npa_sample_index": [3, 1], "actions": "b"},
            {"_npa_sample_index": [1, 3], "actions": "c"},
        )
    )
    fake_jax = types.SimpleNamespace(
        device_get=lambda value: value,
        device_put=lambda value, _sharding: value,
        block_until_ready=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    _fake_observation_modules(monkeypatch)
    runtime.data_sharding = "gpu-sharding"
    runtime.dataset_size = 4
    runtime.config = types.SimpleNamespace(seed=42, batch_size=2)
    runtime.cursor = module.CommittedCursor(0, 0, 0)
    runtime._pending = None
    delivered = []
    for _ in range(3):
        _batch, row = runtime.next_delivered_batch()
        delivered.append(row["sampler_indices"])
        runtime.commit_cursor(row)
    assert delivered == [[2, 0], [3, 1], [1, 3]]
    assert runtime.cursor == module.CommittedCursor(1, 1, 3)


def _install_initialization_modules(monkeypatch, placed):
    def named(mesh, partition):
        return ("named", mesh, partition)

    fake_jax = types.SimpleNamespace(
        devices=lambda platform=None: ["cpu0"] if platform == "cpu" else ["gpu0"],
        device_get=lambda value: value,
        device_put=lambda value, sharding: placed.append(sharding) or value,
        block_until_ready=lambda value: value,
        sharding=types.SimpleNamespace(
            Mesh=lambda devices, axes: ("mesh", tuple(devices), axes),
            NamedSharding=named,
            PartitionSpec=lambda *axes: ("partition", *axes),
        ),
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    sharding = types.ModuleType("openpi.training.sharding")
    sharding.DATA_AXIS = "data"
    sharding.make_mesh = lambda _devices: "gpu-mesh"
    training = types.ModuleType("openpi.training")
    training.sharding = sharding
    openpi = types.ModuleType("openpi")
    openpi.training = training
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.sharding", sharding)
    _fake_observation_modules(monkeypatch)


def test_native_initialization_stores_sharding_used_by_first_batch(monkeypatch):
    module, placed, loader_shardings = load(), [], []
    _install_initialization_modules(monkeypatch, placed)
    runtime = module.CometOpenPIRuntime.__new__(module.CometOpenPIRuntime)
    runtime.args = types.SimpleNamespace(
        cursor={"epoch": 0, "batch_offset": 0, "committed_global_batch": 0},
        durable_milestones={"0": {"role": "parent"}},
        source_root=Path("source"),
    )
    config = types.SimpleNamespace(fsdp_devices=1, seed=42, batch_size=1)
    monkeypatch.setattr(runtime, "_configure_data", lambda: (config, object(), [0]))

    def loader(_data, sharding):
        loader_shardings.append(sharding)
        return [{"_npa_sample_index": [0], "actions": "action"}]

    monkeypatch.setattr(runtime, "_make_loader", loader)
    monkeypatch.setattr(runtime, "_initialize_state", lambda value: None)
    monkeypatch.setattr(module, "_load_native_train", lambda _root: object())
    monkeypatch.setattr(module, "native_epoch_indices", lambda *_args: [0])
    runtime._initialize_native()
    batch, delivery = runtime.next_delivered_batch()
    assert runtime.data_sharding == ("named", "gpu-mesh", ("partition", "data"))
    assert loader_shardings[0] != runtime.data_sharding and placed == [
        runtime.data_sharding
    ]
    assert batch[1] == "action" and delivery["sampler_indices"] == [0]


def _milestone_runtime(module, tmp_path):
    import argparse

    runtime = module.CometOpenPIRuntime.__new__(module.CometOpenPIRuntime)
    runtime.args = argparse.Namespace(
        durable_milestones={"0": {"role": "parent"}},
        output_prefix="s3://example/run",
        checkpoint_root=tmp_path,
        workflow_inputs={"schema": "inputs"},
    )
    runtime.admission = {
        "identity": {"bytes": 1, "sha256": "a" * 64},
        "static_reconstruction": {"seed": 42},
        "minimum_checkpoint_free_bytes": 0,
    }
    runtime.state = types.SimpleNamespace(step=2)
    runtime.cursor = module.CommittedCursor(0, 2, 2)
    runtime._published, runtime._records, runtime.data_config = {}, {}, object()
    return runtime


def _install_milestone_fakes(module, monkeypatch):
    monkeypatch.setattr(
        module,
        "_state_contract",
        lambda state: {
            "schema": "state",
            "step": state.step,
            "optimizer_state": {"rows": [1]},
        },
    )

    def save(_state, _config, target, manager_step):
        for name, payload in (
            ("params/value", f"state-{manager_step}"),
            ("train_state/optimizer", f"optimizer-{manager_step}"),
        ):
            path = target / str(manager_step) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload.encode())

    monkeypatch.setattr(module, "_save_state", save)
    monkeypatch.setattr(
        module,
        "publish_with_control",
        lambda *_args: {"provider_manifest": {"provider_readback": True}},
    )


def _save_steps(runtime, module, root, steps):
    for step in steps:
        runtime.state.step = step
        runtime.cursor = module.CommittedCursor(0, step, step)
        runtime.save_and_publish(root, step)
        if step != steps[0]:
            runtime.retire_prior(step)


def test_two_adapter_milestones_snapshot_history_without_cycle(tmp_path, monkeypatch):
    import argparse
    import json

    module = load()
    _install_milestone_fakes(module, monkeypatch)
    runtime = _milestone_runtime(module, tmp_path)
    _save_steps(runtime, module, tmp_path, (2, 4))
    json.dumps(runtime.args.durable_milestones)
    assert set(runtime._records[4]["durable_milestones_before"]) == {"0", "2"}
    assert "4" not in runtime._records[4]["durable_milestones_before"]
    resumed = _milestone_runtime(module, tmp_path)
    resumed.args = argparse.Namespace(
        durable_milestones=dict(runtime.args.durable_milestones),
        output_prefix="s3://example/run",
        checkpoint_root=tmp_path,
        workflow_inputs={"schema": "inputs"},
    )
    resumed.state.step, resumed.cursor = 6, module.CommittedCursor(0, 6, 6)
    resumed._records = {
        int(name): {
            key: value for key, value in row.items() if key != "provider_manifest"
        }
        for name, row in resumed.args.durable_milestones.items()
        if int(name) > 0
    }
    resumed.save_and_publish(tmp_path, 6)
    resumed.retire_prior(6)
    assert not (tmp_path / "step-00004").exists()
    assert (tmp_path / "step-00006").is_dir()


def test_public_entrypoint_loader_registers_runtime_for_dataclasses():
    entrypoint = IMPLEMENTATION.with_name("train_comet_native.py")
    spec = importlib.util.spec_from_file_location("held_train_entrypoint", entrypoint)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    module = entry._implementation(IMPLEMENTATION)
    assert module.RuntimePaths.__module__ == "npa_comet_openpi_runtime"
    assert sys.modules["npa_comet_openpi_runtime"] is module
