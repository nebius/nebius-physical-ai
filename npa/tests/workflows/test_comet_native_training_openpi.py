from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SPAWN_DATASET_PROBE = r"""
import importlib.util
import sys
from pathlib import Path

import torch


class FailingSampleDataset:
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        raise RuntimeError("deliberate sample failure")


def main():
    adapter = Path(sys.argv[1])
    name = sys.argv[2]
    spec = importlib.util.spec_from_file_location(name, adapter)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    source = [{"value": torch.tensor([3])}, {"value": torch.tensor([7])}]
    dataset = module.DeliveredSampleDataset(source)
    assert type(dataset).__module__ == "npa.workflows.behavior_challenge.comet_training_data"
    assert module.NativeCursorSampler.__module__ == "npa.workflows.behavior_challenge.comet_training_sampler"
    probe = module.verify_spawn_dataset_sample(dataset)
    assert probe == {"start_method": "spawn", "status": "sample_read", "sample_index": 0}
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn"
    )
    batch = next(iter(loader))
    assert batch["_npa_sample_index"].tolist() == [0, 1]
    assert batch["value"].reshape(-1).tolist() == [3, 7]
    failed = module.DeliveredSampleDataset(FailingSampleDataset())
    try:
        module.verify_spawn_dataset_sample(failed)
    except ValueError as exc:
        assert str(exc) == "spawn dataset sample failed: RuntimeError"
    else:
        raise AssertionError("spawn sample failure was accepted")


if __name__ == "__main__":
    main()
"""


def _native_modules(tmp_path):
    from npa.workflows.behavior_challenge.comet_policy import (
        build_training_source_overlay,
    )

    source = Path(os.environ["NPA_COMET_SOURCE_ROOT"])
    overlay = build_training_source_overlay(source, tmp_path / "overlay")
    sys.path[:0] = [str(overlay), str(source / "packages/openpi-client/src")]
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.models import model as model_module
    from openpi.training import config as training_config
    from openpi.training import data_loader
    from openpi.training.utils import TrainState

    implementation = (
        Path(__file__).parents[3]
        / "workflows/implementations/behavior-comet12/comet_openpi_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("held_real_orbax", implementation)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return (
        source,
        module,
        jax,
        jnp,
        optax,
        nnx,
        model_module,
        training_config,
        data_loader,
        TrainState,
    )


@pytest.mark.parametrize(
    "alias", ["npa_public_comet_adapter_qualification", "npa_comet_openpi_runtime"]
)
def test_dynamic_adapter_dataset_is_importable_in_spawn_worker(tmp_path, alias):
    pytest.importorskip("torch")
    implementation = (
        Path(__file__).parents[3]
        / "workflows/implementations/behavior-comet12/comet_openpi_runtime.py"
    )
    script = tmp_path / "spawn_dataset_probe.py"
    script.write_text(_SPAWN_DATASET_PROBE)
    result = subprocess.run(
        [sys.executable, str(script), str(implementation), alias],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _data_contract(task_id=1):
    names = {0: "turning_on_radio", 1: "picking_up_trash", 22: "putting_shoes_on_rack"}
    return {
        "schema": "npa.behavior.comet-native-data-reconstruction.v1",
        "dataset_repository": "behavior-1k/2026-challenge-demos",
        "dataset_revision": "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
        "task_id": task_id,
        "task_name": names[task_id],
        "modalities": ["rgb"],
        "tolerance_s": 5e-4,
        "prompt_from_task": True,
        "fine_grained_level": 0,
    }


def _data_factory(module, training_config, tmp_path, task_id, episodes):
    contract = module.validate_data_reconstruction(_data_contract(task_id))
    return module._explicit_data_factory(
        training_config,
        SimpleNamespace(dataset_root=tmp_path / "dataset"),
        episodes,
        contract,
    )


def _assert_full_recipe(module, training_config, tmp_path):
    data_contract = _data_contract()
    factory = _data_factory(module, training_config, tmp_path, 1, (200, 201))
    recipe = module._explicit_config(
        training_config.get_config("pi05_b1k-base"),
        tmp_path / "released-parent",
        factory,
    )
    recipe_contract = module._config_contract(recipe)
    assert recipe.model.pi05 is True and recipe.model.action_horizon == 32
    assert recipe.data.repo_id == data_contract["dataset_repository"]
    assert recipe.data.base_config.tasks == ["picking_up_trash"]
    assert recipe.data.base_config.episodes_index == [200, 201]
    assert recipe.data.base_config.tolerance_s == 5e-4
    assert recipe_contract == {
        **recipe_contract,
        "name": "comet_native_full_parameter_adamw_20k",
    }
    assert (recipe_contract["batch_size"], recipe_contract["num_workers"]) == (256, 8)
    assert (recipe_contract["num_train_steps"], recipe_contract["seed"]) == (
        20_000,
        42,
    )
    assert recipe_contract["freeze_filter"] == "full_parameter_nnx.Nothing"

    radio = _data_factory(module, training_config, tmp_path, 0, (0,))
    assert radio.base_config.tasks == ["turning_on_radio"]
    with pytest.raises(ValueError, match="base config name is unknown"):
        module._native_config(
            "not-a-real-openpi-config", tmp_path / "parent", radio, {}
        )


def _train_entrypoint():
    implementation = (
        Path(__file__).parents[3]
        / "workflows/implementations/behavior-comet12/train_comet_native.py"
    )
    spec = importlib.util.spec_from_file_location(
        "held_train_entrypoint", implementation
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _admission_value(task_id=1):
    return {
        "schema": "npa.behavior.comet-native-training-admission.v1",
        "status": "qualified_inputs_bound_for_native_training",
        "source_files": {"scripts/train.py": {"bytes": 1, "sha256": "a" * 64}},
        "minimum_materialization_free_bytes": 1,
        "minimum_checkpoint_free_bytes": 1,
        "data_reconstruction": _data_contract(task_id),
    }


def _write_admission(module, path, value):
    path.write_text(json.dumps(value) + "\n")
    return module._admission(path, module.file_identity(path)["sha256"])


@pytest.mark.parametrize("task_id", [0, 1])
def test_admission_accepts_supported_task_contracts(tmp_path, task_id):
    module = _train_entrypoint()
    result = _write_admission(
        module, tmp_path / "admission.json", _admission_value(task_id)
    )
    assert result["data_reconstruction"]["task_id"] == task_id


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("dataset_revision", "0" * 40),
        ("modalities", ["depth"]),
        ("tolerance_s", 1e-4),
    ],
)
def test_admission_rejects_fixed_data_contract_drift(tmp_path, field, wrong):
    module = _train_entrypoint()
    value = _admission_value()
    value["data_reconstruction"][field] = wrong
    with pytest.raises(ValueError, match="data reconstruction contract differs"):
        _write_admission(module, tmp_path / "admission.json", value)


def test_admission_rejects_mismatched_task_identity(tmp_path):
    module = _train_entrypoint()
    value = _admission_value()
    value["data_reconstruction"]["task_name"] = "turning_on_radio"
    with pytest.raises(ValueError, match="task identity differs"):
        _write_admission(module, tmp_path / "admission.json", value)


def _tiny_state(jnp, optax, nnx, model_module, TrainState):
    class TinyModel(model_module.BaseModel):
        def __init__(self):
            super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
            self.weight = nnx.Param(jnp.asarray([0.75], dtype=jnp.float32))

        def compute_loss(self, rng, observation, actions, *, train=False):
            return jnp.square(
                self.weight.value * observation.state[..., :1] - actions[..., 0]
            )

        def sample_actions(self, rng, observation, **kwargs):
            return jnp.zeros((1, 1, 1), dtype=jnp.float32)

    graph, params = nnx.split(TinyModel())
    tx = optax.adamw(1e-3)
    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=graph,
        opt_state=tx.init(params.filter(nnx.Param)),
        tx=tx,
        ema_decay=None,
        ema_params=None,
    )


def _loader_batch(module, jax, jnp, data_loader):
    class OneRow:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "image": {},
                "image_mask": {},
                "state": jnp.asarray([2.0], dtype=jnp.float32),
                "actions": jnp.asarray([[0.25]], dtype=jnp.float32),
                "_npa_sample_index": jnp.asarray(0, dtype=jnp.int32),
            }

    mesh = jax.sharding.Mesh(jax.devices("cpu"), ("B",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("B"))
    loader = data_loader.TorchDataLoader(
        OneRow(),
        local_batch_size=1,
        sharding=sharding,
        shuffle=False,
        num_workers=0,
        seed=42,
    )
    batch, placement = module.make_native_batch(next(iter(loader)), sharding)
    assert batch[0].state.shape == (1, 1) and batch[1].shape == (1, 1, 1)
    assert (
        placement["placement"] == "explicit_cpu_normalization_then_gpu_batch_placement"
    )


def _assert_roundtrip(module, native, config, state, batch, rng, tmp_path, jax, jnp):
    @dataclasses.dataclass
    class DataConfig:
        norm_stats: None = None
        asset_id: None = None

    state1, _ = native.train_step(config, rng, state, batch)
    state2, _ = native.train_step(config, rng, state1, batch)
    checkpoint = tmp_path / "checkpoint"
    module._save_state(state2, DataConfig(), checkpoint, 1)
    restored = module._restore_state(state2, DataConfig(), checkpoint, 1)
    uninterrupted3, _ = native.train_step(config, rng, state2, batch)
    restored3, _ = native.train_step(config, rng, restored, batch)
    assert module._state_contract(restored3) == module._state_contract(uninterrupted3)
    corrupted = dataclasses.replace(
        restored3,
        params=jax.tree.map(
            lambda value: value + jnp.asarray(1, dtype=value.dtype), restored3.params
        ),
    )
    assert module._state_contract(corrupted) != module._state_contract(restored3)
    with pytest.raises(ValueError, match="inventory|AdamW"):
        module._state_contract(dataclasses.replace(restored3, opt_state=()))
    assert module._state_contract(restored)["params"]["row_count"] > 0
    assert int(restored.step) == 2 and jax.tree_util.tree_structure(
        restored
    ) == jax.tree_util.tree_structure(state)


def test_actual_openpi_orbax_state_roundtrip(tmp_path):
    if os.environ.get("NPA_REQUIRE_OPENPI_TINY") != "1":
        pytest.skip("pinned OpenPI runtime required")
    (
        source,
        module,
        jax,
        jnp,
        optax,
        nnx,
        model_module,
        training_config,
        data_loader,
        TrainState,
    ) = _native_modules(tmp_path)
    _assert_full_recipe(module, training_config, tmp_path)
    state = _tiny_state(jnp, optax, nnx, model_module, TrainState)
    native = module._load_native_train(source)
    config = training_config.TrainConfig(
        name="tiny-native-proof",
        exp_name="tiny-native-proof",
        freeze_filter=nnx.Nothing,
        ema_decay=None,
    )
    observation = model_module.Observation(
        images={}, image_masks={}, state=jnp.asarray([[2.0]], dtype=jnp.float32)
    )
    batch = (observation, jnp.asarray([[[0.25]]], dtype=jnp.float32))
    _loader_batch(module, jax, jnp, data_loader)
    _assert_roundtrip(
        module, native, config, state, batch, jax.random.key(42), tmp_path, jax, jnp
    )
