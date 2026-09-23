from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import pytest


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


def _assert_full_recipe(module, training_config, tmp_path):
    recipe = module._explicit_config(
        training_config.get_config("pi05_b1k-base"), tmp_path / "released-parent"
    )
    contract = module._config_contract(recipe)
    assert recipe.model.pi05 is True and recipe.model.action_horizon == 32
    assert module._data_factory(recipe) is not None
    assert contract == {**contract, "name": "comet_native_full_parameter_adamw_20k"}
    assert (contract["batch_size"], contract["num_workers"]) == (256, 8)
    assert (contract["num_train_steps"], contract["seed"]) == (20_000, 42)
    assert contract["freeze_filter"] == "full_parameter_nnx.Nothing"


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
