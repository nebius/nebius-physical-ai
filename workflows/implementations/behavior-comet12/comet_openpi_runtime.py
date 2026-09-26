"""Adapt the verified Comet dataset to native OpenPI training and Orbax state."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from npa.workflows.behavior_challenge.comet_training_data import (
    CometTaskDataset,
    DeliveredSampleDataset,
    validate_data_reconstruction,
    verify_spawn_dataset_sample,
)
from npa.workflows.behavior_challenge.comet_training_sampler import (
    CommittedCursor,
    NativeCursorSampler,
    native_epoch_indices,
)
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    atomic_json,
    checkpoint_inventory,
    serving_params,
    validate_complete_checkpoint,
)
from npa.workflows.behavior_challenge.native_training_control import (
    publish_with_control,
)


@dataclasses.dataclass(frozen=True)
class RuntimePaths:
    """Bind scientific input paths.

    Args: source_root: OpenPI source. dataset_root: Dataset view. split: Split
        receipt. parent_checkpoint: Released parent. workspace: Owned workspace.
    Returns: Immutable path binding. Raises: None.
    """

    source_root: Path
    dataset_root: Path
    split: Path
    parent_checkpoint: Path
    workspace: Path


def _load_native_train(source_root: Path) -> Any:
    script = source_root / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("npa_comet_native_train", script)
    if spec is None or spec.loader is None:
        raise ValueError("native OpenPI train module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_no_openpi_import() -> None:
    if any(name == "openpi" or name.startswith("openpi.") for name in sys.modules):
        raise ValueError("OpenPI was imported before source-overlay verification")


def _explicit_data_factory(
    training_config: Any, args: Any, episodes: tuple[int, ...], contract: dict[str, Any]
) -> Any:
    return training_config.LeRobotB1KDataConfig(
        repo_id=contract["dataset_repository"],
        base_config=training_config.DataConfig(
            prompt_from_task=True,
            behavior_dataset_root=str(args.dataset_root),
            episodes_index=list(episodes),
            tasks=[contract["task_name"]],
            modalities=list(contract["modalities"]),
            tolerance_s=contract["tolerance_s"],
            fine_grained_level=contract["fine_grained_level"],
        ),
    )


def _explicit_config(base: Any, parent: Path, data_factory: Any) -> Any:
    from flax import nnx
    from openpi.training import optimizer as optimizer_config
    from openpi.training import weight_loaders

    return dataclasses.replace(
        base,
        name="comet_native_full_parameter_adamw_20k",
        data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(parent / "params")),
        lr_schedule=optimizer_config.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=2.5e-6, decay_steps=20_000, decay_lr=0.0
        ),
        optimizer=optimizer_config.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        freeze_filter=nnx.Nothing,
        ema_decay=None,
        batch_size=256,
        num_workers=8,
        num_train_steps=20_000,
        seed=42,
        save_interval=5_000,
        keep_period=5_000,
        fsdp_devices=1,
        wandb_enabled=False,
    )


def _config_contract(config: Any) -> dict[str, Any]:
    return {
        "name": config.name,
        "model": dataclasses.asdict(config.model),
        "optimizer": dataclasses.asdict(config.optimizer),
        "lr_schedule": dataclasses.asdict(config.lr_schedule),
        "ema_decay": config.ema_decay,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "num_train_steps": config.num_train_steps,
        "seed": config.seed,
        "save_interval": config.save_interval,
        "keep_period": config.keep_period,
        "fsdp_devices": config.fsdp_devices,
        "freeze_filter": "full_parameter_nnx.Nothing",
    }


def _native_config(
    config_name: str, parent: Path, data_factory: Any, expected: dict[str, Any]
) -> Any:
    from openpi.training import config as training_config

    configs = getattr(training_config, "_CONFIGS_DICT", {})
    if config_name not in configs:
        raise ValueError("native OpenPI base config name is unknown")
    config = _explicit_config(configs[config_name], parent, data_factory)
    observed = _config_contract(config)
    if observed != expected:
        raise ValueError("native OpenPI static reconstruction differs")
    return config


def _configured_data(args: Any, admission: dict[str, Any]) -> tuple[Any, Any, Any]:
    from openpi.training import config as training_config

    contract = validate_data_reconstruction(admission.get("data_reconstruction"))
    dataset = CometTaskDataset(
        args.dataset_root,
        args.split,
        task_id=contract["task_id"],
        expected_split_sha256=args.split_sha256,
        partition="training",
    )
    factory = _explicit_data_factory(training_config, args, dataset.episodes, contract)
    config = _native_config(
        args.config_name,
        args.parent_checkpoint,
        factory,
        admission["static_reconstruction"],
    )
    assets = training_config.AssetsConfig(
        assets_dir=str(args.parent_checkpoint / "assets"),
        asset_id=admission["data_asset_id"],
    )
    data_config = dataclasses.replace(factory, assets=assets).create(
        config.assets_dirs, config.model
    )
    return config, data_config, dataset


def _save_state(state: Any, data_config: Any, target: Path, manager_step: int) -> None:
    from openpi.training import checkpoints

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        target, keep_period=None, overwrite=True, resume=False
    )
    if resuming:
        raise ValueError("new milestone entered resume mode")

    class DataConfig:
        """Expose native data config. Args: None. Returns: Adapter. Raises: None."""

        def data_config(self) -> Any:
            """Return config. Args: None. Returns: Data config. Raises: None."""
            return data_config

    checkpoints.save_state(manager, state, DataConfig(), manager_step)
    manager.wait_until_finished()
    manager.close()


def _restore_state(
    template: Any, data_config: Any, target: Path, manager_step: int
) -> Any:
    import jax
    from openpi.training import checkpoints

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        target, keep_period=None, overwrite=False, resume=True
    )
    if not resuming or tuple(manager.all_steps()) != (manager_step,):
        raise ValueError("native Orbax manager step differs")

    class DataConfig:
        """Expose native data config. Args: None. Returns: Adapter. Raises: None."""

        def data_config(self) -> Any:
            """Return config. Args: None. Returns: Data config. Raises: None."""
            return data_config

    state = checkpoints.restore_state(manager, template, DataConfig(), manager_step)
    jax.block_until_ready(state)
    manager.close()
    return state


def _array_record(value: Any) -> dict[str, Any]:
    import jax
    import numpy as np

    array = np.asarray(jax.device_get(value))
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError("native state contains a nonnumeric or nonfinite leaf")
    raw = array.tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _array_inventory(tree: Any) -> dict[str, Any]:
    import jax

    rows = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError("native state contains a non-array leaf")
        name = jax.tree_util.keystr(path)
        if name in rows:
            raise ValueError("native state array path is duplicated")
        rows[name] = _array_record(value)
    if not rows:
        raise ValueError("native state array inventory differs")
    encoded = json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    return {
        "leaves": rows,
        "row_count": len(rows),
        "element_count": sum(row["elements"] for row in rows.values()),
        "total_bytes": sum(row["bytes"] for row in rows.values()),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _adam_pairs(value: Any) -> list[tuple[Any, Any]]:
    if hasattr(value, "mu") and hasattr(value, "nu"):
        return [(value.mu, value.nu)]
    if dataclasses.is_dataclass(value):
        return [
            pair
            for field in dataclasses.fields(value)
            for pair in _adam_pairs(getattr(value, field.name))
        ]
    if isinstance(value, (tuple, list)):
        return [pair for item in value for pair in _adam_pairs(item)]
    return []


def _optimizer_counts(value: Any, step: int) -> dict[str, int]:
    import jax
    import numpy as np

    rows = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        array = np.asarray(jax.device_get(leaf))
        if array.shape == () and np.issubdtype(array.dtype, np.integer):
            rows[jax.tree_util.keystr(path)] = int(array)
    if not rows or any(count != step for count in rows.values()):
        raise ValueError("AdamW/schedule scalar progress differs from TrainState step")
    return rows


def _state_contract(state: Any) -> dict[str, Any]:
    params = _array_inventory(state.params)
    optimizer = _array_inventory(state.opt_state)
    if any(row["dtype"] != "float32" for row in params["leaves"].values()):
        raise ValueError("full native TrainState parameters must remain FP32")
    pairs = _adam_pairs(state.opt_state)
    if len(pairs) != 1:
        raise ValueError("complete AdamW mu/nu state is absent or ambiguous")
    mu, nu = (_array_inventory(value) for value in pairs[0])
    signature = lambda inv: sorted(
        (row["shape"], row["dtype"], row["elements"]) for row in inv["leaves"].values()
    )
    if signature(mu) != signature(params) or signature(nu) != signature(params):
        raise ValueError("AdamW moments do not cover every parameter leaf")
    return {
        "schema": "npa.behavior.comet-native-full-train-state.v1",
        "step": int(state.step),
        "params": params,
        "optimizer_state": optimizer,
        "adamw_mu": mu,
        "adamw_nu": nu,
        "optimizer_scalar_progress": _optimizer_counts(
            state.opt_state, int(state.step)
        ),
        "ema_decay": state.ema_decay,
        "ema_params_present": state.ema_params is not None,
    }


def make_native_batch(raw: dict[str, Any], data_sharding: Any) -> tuple[Any, dict]:
    """Normalize on host CPU, then place the complete native batch explicitly.

    Args: raw: CPU-sharded transformed loader batch. data_sharding: GPU sharding.
    Returns: Native Observation/actions and the placement contract.
    Raises: Propagates malformed Observation and JAX placement errors.
    """
    import jax
    from openpi.models import model as model_module

    host = jax.device_get(raw)
    payload = {
        name: value for name, value in host.items() if name != "_npa_sample_index"
    }
    host_batch = (model_module.Observation.from_dict(payload), payload["actions"])
    batch = jax.device_put(host_batch, data_sharding)
    jax.block_until_ready(batch)
    return batch, {
        "placement": "explicit_cpu_normalization_then_gpu_batch_placement",
        "identity_side_channel_excluded": True,
    }


class CometOpenPIRuntime:
    """Own the real OpenPI update and state lifecycle.

    Args: args: Bound scientific arguments. admission: Qualified admission.
    Returns: Initialized concrete native runtime.
    Raises: ValueError when source, data, recipe, state, or cursor differs.
    """

    def __init__(self, args: Any, admission: dict[str, Any]) -> None:
        _require_no_openpi_import()
        self.args = args
        self.admission = admission
        self._verify_and_install_overlay()
        self._initialize_native()

    def _verify_and_install_overlay(self) -> None:
        from npa.workflows.behavior_challenge.comet_policy import (
            build_training_source_overlay,
        )

        overlay = build_training_source_overlay(
            self.args.source_root, self.args.workspace / "source-overlay"
        )
        source = (self.args.source_root / "src").resolve()
        sys.path[:0] = [str(overlay), str(source)]
        expected = self.admission.get("source_files", {})
        for relative, identity in expected.items():
            path = self.args.source_root / relative
            from npa.workflows.behavior_challenge.native_training_checkpoint import (
                file_identity,
            )

            if file_identity(path) != identity:
                raise ValueError(f"OpenPI source identity differs: {relative}")
        if not expected:
            raise ValueError("OpenPI source identities are absent")
        self.overlay = overlay

    def _initialize_native(self) -> None:
        import jax
        from openpi.training import sharding

        config, data_config, transformed = self._configure_data()
        self.dataset_size = len(transformed)
        self.config = config
        self.data_config = data_config
        self.cursor = CommittedCursor(**self.args.cursor)
        mesh = sharding.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        cpu = jax.devices("cpu")
        if len(cpu) != 1:
            raise ValueError("native transformation requires exactly one CPU device")
        transform_mesh = jax.sharding.Mesh(cpu, ("B",))
        transform_sharding = jax.sharding.NamedSharding(
            transform_mesh, jax.sharding.PartitionSpec("B")
        )
        self.iterator = iter(self._make_loader(transformed, transform_sharding))
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.native = _load_native_train(self.args.source_root)
        self._initialize_state(data_sharding)
        self._published: dict[int, dict[str, Any]] = {}
        self._records = {
            int(name): {
                key: value for key, value in row.items() if key != "provider_manifest"
            }
            for name, row in self.args.durable_milestones.items()
            if int(name) > 0
        }

    def _configure_data(self) -> tuple[Any, Any, Any]:
        from openpi.training import data_loader

        config, data_config, dataset = _configured_data(self.args, self.admission)
        transformed = DeliveredSampleDataset(
            data_loader.transform_dataset(dataset, data_config)
        )
        return config, data_config, transformed

    def _make_loader(self, transformed: Any, data_sharding: Any) -> Any:
        from openpi.training import data_loader

        sampler = NativeCursorSampler(
            transformed,
            seed=self.config.seed,
            batch_size=self.config.batch_size,
            cursor=self.cursor,
        )
        return data_loader.TorchDataLoader(
            transformed,
            local_batch_size=self.config.batch_size,
            sharding=data_sharding,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
        )

    def _initialize_state(self, data_sharding: Any) -> None:
        import jax

        rng = jax.random.key(self.config.seed)
        self.train_rng, init_rng = jax.random.split(rng)
        template, state_sharding = self.native.init_train_state(
            self.config,
            init_rng,
            self.mesh,
            resume=self.args.resume_checkpoint is not None,
        )
        if self.args.resume_checkpoint is None:
            self.state = template
            if int(template.step) != 0:
                raise ValueError("released parent TrainState step differs")
            _state_contract(template)
        else:
            self.state = self._resume_state(template)
        replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        self.step_fn = jax.jit(
            functools.partial(self.native.train_step, self.config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        self._pending: dict[str, Any] | None = None

    def _resume_state(self, template: Any) -> Any:
        state = _restore_state(
            template,
            self.data_config,
            self.args.resume_checkpoint,
            self.cursor.committed_global_batch - 1,
        )
        receipt = self.args.resume_receipt
        expected = {
            "logical_update_count": self.cursor.committed_global_batch,
            "manager_step": self.cursor.committed_global_batch - 1,
            "cursor": dataclasses.asdict(self.cursor),
            "static_reconstruction": self.admission["static_reconstruction"],
            "admission": self.admission["identity"],
            "workflow_inputs": self.args.workflow_inputs,
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise ValueError("durable resume admission/static/cursor differs")
        if _state_contract(state) != receipt.get("train_state"):
            raise ValueError("restored complete FP32 TrainState/AdamW differs")
        return state

    def next_delivered_batch(self) -> tuple[Any, dict[str, Any]]:
        """Return the next exact native batch.

        Args: None. Returns: Native batch and delivered sampler identity.
        Raises: ValueError when the delivered order differs.
        """
        import jax

        raw = next(self.iterator)
        indices = [int(value) for value in jax.device_get(raw["_npa_sample_index"])]
        start = self.cursor.batch_offset * self.config.batch_size
        expected = native_epoch_indices(
            self.dataset_size, self.config.seed, self.cursor.epoch
        )[start : start + self.config.batch_size]
        if indices != expected:
            raise ValueError("delivered sampler identities differ")
        delivery = {
            "sampler_indices": indices,
            "identity_source": "delivered_post_transform_side_channel",
        }
        self._pending = delivery
        batch, placement = make_native_batch(raw, self.data_sharding)
        delivery.update(placement)
        return batch, delivery

    def train_step(self, state: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        """Execute one direct native update.

        Args: state: Current TrainState. batch: Native Observation/actions.
        Returns: Updated state and finite metrics. Raises: ValueError on drift.
        """
        import jax

        prior_step = int(state.step)
        new_state, info = self.step_fn(self.train_rng, state, batch)
        metrics = {name: float(jax.device_get(value)) for name, value in info.items()}
        metrics["learning_rate"] = float(
            jax.device_get(self.config.lr_schedule.create()(prior_step))
        )
        if int(new_state.step) != prior_step + 1 or not all(
            math.isfinite(value) for value in metrics.values()
        ):
            raise ValueError("native update chronology or metrics differ")
        return new_state, metrics

    def synchronize(self, value: Any) -> None:
        """Synchronize device work. Args: value: Device tree. Returns: None.

        Raises: Propagates JAX synchronization errors.
        """
        import jax

        jax.block_until_ready(value)

    def commit_cursor(self, delivery: dict[str, Any]) -> None:
        """Commit one completed delivery.

        Args: delivery: Exact pending record. Returns: None.
        Raises: ValueError when delivery is not pending.
        """
        if delivery is not self._pending:
            raise ValueError("delivered batch is not pending")
        self.cursor = self.cursor.advance_after_update(
            dataset_size=self.dataset_size,
            batch_size=self.config.batch_size,
            update_completed=True,
        )
        self._pending = None

    def state_step(self) -> int:
        """Return TrainState step. Args: None. Returns: Step. Raises: None."""
        return int(self.state.step)

    def cursor_step(self) -> int:
        """Return cursor step. Args: None. Returns: Step. Raises: None."""
        return self.cursor.committed_global_batch

    def released_parent_milestone(self) -> dict[str, Any]:
        """Return parent record. Args: None. Returns: Record. Raises: None."""
        return self.admission["released_parent"]

    def durable_milestones(self) -> dict[str, Any]:
        """Return durable history. Args: None. Returns: History. Raises: None."""
        return dict(self.args.durable_milestones)

    def runtime_record(self) -> dict[str, Any]:
        """Return provenance. Args: None. Returns: Runtime record. Raises: None."""
        return {
            "engine": "upstream_openpi_direct_jit",
            "config_name": self.config.name,
            "overlay": "ephemeral_verified_source_overlay",
            "admission": self.admission["identity"],
            "workflow_inputs": self.args.workflow_inputs,
            "durable_milestones_before": dict(self.args.durable_milestones),
        }

    def save_and_publish(self, root: Path, step: int) -> dict[str, Any]:
        """Save and publish one complete milestone.

        Args: root: Durable checkpoint root. step: Logical update.
        Returns: Bound provider result. Raises: ValueError on contract drift.
        """
        target = root / f"step-{step:05d}"
        if target.exists() or target.is_symlink():
            raise ValueError("milestone target must be new")
        if (
            shutil.disk_usage(root).free
            <= self.admission["minimum_checkpoint_free_bytes"]
        ):
            raise ValueError("checkpoint free-space gate failed")
        manager_step = step - 1
        _save_state(self.state, self.data_config, target, manager_step)
        inventory = checkpoint_inventory(target)
        validate_complete_checkpoint(inventory, manager_step)
        record = self._milestone_record(step, manager_step, inventory)
        receipt = root / f"step-{step:05d}.json"
        atomic_json(receipt, record)
        result = publish_with_control(
            target,
            receipt,
            f"{self.args.output_prefix}/milestones/step-{step:05d}",
            root,
        )
        self._published[step] = result
        self._records[step] = record
        self.args.durable_milestones[str(step)] = {
            **record,
            "provider_manifest": result["provider_manifest"],
        }
        return result

    def _milestone_record(
        self, step: int, manager_step: int, inventory: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "schema": "npa.behavior.comet-native-training-receipt.v1",
            "logical_update_count": step,
            "manager_step": manager_step,
            "train_state_step": self.state_step(),
            "train_state": _state_contract(self.state),
            "full_state_resume_ready": True,
            "cursor": dataclasses.asdict(self.cursor),
            "checkpoint": inventory,
            "serving_params": serving_params(inventory, manager_step),
            "static_reconstruction": self.admission["static_reconstruction"],
            "admission": self.admission["identity"],
            "workflow_inputs": self.args.workflow_inputs,
            "durable_milestones_before": dict(self.args.durable_milestones),
        }

    def retire_prior(self, step: int) -> dict[str, Any] | None:
        """Retire prior local state after provider durability.

        Args: step: Current durable step. Returns: Retirement record or None.
        Raises: ValueError when prior or current evidence differs.
        """
        prior = max((value for value in self._records if value < step), default=None)
        if prior is None:
            return None
        if (
            self._published[step].get("provider_manifest", {}).get("provider_readback")
            is not True
        ):
            raise ValueError("current milestone is not provider durable")
        root = self.args.checkpoint_root / f"step-{prior:05d}"
        expected = self._records[prior]["checkpoint"]
        if checkpoint_inventory(root) != expected:
            raise ValueError("prior local checkpoint changed before retirement")
        receipt = self.args.checkpoint_root / f"step-{prior:05d}.json"
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("prior local milestone receipt differs")
        if json.loads(receipt.read_text()) != self._records[prior]:
            raise ValueError("prior local milestone receipt contract differs")
        shutil.rmtree(root)
        if root.exists() or root.is_symlink():
            raise ValueError("prior local checkpoint retirement failed")
        receipt.unlink()
        return {
            "status": "prior_local_milestone_retired_after_next_durable_readback",
            "logical_update": prior,
            "provider_state_retained": True,
        }


def _input_preflight_receipt(args: Any, admission: dict[str, Any]) -> dict[str, Any]:
    import jax
    from openpi.training import data_loader

    config, data_config, dataset = _configured_data(args, admission)
    transformed = DeliveredSampleDataset(
        data_loader.transform_dataset(dataset, data_config)
    )
    worker_probe = verify_spawn_dataset_sample(transformed)
    if jax.default_backend() != "cpu" or any(
        device.platform != "cpu" for device in jax.devices()
    ):
        raise ValueError("input-only verification requires CPU JAX devices")
    return {
        "schema": "npa.behavior.comet-native-input-preflight.v1",
        "status": "real_overlay_config_dataset_verified_before_policy_initialization",
        "config_name": config.name,
        "dataset_size": len(dataset),
        "data_reconstruction": admission["data_reconstruction"],
        "data_asset_id": data_config.asset_id,
        "source_overlay": "ephemeral_verified_source_overlay",
        "admission": admission["identity"],
        "workflow_inputs": args.workflow_inputs,
        "full_policy_initialized": False,
        "optimizer_updates": 0,
        "jax_backend": "cpu",
        "worker_spawn_probe": worker_probe,
    }


def verify_inputs_only(args: Any, admission: dict[str, Any], output: Path) -> None:
    """Exercise real inputs without policy initialization.

    Args: args: Scientific bindings. admission: Qualified admission.
        output: New preflight receipt. Returns: None.
    Raises: ValueError for runtime, source, recipe, data, or CPU drift.
    """
    _require_no_openpi_import()
    runtime = CometOpenPIRuntime.__new__(CometOpenPIRuntime)
    runtime.args = args
    runtime.admission = admission
    runtime._verify_and_install_overlay()
    atomic_json(output, _input_preflight_receipt(args, admission))
