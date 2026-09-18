"""Fine-tune the native XR1 model and select a checkpoint on disjoint robot episodes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from mmengine import Config
from torch.utils.data import DataLoader

from mibot.data.datamodule.base_datamodule import BaseDataModule
from mibot.data.datasets.json_dataset import JsonDataset
from mibot.models.runner.base_runner import BaseRunner


class _ValidationDataset(JsonDataset):
    def _samples(self):
        return [{"file": path, "frame_index": frame} for path in self.files
                for frame in range(0, int(self._read_json(path)["num_frames"]) - 29, 30)]

    def _augment(self, images):
        return images


class _DataModule(BaseDataModule):
    def __init__(self, params, validation_paths):
        super().__init__(params)
        self.validation_params = deepcopy(params)
        self.validation_params.train_datasets.paths = validation_paths

    def val_dataloader(self):
        return DataLoader(_ValidationDataset(self.validation_params), batch_size=self.batch_size,
                          num_workers=2, collate_fn=self.collate_fn, pin_memory=True)


class _Runner(BaseRunner):
    def validation_step(self, batch, batch_idx):
        batch_size = batch["state"].shape[0]
        # Upstream computes its flow/choice losses only in training mode. Disable
        # autograd and fix its noise while preserving that native objective.
        prior_mode = self.model.training
        self.model.train()
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=[self.device.index]):
                torch.manual_seed(4096 + batch_idx * self.trainer.world_size + self.global_rank)
                losses = self.model(batch, return_loss=True)
        finally:
            self.model.train(prior_mode)
        self.log("val_loss", losses["loss"], sync_dist=True, batch_size=batch_size,
                 on_step=False, on_epoch=True)
        for name, value in losses.items():
            self.log(f"validation/{name}", value, sync_dist=True, batch_size=batch_size)


class _Evidence(Callback):
    def __init__(self, output: Path):
        self.output = output

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        row = {"optimizer_step": trainer.global_step, "baseline": trainer.sanity_checking,
               "metrics": {key: float(value) for key, value in trainer.callback_metrics.items()
                           if key == "val_loss" or key.startswith("validation/")}}
        if "val_loss" not in row["metrics"]:
            raise ValueError("Native XR1 validation did not produce a checkpoint-selection metric")
        with (self.output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print("XR1_VALIDATION " + json.dumps(row), flush=True)


def _trainer(configuration: Config, output: Path) -> tuple[Trainer, ModelCheckpoint]:
    settings = deepcopy(configuration.trainer)
    settings.pop("optimizer")
    settings.pop("scheduler")
    seed_everything(settings.pop("seed"), workers=True)
    settings.pop("save_interval")
    settings.pop("project")
    settings.pop("exp_name")
    settings.pop("ckpt_path")
    strategy = DeepSpeedStrategy(stage=2, **settings.pop("strategy")["params"])
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val_loss", mode="min",
                                 save_top_k=1, save_last=True, save_on_train_epoch_end=False,
                                 filename="step={step}-loss={val_loss:.6f}")
    trainer = Trainer(**settings, strategy=strategy, val_check_interval=1000,
                      num_sanity_val_steps=-1, log_every_n_steps=10,
                      callbacks=[_Evidence(output), checkpoint],
                      logger=CSVLogger(str(output), name="metrics"))
    return trainer, checkpoint


def _fit(configuration_path: Path) -> None:
    configuration = Config.fromfile(str(configuration_path))
    output = configuration_path.parent
    params = deepcopy(configuration.model.params)
    params.optimizer = configuration.trainer.optimizer
    params.scheduler = configuration.trainer.scheduler
    data = _DataModule(configuration.data.params, configuration.validation_paths)
    model = _Runner(params)
    trainer, checkpoint = _trainer(configuration, output)
    trainer.fit(model=model, datamodule=data)
    trainer.strategy.barrier()
    if trainer.is_global_zero:
        _export(checkpoint, configuration, output, trainer.global_step)


def _export(checkpoint, configuration: Config, output: Path, steps: int) -> None:
    state_path = Path(checkpoint.best_model_path) / "checkpoint/mp_rank_00_model_states.pt"
    candidate = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)["module"]
    baseline = torch.load(configuration.model.params.pretrained, map_location="cpu",
                          mmap=True, weights_only=True)["module"]
    if set(candidate) != set(baseline):
        raise ValueError("Trained checkpoint keys differ from the exact baseline model")
    changed, total = 0, 0
    for name, value in candidate.items():
        changed += int(torch.count_nonzero(value != baseline[name]))
        total += value.numel()
    if not changed:
        raise ValueError("Fine-tuning did not change any model parameters")
    export = output / "candidate"
    export.mkdir()
    torch.save({"module": candidate}, export / "model_states.pt")
    configuration.dump(str(export / "config.py"))
    report = {"optimizer_steps": steps, "checkpoint_selection": "minimum held-out native XR1 loss",
              "best_validation_loss": float(checkpoint.best_model_score),
              "changed_parameters": changed, "total_parameters": total,
              "closed_loop_evaluation_required": True}
    (export / "training.json").write_text(json.dumps(report, indent=2))
