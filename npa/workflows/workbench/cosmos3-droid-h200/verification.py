"""Record actual per-rank samples, finite losses and optimizer operations."""

import json
import math
import os
from pathlib import Path

import torch
from cosmos_framework.utils.callback import Callback


class _Receipt(Callback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.rank = int(os.environ["RANK"])
        self.microsteps = 0
        self.samples = 0
        self.updates = 0
        self.optimizer_operations = 0
        self.last_update_microsteps = 0
        self.finite_losses = True
        self.root = Path(os.environ["NPA_DROID_SETTINGS"]).parent / "verification"

    def _persist(self, terminal=False):
        self.root.mkdir(mode=0o700, exist_ok=True)
        path = self.root / f"rank-{self.rank}.json"
        data = {
            "rank": self.rank,
            "microsteps": self.microsteps,
            "samples": self.samples,
            "updates": self.updates,
            "optimizer_operations": self.optimizer_operations,
            "finite_losses": self.finite_losses,
            "terminal": terminal,
            "gpu_name": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "TORCH_COMPILE_DISABLE": os.environ.get("TORCH_COMPILE_DISABLE"),
            "checkpoint_saves_suppressed": True,
            "configuration": _native_configuration(self.trainer.config),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)

    def on_train_start(self, model, iteration=0):
        config = self.trainer.config
        if (
            int(os.environ["WORLD_SIZE"]) != 8
            or "H200" not in torch.cuda.get_device_name()
            or config.dataloader_train.max_samples_per_batch != 32
            or config.trainer.grad_accum_iter != 32
        ):
            raise ValueError(
                "verification requires eight H200 ranks and batch 32/accumulation 32"
            )
        self._persist()

    def on_training_step_batch_start(self, model, data_batch, iteration=0):
        batch = data_batch.get("video", data_batch.get("images"))
        if not isinstance(batch, (list, tuple)) or len(batch) != 32:
            raise ValueError("native packed batch does not contain 32 samples")

    def on_training_step_batch_end(
        self, model, data_batch, output_batch, loss, iteration=0
    ):
        self.finite_losses = self.finite_losses and math.isfinite(
            float(loss.detach().float().item())
        )
        if not self.finite_losses:
            raise ValueError("nonfinite DROID loss")
        self.microsteps += 1
        self.samples += 32
        self._persist()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.updates += 1
        if (
            self.microsteps - self.last_update_microsteps != 32
            or self.optimizer_operations != self.updates
        ):
            raise ValueError(
                "optimizer update does not match 32 accumulated microsteps"
            )
        self.last_update_microsteps = self.microsteps
        self._persist()

    def on_train_end(self, model, iteration=0):
        if self.updates != self.trainer.config.trainer.max_iter:
            raise ValueError("requested updates did not complete")
        self._persist(terminal=True)


def _native_configuration(config):
    loader = config.dataloader_train.dataloader
    dataset = loader.datasets.droid.dataset
    return {
        "batch_per_rank": config.dataloader_train.max_samples_per_batch,
        "accumulation": config.trainer.grad_accum_iter,
        "workers_per_rank": loader.num_workers,
        "inner_batch": loader.batch_size,
        "prefetch_factor": loader.prefetch_factor,
        "fps": dataset.fps,
        "chunk_length": dataset.chunk_length,
        "action_space": dataset.action_space,
        "use_state": dataset.use_state,
        "viewpoint": dataset.viewpoint,
        "action_normalization": dataset.action_normalization,
        "use_filter_dict": dataset.use_filter_dict,
        "use_image_augmentation": dataset.use_image_augmentation,
        "model_compilation": config.model.config.compile.enabled,
        "vae_compilation": config.trainer.callbacks.compile_tokenizer.enabled,
        "validation": config.trainer.run_validation,
    }


def install():
    """Attach sample and optimizer observations without saving training checkpoints.

    Args:
        None.
    Returns:
        None.
    Raises:
        ImportError: The pinned native trainer is unavailable.
    """
    from cosmos_framework.trainer import ImaginaireTrainer

    original_initialize = ImaginaireTrainer.__init__
    original_optimizer = ImaginaireTrainer._optimizer_step

    def initialize(trainer, config):
        original_initialize(trainer, config)
        trainer._droid_receipt = _Receipt(trainer)
        trainer.callbacks._callbacks.append(trainer._droid_receipt)
        trainer.checkpointer.save = lambda *args, **kwargs: None

    def optimizer(trainer, *args, **kwargs):
        result = original_optimizer(trainer, *args, **kwargs)
        trainer._droid_receipt.optimizer_operations += 1
        return result

    ImaginaireTrainer.__init__ = initialize
    ImaginaireTrainer._optimizer_step = optimizer
