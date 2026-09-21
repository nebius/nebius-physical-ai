"""Measure exact distributed epochs using the upstream Flex-Pi model and loss."""

import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from flexpi.datasets.lerobot import base_lerobot_dataset
from flexpi.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from flexpi.trainer import Wan22Trainer

from npa.workbench.flex_pi.training import GLOBAL_BATCH, TRAIN_FRAMES, VALIDATION_FRAMES
from npa.workbench.flex_pi.training_metrics import PROFILE_UPDATES
from npa.workbench.flex_pi.training_state import rng_digest, state_digest


class ExactTrainingDataset(RobotVideoDataset):
    """Keep upstream transforms and reject decode errors without substitution.

    Args:
        **kwargs: Upstream RobotVideoDataset configuration.
    Returns:
        A dataset whose samples include their exact anchor indices.
    Raises:
        RuntimeError: Sampling requests replacement of padded anchors.
    """

    def __getitem__(self, index):
        base_lerobot_dataset.MAX_GETITEM_ATTEMPT = 1
        if self.skip_padding_as_possible:
            raise RuntimeError("exact training forbids replacement sampling")
        sample = self._get(index)
        sample["npa_sample_index"] = int(index)
        return sample


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        tensor = value.detach().contiguous().reshape(-1).view(torch.uint8)
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


class VerifiedTrainer(Wan22Trainer):
    """Add exact accounting, complete validation and strict checkpoint gates.

    Args:
        cfg: Resolved upstream training configuration.
        model: Initialized upstream Flex-Pi model.
        train_dataset: Complete frozen training split.
        val_dataset: Complete frozen held-out split.
    Returns:
        An upstream trainer with acceptance checks.
    Raises:
        RuntimeError: Execution fails a distributed or workload acceptance check.
    """

    def _build_optimizer(self, parameters):
        self._configure_ddp()
        mode = str(self.cfg.npa_optimizer)
        options = {} if mode == "default" else {mode: True}
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            **options,
        )

    def _configure_ddp(self):
        import accelerate
        from accelerate.utils import DistributedDataParallelKwargs

        # This upstream constructor hook runs before accelerator.prepare().
        if accelerate.__version__ != "1.12.0":
            raise RuntimeError(
                "the fixed reducer adapter requires pinned Accelerate 1.12.0"
            )
        handler = self.accelerator.ddp_handler
        if handler is None:
            if (
                str(self.accelerator.distributed_type.value) != "MULTI_GPU"
                or isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                or getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
                or getattr(self.accelerator.state, "fsdp_plugin", None) is not None
            ):
                raise RuntimeError(
                    "fixed reducer setup requires native DDP before prepare"
                )
            handler = DistributedDataParallelKwargs(
                find_unused_parameters=True, static_graph=False
            )
            self.accelerator.ddp_handler = handler
        if not handler.find_unused_parameters or handler.static_graph:
            raise RuntimeError("checkpoint replay requires fixed DDP bucket layout")

    def _build_loader(self, dataset, worker_init_fn=None):
        loader = super()._build_loader(dataset, worker_init_fn)
        if self.num_workers:
            loader.prefetch_factor = int(self.cfg.npa_prefetch_factor)
        return loader

    def _check_contract(self):
        if str(self.accelerator.distributed_type.value) != "MULTI_GPU":
            raise RuntimeError("the frozen optimizer contract requires native DDP")
        if not self.model.find_unused_parameters or self.model.static_graph:
            raise RuntimeError(
                "actual DDP wrapping differs from the fixed bucket policy"
            )
        if self.accelerator.num_processes != 4 or self.batch_size != 1:
            raise RuntimeError(
                "the accepted contract requires four ranks and microbatch one"
            )
        if self.gradient_accumulation_steps * self.batch_size != 24:
            raise RuntimeError("the accepted effective batch is 96")
        if (
            len(self.train_dataset) != TRAIN_FRAMES
            or len(self.val_dataset) != VALIDATION_FRAMES
        ):
            raise RuntimeError("actual dataset lengths differ from the frozen split")
        if self.train_dataset is self.val_dataset:
            raise RuntimeError("validation must be the distinct held-out split")
        split = json.loads(Path(__file__).with_name("training_split.json").read_text())
        for dataset, key in (
            (self.train_dataset, "train_episode_ids"),
            (self.val_dataset, "validation_episode_ids"),
        ):
            children = dataset.lerobot_dataset.multi_dataset._datasets
            actual = [int(index) for child in children for index in child.episodes]
            if sorted(actual) != sorted(split[key]):
                raise RuntimeError(
                    "selected episode membership differs from the immutable split"
                )
        self.accelerator.even_batches = False
        self.train_loader.batch_sampler.even_batches = False

    def _record(self, value):
        if self.accelerator.is_main_process:
            with (Path(self.output_dir) / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            print(json.dumps(value, allow_nan=False), flush=True)

    def _backward(self, sample, divisor):
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss, _ = self.model(sample)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            self.accelerator.backward(
                loss * (self.gradient_accumulation_steps / divisor)
            )
            if not self.accelerator.sync_gradients:
                return float(loss.detach())
            norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite training gradient")
            if self.global_step == 0:
                missing = [
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                if missing:
                    raise RuntimeError(
                        f"trainable parameters have no distributed gradient: {missing}"
                    )
            self.optimizer.step()
            if self.accelerator.optimizer_step_was_skipped:
                raise RuntimeError("optimizer update was skipped")
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            return float(loss.detach())

    def _train_updates(self, updates):
        indices = []
        losses = []
        iterator = iter(self.train_loader)
        start = time.perf_counter()
        start_step = self.global_step
        local_epoch_batches = TRAIN_FRAMES // (4 * self.batch_size)
        loader_wait = 0.0
        for _ in range(local_epoch_batches):
            wait_start = time.perf_counter()
            try:
                sample = next(iterator)
            except StopIteration:
                break
            loader_wait += time.perf_counter() - wait_start
            indices.extend(sample.pop("npa_sample_index").detach().cpu().tolist())
            self.batch_in_epoch += 1
            remainder = local_epoch_batches % self.gradient_accumulation_steps
            tail = self.batch_in_epoch > local_epoch_batches - remainder
            loss = self._backward(
                sample, remainder if tail else self.gradient_accumulation_steps
            )
            losses.append(loss)
            if self.accelerator.sync_gradients:
                self._record_update(start, losses, loader_wait)
                if getattr(self, "_profiler", None) is not None:
                    self._profiler.step()
                losses = []
                loader_wait = 0.0
                start = time.perf_counter()
                if self.global_step - start_step >= updates:
                    break
        return indices

    def _record_update(self, start, losses, loader_wait):
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            time.perf_counter() - start, device=self.accelerator.device
        )
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        elapsed = float(elapsed.item())
        count = len(losses) * 4 * self.batch_size
        loss_sum = torch.tensor(
            sum(losses) * self.batch_size, device=self.accelerator.device
        )
        torch.distributed.all_reduce(loss_sum)
        measurement = {
            "step": self.global_step,
            "samples": count,
            "seconds": elapsed,
            "samples_per_second": count / elapsed,
            "loss": float(loss_sum.item()) / count,
            "rank_zero_loader_wait_seconds": loader_wait,
        }
        self._record(measurement)

    def _profile_updates(self, updates):
        if not self.accelerator.is_main_process:
            return self._train_updates(updates)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=1),
            # PyTorch 2.7.1 shape capture overflows on deterministic uint64 fills
            # in FlashAttention (pytorch/pytorch#150601). Keep its math unchanged.
            record_shapes=False,
            profile_memory=True,
            on_trace_ready=lambda trace: trace.export_chrome_trace(
                str(Path(self.output_dir) / "profile.json")
            ),
        ) as profiler:
            self._profiler = profiler
            indices = self._train_updates(updates)
        self._profiler = None
        return indices

    def _verify_indices(self, local_indices, expected):
        gathered = [None] * 4
        torch.distributed.all_gather_object(gathered, local_indices)
        indices = [index for rank in gathered for index in rank]
        if len(indices) != expected or len(set(indices)) != expected:
            raise RuntimeError(
                "sample accounting detected dropped or duplicate anchors"
            )
        if expected == TRAIN_FRAMES and sorted(indices) != list(range(TRAIN_FRAMES)):
            raise RuntimeError("training epoch did not cover the complete split")
        return hashlib.sha256(json.dumps(indices).encode()).hexdigest()

    def _validation(self):
        import numpy as np

        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        loader = self._validation_loader()
        total = torch.zeros(2, dtype=torch.float64, device=self.accelerator.device)
        start = time.perf_counter()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with (
            torch.no_grad(),
            torch.random.fork_rng(devices=[torch.cuda.current_device()]),
        ):
            for sample in loader:
                index = int(sample.pop("npa_sample_index").item())
                torch.manual_seed(1000000 + index)
                torch.cuda.manual_seed(1000000 + index)
                with self.accelerator.autocast():
                    loss, _ = model(sample)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite validation loss")
                total[0] += loss.detach().double()
                total[1] += 1
        torch.distributed.all_reduce(total)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if int(total[1].item()) != VALIDATION_FRAMES:
            raise RuntimeError("full validation sample count mismatch")
        self._set_dit_only_train_mode()
        return {
            "loss": float((total[0] / total[1]).item()),
            "samples": VALIDATION_FRAMES,
            "seconds": time.perf_counter() - start,
        }

    def _validation_loader(self):
        rank = self.accelerator.process_index
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=list(range(rank, VALIDATION_FRAMES, 4)),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _checkpoint(self):
        start = time.perf_counter()
        checkpoint = self.save_checkpoint()
        if not checkpoint.get("state_path"):
            raise RuntimeError("upstream checkpoint save failed")
        root = Path(checkpoint["state_path"])
        if not (root / "trainer_state.json").is_file():
            raise RuntimeError("checkpoint lacks the exact training cursor")
        if self.accelerator.is_main_process:
            shutil.copyfile(
                Path(self.output_dir) / "dataset_stats.json",
                root / "dataset_stats.json",
            )
        self.accelerator.wait_for_everyone()
        digest = _state_digest(self.accelerator.unwrap_model(self.model))
        gathered = [None] * 4
        torch.distributed.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise RuntimeError("distributed model parameters diverged")
        return {
            "state_path": str(root),
            "model_sha256": digest,
            "training_state": self._training_state_receipt(),
            "seconds": time.perf_counter() - start,
            "step": self.global_step,
        }

    def _training_state_receipt(self, *, continuation=False):
        local = {
            "rank": self.accelerator.process_index,
            "rng_sha256": rng_digest(cuda_only=continuation),
            "rng_scope": "current_cuda" if continuation else "all_generators",
            "optimizer_sha256": state_digest(self.optimizer.state_dict()),
            "scheduler_sha256": state_digest(self.scheduler.state_dict()),
            "accelerator_step": self.accelerator.step,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }
        gathered = [None] * 4
        torch.distributed.all_gather_object(gathered, local)
        if len({row["optimizer_sha256"] for row in gathered}) != 1:
            raise RuntimeError("distributed optimizer states diverged")
        return gathered

    def _synchronized_digest(self):
        digest = _state_digest(self.accelerator.unwrap_model(self.model))
        gathered = [None] * 4
        torch.distributed.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise RuntimeError(
                "distributed model states diverged after optimizer updates"
            )
        return digest

    def _resume_probe(self, *, profile=False):
        from npa.workbench.flex_pi.training_resume import prepare_continuation

        expected = prepare_continuation(self, profile=profile)
        indices = self._train_updates(1)
        if indices != expected:
            raise RuntimeError("checkpoint continuation used the wrong ordered anchors")
        return {
            "step": self.global_step,
            "sample_order_sha256": self._verify_indices(indices, GLOBAL_BATCH),
            "model_sha256": self._synchronized_digest(),
            "scheduler": self.scheduler.state_dict(),
            "training_state": self._training_state_receipt(continuation=True),
        }

    def execute(self, mode, *, profile_resume=False):
        """Run the selected acceptance phase without changing its data or objective.

        Args:
            mode: Profile, complete epoch, or checkpoint continuation.
            profile_resume: Restrict a fresh diagnostic probe to the profile cursor.
        Returns:
            Measured phase evidence with exact sample accounting.
        Raises:
            RuntimeError: Distributed, data, numeric, or persistence checks fail.
        """
        self._check_contract()
        self._set_dit_only_train_mode()
        initial_digest = _state_digest(self.accelerator.unwrap_model(self.model))
        result = {
            "mode": mode,
            "world_size": 4,
            "initial_model_sha256": initial_digest,
        }
        if mode == "resume":
            return self._resume_result(result, profile=profile_resume)
        if mode == "train":
            result["initial_validation"] = self._validation()
        self._measure_training(mode, result)
        if mode == "train":
            self._finish_epoch(result)
        elif mode == "profile-resume":
            result["checkpoint"] = self._checkpoint()
            result["resume_probe"] = self._resume_probe(profile=True)
            result.update(full_epoch_completed=False, validation_completed=False)
        return result

    def _measure_training(self, mode, result):
        profiling = mode in {"profile", "profile-resume"}
        updates = (
            PROFILE_UPDATES if profiling else math.ceil(TRAIN_FRAMES / GLOBAL_BATCH)
        )
        indices = (
            self._profile_updates(updates)
            if profiling
            else self._train_updates(updates)
        )
        expected = PROFILE_UPDATES * GLOBAL_BATCH if profiling else TRAIN_FRAMES
        result["sample_order_sha256"] = self._verify_indices(indices, expected)
        result["samples"] = expected
        result["final_model_sha256"] = self._synchronized_digest()

    def _resume_result(self, result, *, profile=False):
        result["loaded_model_sha256"] = _state_digest(
            self.accelerator.unwrap_model(self.model)
        )
        result["loaded_step"] = self.global_step
        result["loaded_training_state"] = self._training_state_receipt()
        result["resume_probe"] = self._resume_probe(profile=profile)
        return result

    def _finish_epoch(self, result):
        result["checkpoint"] = self._checkpoint()
        result["final_validation"] = self._validation()
        before, after = (
            result["initial_validation"]["loss"],
            result["final_validation"]["loss"],
        )
        if after > before:
            raise RuntimeError("full held-out validation loss regressed")
        result["full_epoch_completed"] = True
        result["validation_completed"] = True
        result["resume_probe"] = self._resume_probe()
