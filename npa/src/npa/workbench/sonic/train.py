"""In-job SONIC locomotion training.

``npa workbench sonic train --runtime local`` trains inside the container it is
invoked from instead of delegating to another VM, container, or serverless Job.
That is what makes SONIC usable as a stage of an ``npa.workflow``: the workflow
already holds a GPU, so a stage that submits *more* infrastructure both needs
credentials the job does not have and doubles the spend.

Two trainers, picked by what the container actually ships:

``sonic-entrypoint``
    The ``npa-sonic`` image ships the upstream GR00T-WholeBodyControl trainer
    behind ``/entrypoint.sh train``. When it is present, run it — that is real
    SONIC/Isaac Lab training.

``reference-locomotion``
    Otherwise fit the reference locomotion actor
    (:mod:`npa.workbench.sonic.reference_policy`) to a command-conditioned gait
    teacher with real gradient descent on the job's GPU. It produces a real
    ``torch.nn.Module`` checkpoint that ``sonic export`` and ``sonic eval``
    consume unchanged, so the train -> export -> eval chain is self-contained on
    any image with torch.

Both write ``checkpoint.pt`` plus a ``checkpoint.json`` manifest to
``--output-path`` (local directory or ``s3://`` prefix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

CHECKPOINT_FILE_NAME = "checkpoint.pt"
MANIFEST_FILE_NAME = "checkpoint.json"
TRAIN_MANIFEST_FORMAT = "npa_sonic_train_manifest_v1"
CHECKPOINT_FORMAT = "npa_sonic_checkpoint_v1"
ENTRYPOINT_TRAINER = "sonic-entrypoint"
REFERENCE_TRAINER = "reference-locomotion"
DEFAULT_ENTRYPOINT = "/entrypoint.sh"
#: Gradient steps per reported iteration. Iterations are the operator-facing
#: knob (``--max-iterations``); this keeps one iteration a meaningful amount of
#: work without adding a second flag.
STEPS_PER_ITERATION = 64
MIN_BATCH_SIZE = 256


class SonicTrainError(ValueError):
    """Raised when an in-job SONIC training run cannot complete."""


@dataclass
class LocalTrainResult:
    """Outcome of one in-job SONIC training run."""

    status: str
    runtime: str
    trainer: str
    output_path: str
    checkpoint_uri: str
    manifest_uri: str
    embodiment: str
    iterations: int
    device: str = ""
    batch_size: int = 0
    observation_dim: int = 0
    action_dim: int = 0
    initial_loss: float = 0.0
    final_loss: float = 0.0
    loss_reduction: float = 0.0
    warm_start: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "runtime": self.runtime,
            "trainer": self.trainer,
            "output_path": self.output_path,
            "checkpoint_uri": self.checkpoint_uri,
            "manifest_uri": self.manifest_uri,
            "embodiment": self.embodiment,
            "iterations": self.iterations,
            "device": self.device,
            "batch_size": self.batch_size,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_reduction": self.loss_reduction,
            "warm_start": self.warm_start,
        }
        if self.metrics:
            payload["metrics"] = self.metrics
        return payload


def resolve_entrypoint(explicit: str = "") -> str:
    """Return the SONIC trainer entrypoint available in this container, if any."""

    candidate = (
        explicit
        or os.environ.get("SONIC_TRAIN_ENTRYPOINT", "")
        or DEFAULT_ENTRYPOINT
    ).strip()
    if not candidate:
        return ""
    path = Path(candidate)
    return str(path) if path.is_file() and os.access(path, os.X_OK) else ""


def train_local(
    *,
    output_path: str,
    checkpoint: str = "",
    data_path: str = "",
    embodiment: str = "UNITREE_G1_SONIC",
    num_envs: int = 16,
    max_iterations: int = 5,
    seed: int = 0,
    device: str = "",
    entrypoint: str = "",
    allow_entrypoint: bool = True,
    storage_client: "StorageClient | None" = None,
) -> dict[str, Any]:
    """Train a SONIC locomotion policy inside the current container."""

    if not output_path:
        raise SonicTrainError("SONIC train --runtime local requires --output-path")
    if max_iterations <= 0:
        raise SonicTrainError(f"--max-iterations must be positive, got {max_iterations}")
    if num_envs <= 0:
        raise SonicTrainError(f"--num-envs must be positive, got {num_envs}")

    resolved_entrypoint = resolve_entrypoint(entrypoint) if allow_entrypoint else ""
    with tempfile.TemporaryDirectory(prefix="npa-sonic-train-") as tmp:
        work_dir = Path(tmp)
        if resolved_entrypoint:
            result = _run_entrypoint_trainer(
                entrypoint=resolved_entrypoint,
                work_dir=work_dir,
                checkpoint=checkpoint,
                data_path=data_path,
                embodiment=embodiment,
                num_envs=num_envs,
                max_iterations=max_iterations,
            )
        else:
            result = _run_reference_trainer(
                work_dir=work_dir,
                checkpoint=checkpoint,
                embodiment=embodiment,
                num_envs=num_envs,
                max_iterations=max_iterations,
                seed=seed,
                device=device,
                storage_client=storage_client,
            )
        manifest = _manifest_payload(result, data_path=data_path)
        (work_dir / MANIFEST_FILE_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checkpoint_uri, manifest_uri = _publish(
            work_dir, output_path, storage_client=storage_client
        )
    result.checkpoint_uri = checkpoint_uri
    result.manifest_uri = manifest_uri
    result.output_path = output_path
    return result.as_dict()


def _publish(
    work_dir: Path,
    output_path: str,
    *,
    storage_client: "StorageClient | None",
) -> tuple[str, str]:
    """Copy the produced artifacts to ``output_path``; return their URIs."""

    if output_path.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        prefix = output_path.rstrip("/") + "/"
        client.upload_directory(str(work_dir), prefix)
        return f"{prefix}{CHECKPOINT_FILE_NAME}", f"{prefix}{MANIFEST_FILE_NAME}"

    dest = Path(output_path)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (CHECKPOINT_FILE_NAME, MANIFEST_FILE_NAME):
        source = work_dir / name
        if source.exists():
            (dest / name).write_bytes(source.read_bytes())
    return str(dest / CHECKPOINT_FILE_NAME), str(dest / MANIFEST_FILE_NAME)


def _manifest_payload(result: LocalTrainResult, *, data_path: str) -> dict[str, Any]:
    return {
        "format": TRAIN_MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": result.runtime,
        "trainer": result.trainer,
        "embodiment": result.embodiment,
        "iterations": result.iterations,
        "steps_per_iteration": STEPS_PER_ITERATION,
        "batch_size": result.batch_size,
        "device": result.device,
        "observation_dim": result.observation_dim,
        "action_dim": result.action_dim,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction": result.loss_reduction,
        "warm_start": result.warm_start,
        "data_path": data_path,
        "checkpoint": CHECKPOINT_FILE_NAME,
        "metrics": result.metrics,
    }


def _run_entrypoint_trainer(
    *,
    entrypoint: str,
    work_dir: Path,
    checkpoint: str,
    data_path: str,
    embodiment: str,
    num_envs: int,
    max_iterations: int,
) -> LocalTrainResult:
    """Run the SONIC image's own trainer in this container."""

    env = dict(os.environ)
    env.update(
        {
            "SONIC_RUN_REAL_TRAIN": "1",
            "NPA_LOCAL_OUTPUT_DIR": str(work_dir),
            "SONIC_CHECKPOINT": checkpoint,
            "SONIC_CHECKPOINT_PATH": checkpoint,
            "SONIC_DATA_PATH": data_path,
            "SONIC_SAMPLE_DATA": "0" if data_path else "1",
            "SONIC_EMBODIMENT": embodiment,
            "SONIC_NUM_ENVS": str(num_envs),
            "SONIC_HEADLESS": "True",
            "SONIC_MAX_ITERATIONS": str(max_iterations),
        }
    )
    completed = subprocess.run(  # noqa: S603 - fixed entrypoint resolved above
        [entrypoint, "train"],
        env=env,
        cwd=str(work_dir),
        check=False,
    )
    if completed.returncode != 0:
        raise SonicTrainError(
            f"SONIC trainer {entrypoint} train exited {completed.returncode}"
        )
    produced = sorted(p.name for p in work_dir.rglob("*") if p.is_file())
    if CHECKPOINT_FILE_NAME not in produced:
        candidate = next(iter(sorted(work_dir.rglob("*.pt"))), None)
        if candidate is None:
            raise SonicTrainError(
                f"SONIC trainer {entrypoint} produced no checkpoint under {work_dir}"
            )
        (work_dir / CHECKPOINT_FILE_NAME).write_bytes(candidate.read_bytes())
    return LocalTrainResult(
        status="trained",
        runtime="local",
        trainer=ENTRYPOINT_TRAINER,
        output_path="",
        checkpoint_uri="",
        manifest_uri="",
        embodiment=embodiment,
        iterations=max_iterations,
        metrics={"artifacts": produced},
    )


def _run_reference_trainer(
    *,
    work_dir: Path,
    checkpoint: str,
    embodiment: str,
    num_envs: int,
    max_iterations: int,
    seed: int,
    device: str,
    storage_client: "StorageClient | None",
) -> LocalTrainResult:
    """Fit the reference locomotion actor with real gradient descent."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SonicTrainError(
            "SONIC train --runtime local requires torch. Install the npa[sonic] "
            "extra or run on the npa-sonic image."
        ) from exc
    from npa.workbench.sonic.reference_policy import (
        DEFAULT_ACTION_DIM,
        DEFAULT_OBS_DIM,
        ReferenceLocomotionPolicy,
        obs_field_spec,
        reference_action,
        sample_observations,
    )

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    batch_size = max(int(num_envs), MIN_BATCH_SIZE)

    policy = ReferenceLocomotionPolicy().to(resolved_device)
    warm_start = _load_warm_start(
        policy, checkpoint, torch=torch, storage_client=storage_client
    )

    # Normalization statistics come from the sampled state distribution, and are
    # recorded on the policy so `sonic export --normalize baked` folds them into
    # the ONNX graph. The actor itself consumes normalized observations.
    calibration = sample_observations(
        4096,
        observation_dim=DEFAULT_OBS_DIM,
        generator=generator,
        device=resolved_device,
    )
    mean = calibration.mean(dim=0)
    var = calibration.var(dim=0, unbiased=False) + 1e-6

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    losses: list[float] = []
    for _ in range(max_iterations):
        iteration_loss = 0.0
        for _ in range(STEPS_PER_ITERATION):
            obs = sample_observations(
                batch_size,
                observation_dim=DEFAULT_OBS_DIM,
                generator=generator,
                device=resolved_device,
            )
            target = reference_action(obs, action_dim=DEFAULT_ACTION_DIM)
            predicted = policy((obs - mean) / torch.sqrt(var))
            loss = torch.nn.functional.mse_loss(predicted, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iteration_loss += float(loss.detach())
        losses.append(iteration_loss / STEPS_PER_ITERATION)

    policy = policy.to("cpu").eval()
    normalization = {
        "mean": mean.detach().cpu().tolist(),
        "var": var.detach().cpu().tolist(),
        "epsilon": 1e-6,
        "clip": 10.0,
        "source": "reference-locomotion-train",
    }
    # Weights plus a description of how to rebuild the actor, never the pickled
    # module: a pickled module binds the artifact to this exact class path (a
    # later rename silently breaks old checkpoints) and can only be read back by
    # executing whatever the file says. `sonic export` reconstructs the class
    # named here and loads the state dict into it.
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "policy": {
                "class": f"{ReferenceLocomotionPolicy.__module__}."
                f"{ReferenceLocomotionPolicy.__qualname__}",
                "kwargs": {
                    "observation_dim": DEFAULT_OBS_DIM,
                    "action_dim": DEFAULT_ACTION_DIM,
                },
            },
            "policy_state_dict": policy.state_dict(),
            "obs_spec": {"name": "obs", "fields": obs_field_spec()},
            "action_spec": {"name": "action", "dim": DEFAULT_ACTION_DIM},
            "normalization": normalization,
            "control_dt": 0.02,
            "embodiment": embodiment,
        },
        str(work_dir / CHECKPOINT_FILE_NAME),
    )

    initial, final = (losses[0], losses[-1]) if losses else (0.0, 0.0)
    return LocalTrainResult(
        status="trained",
        runtime="local",
        trainer=REFERENCE_TRAINER,
        output_path="",
        checkpoint_uri="",
        manifest_uri="",
        embodiment=embodiment,
        iterations=max_iterations,
        device=str(resolved_device),
        batch_size=batch_size,
        observation_dim=DEFAULT_OBS_DIM,
        action_dim=DEFAULT_ACTION_DIM,
        initial_loss=initial,
        final_loss=final,
        loss_reduction=(initial - final),
        warm_start=warm_start,
        metrics={"iteration_loss": losses},
    )


def _load_warm_start(
    policy: Any,
    checkpoint: str,
    *,
    torch: Any,
    storage_client: "StorageClient | None",
) -> str:
    """Warm-start from a previous reference checkpoint when one is reachable.

    ``--checkpoint`` doubles as a base-model reference in the SONIC CLI, so a
    Hugging Face repo id (``nvidia/GEAR-SONIC``) or a path that is not there yet
    simply means "train from scratch" rather than an error.
    """

    ref = (checkpoint or "").strip()
    if not ref or not (ref.startswith("s3://") or ref.endswith(".pt")):
        return ""
    with tempfile.TemporaryDirectory(prefix="npa-sonic-warmstart-") as tmp:
        local = Path(tmp) / CHECKPOINT_FILE_NAME
        if ref.startswith("s3://"):
            from botocore.exceptions import ClientError

            from npa.clients.storage import StorageClient

            client = storage_client or StorageClient.from_environment()
            try:
                client.download_path(ref, str(local))
            except ClientError:
                return ""
        else:
            source = Path(ref)
            if not source.is_file():
                return ""
            local.write_bytes(source.read_bytes())
        if not local.is_file():
            return ""
        # Everything from here is best-effort: a checkpoint that is corrupt,
        # written by a different trainer, or shaped for another policy means
        # "start from scratch", not "fail the training stage".
        try:
            payload = torch.load(str(local), map_location="cpu", weights_only=True)
            state = _warm_start_state_dict(payload)
            if state is None:
                return ""
            policy.load_state_dict(state)
        except Exception:  # noqa: BLE001 - any unreadable checkpoint is a cold start
            return ""
    return ref


def _warm_start_state_dict(payload: Any) -> dict[str, Any] | None:
    """Pull a policy state dict out of a checkpoint payload, if there is one."""

    if not isinstance(payload, dict):
        return None
    for key in ("policy_state_dict", "state_dict", "model_state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None
