"""Run the pinned public YAM training contract through the vendor interpreter."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

from npa.workbench.flex_pi.runtime import FlexPiError

PUBLIC_TRAINING_DATASET = "flex-pi/sort_utensils"
PUBLIC_TRAINING_REVISION = "0780dd0a0b281df91abcef9434c4b3ac2757448c"
TRAIN_FRAMES = 115620
VALIDATION_FRAMES = 12390
GLOBAL_BATCH = 96


@dataclass(frozen=True)
class TrainingRequest:
    """Execution controls that leave the public training objective unchanged.

    Args:
        output_path: Run-scoped S3 destination for results and checkpoints.
        mode: Profile or complete training with fresh-process resume verification.
        normalization_path: Original normalization as an exact S3 object.
        normalization_sha256: SHA-256 required with normalization_path.
        num_workers: Data loader workers per rank.
        prefetch_factor: Queued batches per worker.
        optimizer: Default, foreach, or fused AdamW implementation.
        memory_fill: Keep deterministic allocation fills on, or qualify them off.
        activation_checkpointing: Recompute activations, or retain them in GPU memory.
        microbatch_per_rank: One or three samples per GPU; effective batch stays 96.
        cuda_graphs: Off for eager execution or mot for training-only CUDA capture.
        run_id: Caller-assigned provenance identifier.
        runtime_image: Exact runtime image reference for provenance.
        dry_run: Return the frozen plan without executing it.
    Returns:
        An immutable request.
    Raises:
        None; run_training validates requests before execution.
    """

    output_path: str
    mode: str = "train"
    normalization_path: str = ""
    normalization_sha256: str = ""
    num_workers: int = 4
    prefetch_factor: int = 4
    optimizer: str = "default"
    memory_fill: str = "on"
    activation_checkpointing: str = "on"
    microbatch_per_rank: int = 1
    cuda_graphs: str = "off"
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request):
    from npa.workbench.flex_pi.training_activation import (
        activation_checkpointing_overrides,
    )
    from npa.workbench.flex_pi.training_normalization import validate_normalization
    from npa.workbench.flex_pi.training_graphs import validate_cuda_graphs

    validate_normalization(request)
    activation_checkpointing_overrides(request.activation_checkpointing)
    validate_cuda_graphs(request.cuda_graphs, request.activation_checkpointing)
    if type(
        request.microbatch_per_rank
    ) is not int or request.microbatch_per_rank not in {1, 3}:
        raise FlexPiError("microbatch-per-rank must be one or three")
    if request.mode not in {"profile", "profile-resume", "train"}:
        raise FlexPiError("mode must be profile, profile-resume or train")
    if request.optimizer not in {"default", "foreach", "fused"}:
        raise FlexPiError("optimizer must be default, foreach, or fused")
    if request.memory_fill not in {"on", "off"}:
        raise FlexPiError("memory-fill must be on or off")
    if request.memory_fill == "off" and not request.normalization_path:
        raise FlexPiError("memory-fill off requires verified original normalization")
    if request.num_workers < 0 or request.prefetch_factor < 1:
        raise FlexPiError("workers must be nonnegative and prefetch positive")
    parsed = urlparse(request.output_path)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise FlexPiError("training paths must be scoped s3:// locations")


def _plan(request):
    return {
        "schema": "npa.flex_pi.training.v1",
        "dataset": PUBLIC_TRAINING_DATASET,
        "dataset_revision": PUBLIC_TRAINING_REVISION,
        "train_frames": TRAIN_FRAMES,
        "validation_frames": VALIDATION_FRAMES,
        "gpu_count": 4,
        "effective_batch": GLOBAL_BATCH,
        "microbatch_per_rank": request.microbatch_per_rank,
        "gradient_accumulation_steps": 24 // request.microbatch_per_rank,
        "final_training_batch": 36,
        "precision": "bf16",
        "peak_learning_rate": 1e-4,
        "non_comparable_to_reference": True,
        "reference_benchmark_beaten": False,
        "execution": asdict(request),
    }


def _vendor_command(request_path):
    return [
        os.environ.get("FLEX_PI_PYTHON", "/opt/conda/bin/python"),
        "-m",
        "npa.workbench.flex_pi.training_worker",
        "--request",
        str(request_path),
    ]


def run_training(request: TrainingRequest) -> dict:
    """Execute real public-data training or its measurement/resume gate.

    Args:
        request: Pinned-workload execution and output controls.
    Returns:
        Verified training result, or a plan when dry_run is true.
    Raises:
        FlexPiError: The request, worker, or result fails validation.
    """
    _validate(request)
    plan = _plan(request)
    if request.dry_run:
        return plan
    from npa.workbench.flex_pi.training_topology import training_topology
    from npa.workbench.flex_pi.training_multinode import launch_training

    if training_topology()["nodes"] == 4:
        return launch_training(request)
    return _execute_training(request, plan)


def _execute_training(request, plan):
    cache = (
        Path(os.environ.get("NPA_MODEL_CACHE_DIR", "/workspace/.cache/npa"))
        / "flex-pi-training"
    )
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flex-pi-training-") as directory:
        root = Path(directory)
        work = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
        identity = hashlib.sha256(
            Path(__file__).with_name("training_sources.json").read_bytes()
        ).hexdigest()
        plan.update(
            work_directory=str(work), asset_directory=str(cache / "assets" / identity)
        )
        _prepare_normalization(request, plan, work)
        qualification = _qualify_memory_fill(plan, root, work)
        result = _run_phase(plan, root)
        if qualification:
            result["memory_fill_qualification"] = [qualification]
        return _publish_result(request, plan, result, root, work)


def _prepare_normalization(request, plan, work):
    from npa.workbench.flex_pi.training_normalization import stage_normalization

    if request.normalization_path:
        plan["normalization_file"] = str(stage_normalization(request, work))


def _qualify_memory_fill(plan, root, work, *, checkpoint=None):
    from npa.workbench.flex_pi.training_qualification import qualify_memory_fill

    if plan["execution"].get("memory_fill", "on") == "on":
        return None
    try:
        return qualify_memory_fill(plan, root, work, _run_phase, checkpoint=checkpoint)
    except (FlexPiError, OSError, ValueError, KeyError):
        _publish_qualification_failure(plan, work, checkpoint=checkpoint)
        raise


def _publish_qualification_failure(plan, work, *, checkpoint):
    from npa.clients.storage import StorageClient
    from npa.workbench.flex_pi.training_artifacts import sha256_file

    stage = "checkpoint" if checkpoint is not None else "initial"
    path = work / f"fill-{stage}-rejection.json"
    if not path.is_file():
        return
    destination = plan["execution"]["output_path"].rstrip("/") + "/" + path.name
    try:
        storage = StorageClient.from_environment()
        storage.upload_file(str(path), destination)
        readback = work / f"fill-{stage}-rejection-readback.json"
        storage.download_file(destination, str(readback))
        digest = sha256_file(path)
        if digest != sha256_file(readback):
            raise FlexPiError("qualification rejection readback differs")
        proof = work / f"fill-{stage}-rejection-publication.json"
        proof.write_text(
            json.dumps({"sha256": digest, "read_after_write_verified": True})
        )
        proof.chmod(0o400)
    except Exception as error:
        # Retain the original numeric rejection and private local receipt.
        print(
            f"qualification evidence publication failed: {type(error).__name__}",
            file=sys.stderr,
        )


def _run_phase(plan, root):
    from npa.workbench.flex_pi.training_multinode import ACTIVE

    if ACTIVE is not None:
        result = ACTIVE.dispatch(plan, root)
    else:
        result = _run_local_phase(plan, root)
    if result.get("reference_benchmark_beaten") is not False:
        raise FlexPiError("public training must remain non-comparable")
    if plan["execution"].get("cuda_graphs", "off") == "mot":
        from npa.workbench.flex_pi.training_graphs import validate_graphs_receipt

        validate_graphs_receipt(result.get("training_graphs"), "mot", plan["gpu_count"])
    return result


def _phase_environment():
    env = os.environ.copy()
    env.pop("NPA_OPENPI_ACCEPT_GEMMA_TERMS", None)
    source = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    return env


def _run_local_phase(plan, root):
    request_path = root / "request.json"
    request_path.write_text(json.dumps(plan), encoding="utf-8")
    request_path.chmod(0o600)
    process = subprocess.run(
        _vendor_command(request_path), env=_phase_environment(), stdout=sys.stderr
    )
    if process.returncode:
        raise FlexPiError(f"training worker exited {process.returncode}")
    return json.loads((root / "result.json").read_text())


def _publish_result(request, plan, result, root, work):
    from npa.clients.storage import StorageClient
    from npa.workbench.flex_pi.training_artifacts import publish_json
    from npa.workbench.flex_pi.training_cleanup import cleanup_checkpoint_copies

    storage = StorageClient.from_environment()
    destination = request.output_path.rstrip("/")
    for filename in (
        "measurements.jsonl",
        "profile.json",
        "workload.json",
        "dataset_stats.json",
        "capability.json",
    ):
        path = work / filename
        if path.is_file():
            storage.upload_file(str(path), destination + "/" + filename)
    copies = None
    if request.mode in {"train", "profile-resume"}:
        copies = _verify_fresh_resume(plan, result, root, work, destination, storage)
    result.update({k: v for k, v in _plan(request).items() if k != "execution"})
    result["execution"] = {
        key: value
        for key, value in asdict(request).items()
        if key not in {"output_path", "normalization_path"}
    }
    publish_json(
        result, root / "published-result.json", destination + "/result.json", storage
    )
    if copies:
        cleanup_checkpoint_copies(work, **copies)
    return result


def _verify_fresh_resume(plan, result, root, work, destination, storage):
    from npa.workbench.flex_pi.training_artifacts import (
        publish_checkpoint,
        publish_json,
        restore_checkpoint,
    )

    checkpoint = result["checkpoint"]
    source = destination + "/checkpoint"
    original = Path(checkpoint["state_path"])
    manifest = publish_checkpoint(original, source, storage)
    restore = work / "restored-state"
    restore_checkpoint(manifest, source, restore, storage)
    publish_json(
        manifest, root / "checkpoint-manifest.json", source + "/manifest.json", storage
    )
    plan = {
        **plan,
        "distributed_checkpoint": {
            "source": source,
            "manifest": manifest,
            "directory": str(restore),
        },
    }
    qualification = _qualify_memory_fill(plan, root, work, checkpoint=restore)
    if qualification:
        result["memory_fill_qualification"].append(qualification)
    resumed = _run_phase(_resume_plan(plan, work, restore), root)
    _assert_resume_parity(result, resumed)
    checkpoint.pop("state_path")
    verified_key = (
        "profile_checkpoint_resume_verified"
        if plan["execution"]["mode"] == "profile-resume"
        else "checkpoint_resume_verified"
    )
    result[verified_key] = True
    result["checkpoint_read_after_write_verified"] = True
    result["resume"] = resumed
    return {
        "original": original,
        "restored": restore,
        "step": checkpoint["step"],
        "manifest": manifest,
    }


def _resume_plan(plan, work, restore):
    return {
        **plan,
        "work_directory": str(work / "resume"),
        "resume_directory": str(restore),
        "normalization_file": str(restore / "dataset_stats.json"),
        "resume_probe_kind": "profile"
        if plan["execution"]["mode"] == "profile-resume"
        else "full",
        "execution": {**plan["execution"], "mode": "resume"},
    }


def _assert_resume_parity(result, resumed):
    checkpoint = result["checkpoint"]
    for key in ("workload_sha256", "normalization_sha256"):
        if resumed[key] != result[key]:
            raise FlexPiError("fresh resume changed the workload or normalization")
    if resumed["loaded_model_sha256"] != checkpoint["model_sha256"]:
        raise FlexPiError("fresh resume did not restore the complete model")
    if resumed["loaded_training_state"] != checkpoint["training_state"]:
        raise FlexPiError(
            "fresh resume did not restore every rank's optimizer, cursor and RNG state"
        )
    if (
        resumed["loaded_step"] != checkpoint["step"]
        or resumed["resume_probe"] != result["resume_probe"]
    ):
        raise FlexPiError(
            "fresh checkpoint continuation differs from uninterrupted continuation"
        )
