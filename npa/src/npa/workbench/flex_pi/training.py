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
    """Execution controls that leave the public training objective unchanged."""

    output_path: str
    mode: str = "train"
    num_workers: int = 4
    prefetch_factor: int = 4
    optimizer: str = "default"
    run_id: str = ""
    runtime_image: str = ""
    dry_run: bool = False


def _validate(request):
    if request.mode not in {"profile", "train"}:
        raise FlexPiError(
            "mode must be profile or train (including automatic fresh resume)"
        )
    if request.optimizer not in {"default", "foreach", "fused"}:
        raise FlexPiError("optimizer must be default, foreach, or fused")
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
        "microbatch_per_rank": 1,
        "gradient_accumulation_steps": 24,
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
        result = _run_phase(plan, root)
        return _publish_result(request, plan, result, root, work)


def _run_phase(plan, root):
    request_path = root / "request.json"
    request_path.write_text(json.dumps(plan), encoding="utf-8")
    request_path.chmod(0o600)
    env = os.environ.copy()
    env.pop("NPA_OPENPI_ACCEPT_GEMMA_TERMS", None)
    source = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source, env.get("PYTHONPATH")]))
    process = subprocess.run(_vendor_command(request_path), env=env, stdout=sys.stderr)
    if process.returncode:
        raise FlexPiError(f"training worker exited {process.returncode}")
    result = json.loads((root / "result.json").read_text())
    if result.get("reference_benchmark_beaten") is not False:
        raise FlexPiError("public training must remain non-comparable")
    return result


def _publish_result(request, plan, result, root, work):
    from npa.clients.storage import StorageClient
    from npa.workbench.flex_pi.training_artifacts import publish_json

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
    if request.mode == "train":
        _verify_fresh_resume(plan, result, root, work, destination, storage)
    result.update({k: v for k, v in _plan(request).items() if k != "execution"})
    result["execution"] = {
        key: value for key, value in asdict(request).items() if key != "output_path"
    }
    publish_json(
        result, root / "published-result.json", destination + "/result.json", storage
    )
    return result


def _verify_fresh_resume(plan, result, root, work, destination, storage):
    from npa.workbench.flex_pi.training_artifacts import (
        publish_checkpoint,
        publish_json,
        restore_checkpoint,
    )

    checkpoint = result["checkpoint"]
    source = destination + "/checkpoint"
    manifest = publish_checkpoint(Path(checkpoint["state_path"]), source, storage)
    restore = work / "restored-state"
    restore_checkpoint(manifest, source, restore, storage)
    publish_json(
        manifest, root / "checkpoint-manifest.json", source + "/manifest.json", storage
    )
    resume_plan = {
        **plan,
        "work_directory": str(work / "resume"),
        "resume_directory": str(restore),
        "execution": {**plan["execution"], "mode": "resume"},
    }
    resumed = _run_phase(resume_plan, root)
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
    checkpoint.pop("state_path")
    result["checkpoint_resume_verified"] = True
    result["checkpoint_read_after_write_verified"] = True
    result["resume"] = resumed
