"""Run the pinned native FLUX trainer and publish verified S3 artifacts last."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient

from .artifacts import sha256, verify_checkpoint, verify_export
from .validation import verify_reload, verify_remote_export
from .configuration import index_arguments, training_config, validate_dataset
from .schemas import (
    BASE_REPOSITORY,
    BASE_REVISION,
    SOURCE_REVISION,
    FinetuneRequest,
    Recipe,
)


class FluxActionError(RuntimeError):
    """Report a failed fine-tuning operation with its retained artifact location.

    Args: A descriptive failure message.
    Returns: An exception.
    Raises: None.
    """


def finetune(request: FinetuneRequest, *, dry_run: bool = False) -> dict:
    """Stage a LeRobot dataset, fine-tune FLUX Action, and publish its export.

    Args: request specifies S3 handoffs; dry_run validates only recipe and metadata.
    Returns: A JSON-ready plan or verified artifact receipt.
    Raises: FluxActionError for runtime, validation, or publication failures.
    """
    root = Path(tempfile.mkdtemp(prefix="npa-flux-action-"))
    storage = _storage_client()
    output = root / "artifacts"
    output.mkdir()
    claimed = False
    try:
        recipe = _stage_metadata(storage, request, root)
        config = training_config(recipe, root / "dataset", output, request.processes)
        if dry_run:
            plan = {
                "status": "planned",
                "recipe": recipe.model_dump(mode="json"),
                "source_revision": SOURCE_REVISION,
                "training": config,
            }
            shutil.rmtree(root)
            return plan
        _claim_output(storage, request)
        claimed = True
        result = _run_and_publish(storage, request, recipe, root, output)
    except Exception as exc:
        _retain_failure(storage, request, root, output, claimed, exc)
        raise FluxActionError(
            f"FLUX Action failed; local artifacts retained at {root}: {exc}"
        ) from exc
    shutil.rmtree(root)
    return result


def _run_and_publish(storage, request, recipe, root, output) -> dict:
    storage.download_directory(request.input_path, str(root / "dataset"))
    validate_dataset(root / "dataset", recipe)
    config = training_config(recipe, root / "dataset", output, request.processes)
    _write_json(output / "recipe.json", recipe.model_dump(mode="json"))
    _write_json(output / "training.json", config)
    _execute(recipe, request.processes, root, output)
    result = _receipt(request, recipe, output)
    storage.upload_directory(str(output), request.output_path)
    verify_remote_export(storage, request.output_path, result["export_sha256"])
    _write_json(root / "result.json", result)
    storage.upload_file(str(root / "result.json"), request.output_path + "/result.json")
    return result


def _stage_metadata(
    storage: StorageClient, request: FinetuneRequest, root: Path
) -> Recipe:
    storage.download_file(request.recipe_uri, str(root / "recipe.json"))
    recipe = Recipe.model_validate_json((root / "recipe.json").read_text())
    storage.download_file(
        request.input_path + "/meta/info.json", str(root / "dataset/meta/info.json")
    )
    validate_dataset(root / "dataset", recipe)
    return recipe


def _claim_output(storage: StorageClient, request: FinetuneRequest) -> None:
    storage.require_empty_prefix(request.output_path)
    # A conditional creation serializes writers before any GPU training starts.
    storage.put_bytes_conditional(
        json.dumps({"status": "started", "source_revision": SOURCE_REVISION}).encode(),
        request.output_path + "/run.json",
        if_none_match=True,
        content_type="application/json",
    )


def _execute(recipe: Recipe, processes: int, root: Path, output: Path) -> None:
    runtime = Path(os.environ.get("NPA_FLUX_ACTION_ROOT", "/opt/flux-action"))
    if (runtime / "SOURCE_REVISION").read_text().strip() != SOURCE_REVISION:
        raise FluxActionError(
            "FLUX runtime source revision differs from the integration pin"
        )
    python = str(runtime / ".venv/bin/python")
    hardware_check = (
        "import torch; assert torch.cuda.is_available(); "
        f"assert torch.cuda.device_count() >= {processes}, 'insufficient visible GPUs'"
    )
    _run([python, "-c", hardware_check], output, "hardware")
    index = index_arguments(recipe, root / "dataset", output / "index")
    _run([python, "-m", "flux_action.cli", *index], output, "index")
    _fetch_weights(runtime, output)
    _train(python, processes, output)
    checkpoint = verify_checkpoint(output, recipe)
    _export(python, checkpoint, recipe, output)
    _reload(python, root, recipe, output)


def _fetch_weights(runtime: Path, output: Path) -> None:
    # Resolve all three assets with the operator's HF credential before torchrun.
    args = [
        str(runtime / ".venv/bin/hf"),
        "download",
        BASE_REPOSITORY,
        "--revision",
        BASE_REVISION,
    ]
    for pattern in (
        "flux-3-action-base.safetensors",
        "video_vae.safetensors",
        "text_encoder/*",
    ):
        args.extend(["--include", pattern])
    _run(args, output, "weights")


def _train(python: str, processes: int, output: Path) -> None:
    args = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(processes),
        "-m",
        "flux_action.cli",
        "train",
        "--config",
        str(output / "training.json"),
    ]
    _run(args, output, "train")


def _export(python: str, checkpoint: Path, recipe: Recipe, output: Path) -> None:
    args = [
        python,
        "-m",
        "flux_action.cli",
        "export-checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output / "export"),
        "--profile",
        recipe.export_profile,
        "--dtype",
        "bfloat16",
    ]
    _run(args, output, "export")


def _reload(python: str, root: Path, recipe: Recipe, output: Path) -> None:
    args = [
        python,
        "-m",
        "flux_action.cli",
        "evaluate",
        "--checkpoint",
        str(output / "export"),
        "--source-root",
        str(root / "dataset"),
        "--index-dir",
        str(output / "index"),
        "--split",
        "val" if recipe.val_episodes else "train",
        "--windows-per-episode",
        "1",
        "--max-windows",
        "1",
        "--device",
        "cuda",
        "--decoder",
        "pyav",
        "--output",
        str(output / "reload.json"),
    ]
    _run(args, output, "reload")


def _run(argv: list[str], output: Path, stage: str) -> None:
    with (output / f"{stage}.log").open("w") as log:
        subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)


def _receipt(request: FinetuneRequest, recipe: Recipe, output: Path) -> dict:
    export = verify_export(output / "export")
    config = json.loads((output / "export/config.json").read_text())
    if (
        config.get("torch_dtype") != "bfloat16"
        or export.get("weight_profile") != recipe.export_profile
    ):
        raise ValueError("export dtype or profile differs from the request")
    return {
        "schema": "npa.flux-action.finetune.v1",
        "status": "completed",
        "source_revision": SOURCE_REVISION,
        "base_repository": BASE_REPOSITORY,
        "base_revision": BASE_REVISION,
        "input_path": request.input_path,
        "recipe_sha256": sha256(output / "recipe.json"),
        "index_sha256": sha256(output / "index/manifest.json"),
        "statistics_sha256": sha256(output / "index/statistics.json"),
        "steps": recipe.training.steps,
        "optimizer_updates": recipe.training.steps,
        "training_mode": "full",
        "reload": verify_reload(output / "reload.json", recipe),
        "processes": request.processes,
        "checkpoint_uri": f"{request.output_path}/train/step-{recipe.training.steps}",
        "export_uri": request.output_path + "/export",
        "export_sha256": export["sha256"],
        "encoder_references": {
            key: config[key] for key in ("video_vae_id", "text_encoder_id")
        },
        "robot": recipe.robot.model_dump(mode="json"),
        "weights_license": "FLUX Kommunity License v1.0",
        "closed_loop_evaluated": False,
    }


def _retain_failure(storage, request, root, output, claimed, exc) -> None:
    if not claimed:
        return
    _write_json(
        root / "failure.json", {"status": "failed", "error_type": type(exc).__name__}
    )
    try:
        storage.upload_directory(str(output), request.output_path)
        storage.upload_file(
            str(root / "failure.json"), request.output_path + "/failure.json"
        )
    except Exception as upload_error:
        # Preserve the original failure; the local directory is deliberately retained.
        exc.add_note(
            f"Failure artifact upload also failed: {type(upload_error).__name__}; retained at {root}"
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _storage_client() -> StorageClient:
    credentials = load_credentials(export_to_environment=True)
    return StorageClient.from_environment(
        endpoint_url=credentials.s3_endpoint,
        aws_access_key_id=credentials.s3_access_key_id,
        aws_secret_access_key=credentials.s3_secret_access_key,
    )
