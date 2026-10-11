"""Load the validation-selected LoRA policy with its exact data and saved assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from npa.workflows.byof import openpi_pipeline as pipeline
from npa.workflows.byof.openpi_service_diagnostics import diagnostic_path


def _selected_lineage(root: Path) -> tuple[dict, dict, dict, int]:
    training = pipeline._read_json_uri(os.environ["NPA_OPENPI_TRAINING_URI"])
    evaluation = pipeline._read_json_uri(os.environ["NPA_OPENPI_EVALUATION_URI"])
    manifest = json.loads((root / "manifest.json").read_text())
    lineage = evaluation["lineage"]
    checkpoint = training["checkpoint"]
    expected_uri = os.environ["OPENPI_CHECKPOINT_URI"].rstrip("/")
    hashes = {
        manifest["content_manifest_sha256"],
        checkpoint["content_manifest_sha256"],
        lineage["trained_checkpoint_manifest_sha256"],
    }
    step = lineage["saved_step"]
    if (
        training.get("status") != "passed"
        or evaluation.get("status") != "passed"
        or not lineage.get("exact_training_checkpoint_consumed")
        or lineage["training_artifact_uri"] != os.environ["NPA_OPENPI_TRAINING_URI"]
        or lineage["trained_checkpoint_uri"].rstrip("/") != expected_uri
        or checkpoint["uri"].rstrip("/") != expected_uri
        or len(hashes) != 1
        or not isinstance(step, int)
        or isinstance(step, bool)
        or step < 1
        or step != checkpoint["saved_step"]
        or not checkpoint.get("reload_passed")
    ):
        raise pipeline.OpenPIPipelineError(
            "selected policy training/evaluation lineage mismatch"
        )
    return training, evaluation, manifest, step


def _selected_configuration(root: Path, training: dict, evaluation: dict, step: int):
    arrays, dataset = pipeline._load_dataset(
        os.environ["NPA_OPENPI_DATASET_URI"],
        os.environ["NPA_OPENPI_DATASET_MANIFEST_URI"],
    )
    expected = training["dataset"]
    if (
        dataset["archive_sha256"] != expected["archive_sha256"]
        or dataset["archive_sha256"] != evaluation["lineage"]["dataset_archive_sha256"]
        or expected["uri"] != os.environ["NPA_OPENPI_DATASET_URI"]
        or expected["manifest_uri"] != os.environ["NPA_OPENPI_DATASET_MANIFEST_URI"]
        or not expected.get("split_disjoint")
    ):
        raise pipeline.OpenPIPipelineError("selected policy dataset lineage mismatch")
    config, data_config = pipeline._training_configuration(
        arrays=arrays,
        checkpoint_base_dir=root / "unused-config",
        train_steps=step,
        batch_size=1,
        seed=0,
        base_checkpoint_location=training["base_checkpoint"]["uri"],
        fsdp_devices=1,
    )
    return config, data_config, dataset


def _verify_saved_normalization(step_dir: Path, data_config: Any) -> None:
    import numpy as np
    from openpi.shared import normalize

    saved = normalize.load(step_dir / "assets" / data_config.asset_id)
    if set(saved) != set(data_config.norm_stats):
        raise pipeline.OpenPIPipelineError(
            "selected policy saved normalization keys differ"
        )
    for key, expected in data_config.norm_stats.items():
        for field in ("mean", "std", "q01", "q99"):
            if not np.array_equal(getattr(saved[key], field), getattr(expected, field)):
                raise pipeline.OpenPIPipelineError(
                    "selected policy saved normalization differs from training"
                )


def _record_loaded_policy(
    root: Path, config: Any, dataset: dict, manifest: dict, step: int
) -> None:
    path = diagnostic_path("npa-openpi-loaded-checkpoint.json")
    proof = json.loads(path.read_text())
    if proof["checkpoint_manifest_sha256"] != manifest["content_manifest_sha256"]:
        raise pipeline.OpenPIPipelineError("selected policy loaded-byte proof mismatch")
    proof["selected_policy"] = {
        "loaded": True,
        "saved_step": step,
        "config_name": config.name,
        "dataset_archive_sha256": dataset["archive_sha256"],
        "paligemma_variant": config.model.paligemma_variant,
        "action_expert_variant": config.model.action_expert_variant,
        "action_horizon": config.model.action_horizon,
        "normalization_source": "selected_step_assets_verified_against_training_only",
        "checkpoint_step_directory": str(step),
    }
    path.write_text(json.dumps(proof, sort_keys=True))


def load_selected_policy(checkpoint_root: str) -> tuple[Any, Any]:
    """Load exact selected parameters and training transforms.

    Args:
        checkpoint_root: Verified local checkpoint root containing its manifest.
    Returns:
        The loaded upstream policy and reconstructed training configuration.
    Raises:
        OpenPIPipelineError: Selection, dataset, saved assets or byte proofs differ.
    """
    from openpi.policies import policy_config

    root = Path(checkpoint_root)
    training, evaluation, manifest, step = _selected_lineage(root)
    step_dir = root / str(step)
    if not (step_dir / "params").is_dir() or not (step_dir / "assets").is_dir():
        raise pipeline.OpenPIPipelineError(
            "selected policy step lacks params or assets"
        )
    config, data_config, dataset = _selected_configuration(
        root, training, evaluation, step
    )
    _verify_saved_normalization(step_dir, data_config)
    policy = policy_config.create_trained_policy(config, step_dir)
    _record_loaded_policy(root, config, dataset, manifest, step)
    return policy, config
