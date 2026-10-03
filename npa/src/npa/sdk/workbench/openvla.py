"""Native OpenVLA-OFT SDK stages.

Every method delegates to the OFT pipeline. The training/rollout path rejects
stock OpenVLA-only checkpoints because their decoder lacks OFT's continuous
action head and proprioception projector.
"""

from __future__ import annotations

from typing import Any

from npa.workflows.byof import openvla_pipeline as pipeline


def prepare(
    *, dataset_uri: str, dataset_name: str, task_suite: str, output_uri: str
) -> dict[str, Any]:
    """Inspect RLDS data and publish hash-bound normalization provenance."""
    return pipeline.prepare(
        pipeline.PrepareConfig(dataset_uri, output_uri, dataset_name, task_suite)
    )


def train(
    *,
    prepared_manifest_uri: str,
    output_uri: str,
    runtime_root: str,
    model_id: str = pipeline.DEFAULT_MODEL_ID,
    model_revision: str = pipeline.MODEL_REVISION,
    batch_size: int = pipeline.DEFAULT_BATCH_SIZE,
    max_steps: int = pipeline.DEFAULT_MAX_STEPS,
    learning_rate: float = pipeline.DEFAULT_LEARNING_RATE,
    lora_rank: int = pipeline.DEFAULT_LORA_RANK,
    processes: int = 8,
    seed: int = pipeline.DEFAULT_SEED,
) -> dict[str, Any]:
    """Run upstream OpenVLA-OFT continuous-action fine-tuning."""
    return pipeline.train(pipeline.TrainConfig(**locals()))


def rollout(
    *,
    training_manifest_uri: str,
    output_uri: str,
    runtime_root: str,
    task_suite: str,
    trials_per_task: int = pipeline.DEFAULT_TRIALS,
    seed: int = pipeline.DEFAULT_SEED,
) -> dict[str, Any]:
    """Run upstream closed-loop LIBERO rollouts and keep MP4 evidence."""
    return pipeline.rollout(pipeline.RolloutConfig(**locals()))


def evaluate(*, rollout_manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Verify raw rollout counts and calculate held-out success statistics."""
    return pipeline.evaluate(pipeline.EvaluateConfig(**locals()))


def visualize(*, evaluation_manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Produce factual CSV/SVG comparison artifacts."""
    return pipeline.visualize(pipeline.VisualizeConfig(**locals()))
