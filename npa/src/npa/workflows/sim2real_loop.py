"""Concrete Sim2Real VLM-to-RL loop and end-to-end runbook runtime."""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.deploy.images import container_image_for_tool
from npa.workbench.lerobot.policy_container import (
    parse_vlm_signal_batch,
    run_vlm_signal_training_step,
)


DEFAULT_S3_ENDPOINT = ""
DEFAULT_BUCKET = ""
DEFAULT_PREFIX = "sim2real-b"
DEFAULT_VLM_IMAGE_TAG = "3.0.0"
DEFAULT_ENVGEN_TAG = "0.1.1"
DEFAULT_REFERENCE_POLICY_TAG = "0.1.1"
DEFAULT_TRAINER_TAG = "0.1.0"
DEFAULT_EVAL_TAG = "0.1.0"
DEFAULT_THRESHOLD = 0.75
DEFAULT_INNER_ITERATIONS = 2
DEFAULT_OUTER_ITERATIONS = 1
DEFAULT_LOOP_OF_LOOPS_ITERATIONS = 1
DEFAULT_ROLLOUT_COUNT = 3
DEFAULT_STEPS_PER_ROLLOUT = 4
DEFAULT_HELDOUT_ENVS = 8
DEFAULT_REFERENCE_VLM_MODEL = "npa-cosmos3-reason"
DEFAULT_LEROBOT_DATASET_ID = "lerobot/pusht"
SCHEMA_VLM_EVAL = "npa.sim2real.vlm_eval.v1"
SCHEMA_RL_SIGNAL = "npa.sim2real.rl_signal.v1"
SCHEMA_HELDOUT_REPORT = "npa.sim2real.heldout_eval.v1"
SCHEMA_THRESHOLD_DECISION = "npa.sim2real.threshold_decision.v1"
SCHEMA_E2E_REPORT = "npa.sim2real.e2e_report.v1"

ERROR_SEVERITY = {
    "collision": 0.95,
    "missed_target": 0.85,
    "unstable": 0.7,
    "late_grasp": 0.55,
    "minor_alignment": 0.3,
    "ok": 0.0,
}

CORRECTIVE_TARGETS = {
    "collision": {
        "nl_correction": "Back off from contact and retry with a shallower approach.",
        "action_delta": [-0.12, 0.0, 0.04],
    },
    "missed_target": {
        "nl_correction": "Move the end effector toward the object center before closing.",
        "action_delta": [0.12, 0.02, 0.0],
    },
    "unstable": {
        "nl_correction": "Reduce vertical speed and stabilize before release.",
        "action_delta": [0.0, 0.0, -0.08],
    },
    "late_grasp": {
        "nl_correction": "Close the gripper earlier once the object is centered.",
        "action_delta": [0.03, 0.0, 0.02],
    },
    "minor_alignment": {
        "nl_correction": "Apply a small lateral correction toward the target marker.",
        "action_delta": [0.04, 0.01, 0.0],
    },
    "ok": {
        "nl_correction": "Preserve the current action.",
        "action_delta": [0.0, 0.0, 0.0],
    },
}


class Sim2RealLoopError(Exception):
    """Raised when the Sim2Real loop cannot produce a valid artifact."""


@dataclass(frozen=True)
class ComponentRecord:
    """Status line for one workflow stage or plug point."""

    name: str
    tier: str
    evidence: str
    artifacts: dict[str, str] = field(default_factory=dict)
    next_action: str = "CONTINUE"


@dataclass(frozen=True)
class Sim2RealLoopConfig:
    """Runtime configuration for the Sim2Real Stage 7-13 loop."""

    run_id: str
    output_dir: Path | None = None
    s3_bucket: str = ""
    s3_prefix: str = DEFAULT_PREFIX
    s3_endpoint: str = DEFAULT_S3_ENDPOINT
    trigger_dataset_uri: str = ""
    trigger_dataset_id: str = DEFAULT_LEROBOT_DATASET_ID
    action_rollouts_uri: str = ""
    train_envs_uri: str = ""
    heldout_envs_uri: str = ""
    assets_uri: str = ""
    scene_spec_uri: str = ""
    augment_image: str = f"npa-sim2real-envgen:{DEFAULT_ENVGEN_TAG}"
    policy_image: str = f"npa-sim2real-reference-policy:{DEFAULT_REFERENCE_POLICY_TAG}"
    trainer_image: str = f"npa-lerobot-vlm-rl:{DEFAULT_TRAINER_TAG}"
    vlm_image: str = f"npa-cosmos3-reason:{DEFAULT_VLM_IMAGE_TAG}"
    eval_image: str = f"npa-sim2real-eval:{DEFAULT_EVAL_TAG}"
    vlm_model: str = DEFAULT_REFERENCE_VLM_MODEL
    threshold: float = DEFAULT_THRESHOLD
    inner_iterations: int = DEFAULT_INNER_ITERATIONS
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS
    loop_of_loops_iterations: int = DEFAULT_LOOP_OF_LOOPS_ITERATIONS
    rollout_count: int = DEFAULT_ROLLOUT_COUNT
    steps_per_rollout: int = DEFAULT_STEPS_PER_ROLLOUT
    heldout_env_count: int = DEFAULT_HELDOUT_ENVS
    seed: int = 42
    upload_artifacts: bool = False
    no_guardrails: bool = False
    signal_loss_weight: float = 1.0
    learning_rate: float = 0.05
    byo_signal_converter: str = ""
    byo_trainer_command: str = ""
    byo_vlm_command: str = ""
    byo_eval_command: str = ""

    def validate(self) -> None:
        if not self.run_id:
            raise Sim2RealLoopError("run_id must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise Sim2RealLoopError(
                f"threshold must be in [0, 1], got {self.threshold}"
            )
        if self.inner_iterations <= 0:
            raise Sim2RealLoopError("inner_iterations must be positive")
        if self.outer_iterations <= 0:
            raise Sim2RealLoopError("outer_iterations must be positive")
        if self.loop_of_loops_iterations <= 0:
            raise Sim2RealLoopError("loop_of_loops_iterations must be positive")
        if self.rollout_count <= 0:
            raise Sim2RealLoopError("rollout_count must be positive")
        if self.steps_per_rollout <= 0:
            raise Sim2RealLoopError("steps_per_rollout must be positive")
        if self.heldout_env_count <= 0:
            raise Sim2RealLoopError("heldout_env_count must be positive")
        if self.learning_rate <= 0:
            raise Sim2RealLoopError("learning_rate must be positive")
        if self.signal_loss_weight < 0:
            raise Sim2RealLoopError("signal_loss_weight must be non-negative")


def new_run_id(prefix: str = "sim2real-b") -> str:
    """Return a run id suitable for S3 prefixes and local artifact paths."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def default_vlm_image(*, registry: str | None = None) -> str:
    """Return the reference Cosmos3-reason-compatible VLM image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("cosmos3-reason", registry=registry)
    return f"npa-cosmos3-reason:{DEFAULT_VLM_IMAGE_TAG}"


def default_envgen_image(*, registry: str | None = None) -> str:
    """Return the reference env-generation image used by Stages 3-6."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-envgen", registry=registry)
    return f"npa-sim2real-envgen:{DEFAULT_ENVGEN_TAG}"


def default_policy_image(*, registry: str | None = None) -> str:
    """Return the reference action-generation policy image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-reference-policy", registry=registry)
    return f"npa-sim2real-reference-policy:{DEFAULT_REFERENCE_POLICY_TAG}"


def default_trainer_image(*, registry: str | None = None) -> str:
    """Return the reference VLM-signal LeRobot trainer image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("lerobot-vlm-rl", registry=registry)
    return f"npa-lerobot-vlm-rl:{DEFAULT_TRAINER_TAG}"


def default_eval_image(*, registry: str | None = None) -> str:
    """Return the reference held-out eval harness image."""

    if registry or os.environ.get("NPA_REGISTRY"):
        return container_image_for_tool("sim2real-eval", registry=registry)
    return f"npa-sim2real-eval:{DEFAULT_EVAL_TAG}"


def build_config_from_env(**overrides: Any) -> Sim2RealLoopConfig:
    """Build a Sim2Real loop config from explicit values and env fallbacks."""

    run_id = str(
        overrides.get("run_id") or os.environ.get("NPA_SIM2REAL_RUN_ID") or new_run_id()
    )
    bucket = str(
        overrides.get("s3_bucket")
        or os.environ.get("NPA_SIM2REAL_BUCKET")
        or os.environ.get("NPA_S3_BUCKET")
        or os.environ.get("S3_BUCKET")
        or ""
    )
    registry = os.environ.get("NPA_REGISTRY", "")
    if "s3_prefix" in overrides and overrides.get("s3_prefix") is not None:
        s3_prefix = str(overrides["s3_prefix"])
    elif "NPA_SIM2REAL_PREFIX" in os.environ:
        s3_prefix = os.environ.get("NPA_SIM2REAL_PREFIX", "")
    else:
        s3_prefix = DEFAULT_PREFIX
    action_rollouts_uri = str(
        overrides.get("action_rollouts_uri")
        or os.environ.get("ACTION_ROLLOUTS_URI")
        or (f"s3://{bucket}/sim2real-a-*/actions/train/" if bucket else "")
    )
    return Sim2RealLoopConfig(
        run_id=run_id,
        output_dir=Path(overrides["output_dir"])
        if overrides.get("output_dir")
        else None,
        s3_bucket=bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=str(
            overrides.get("s3_endpoint")
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("S3_ENDPOINT_URL")
            or DEFAULT_S3_ENDPOINT
        ),
        trigger_dataset_uri=str(
            overrides.get("trigger_dataset_uri")
            or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
            or os.environ.get("TRIGGER_DATASET_URI")
            or (f"s3://{bucket}/sim2real-triggers/{run_id}/" if bucket else "")
        ),
        trigger_dataset_id=str(
            overrides.get("trigger_dataset_id")
            or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
            or os.environ.get("TRIGGER_DATASET_ID")
            or DEFAULT_LEROBOT_DATASET_ID
        ),
        action_rollouts_uri=action_rollouts_uri,
        train_envs_uri=str(
            overrides.get("train_envs_uri") or os.environ.get("TRAIN_ENVS_URI") or ""
        ),
        heldout_envs_uri=str(
            overrides.get("heldout_envs_uri")
            or os.environ.get("HELDOUT_ENVS_URI")
            or ""
        ),
        assets_uri=str(
            overrides.get("assets_uri") or os.environ.get("ASSETS_URI") or ""
        ),
        scene_spec_uri=str(
            overrides.get("scene_spec_uri") or os.environ.get("SCENE_SPEC_URI") or ""
        ),
        augment_image=str(
            overrides.get("augment_image")
            or os.environ.get("AUGMENT_IMAGE")
            or default_envgen_image(registry=registry or None)
        ),
        policy_image=str(
            overrides.get("policy_image")
            or os.environ.get("POLICY_IMAGE")
            or default_policy_image(registry=registry or None)
        ),
        trainer_image=str(
            overrides.get("trainer_image")
            or os.environ.get("TRAINER_IMAGE")
            or default_trainer_image(registry=registry or None)
        ),
        vlm_image=str(
            overrides.get("vlm_image")
            or os.environ.get("VLM_IMAGE")
            or default_vlm_image(registry=registry or None)
        ),
        eval_image=str(
            overrides.get("eval_image")
            or os.environ.get("EVAL_IMAGE")
            or default_eval_image(registry=registry or None)
        ),
        vlm_model=str(
            overrides.get("vlm_model")
            or os.environ.get("VLM_MODEL")
            or DEFAULT_REFERENCE_VLM_MODEL
        ),
        threshold=float(
            overrides.get(
                "threshold", os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD)
            )
        ),
        inner_iterations=int(
            overrides.get(
                "inner_iterations",
                os.environ.get("INNER_ITERATIONS", DEFAULT_INNER_ITERATIONS),
            )
        ),
        outer_iterations=int(
            overrides.get(
                "outer_iterations",
                os.environ.get("OUTER_ITERATIONS", DEFAULT_OUTER_ITERATIONS),
            )
        ),
        loop_of_loops_iterations=int(
            overrides.get(
                "loop_of_loops_iterations",
                os.environ.get(
                    "LOOP_OF_LOOPS_ITERATIONS", DEFAULT_LOOP_OF_LOOPS_ITERATIONS
                ),
            )
        ),
        rollout_count=int(
            overrides.get(
                "rollout_count", os.environ.get("ROLLOUT_COUNT", DEFAULT_ROLLOUT_COUNT)
            )
        ),
        steps_per_rollout=int(
            overrides.get(
                "steps_per_rollout",
                os.environ.get("STEPS_PER_ROLLOUT", DEFAULT_STEPS_PER_ROLLOUT),
            )
        ),
        heldout_env_count=int(
            overrides.get(
                "heldout_env_count",
                os.environ.get("HELDOUT_ENV_COUNT", DEFAULT_HELDOUT_ENVS),
            )
        ),
        seed=int(overrides.get("seed", os.environ.get("SEED", "42"))),
        upload_artifacts=_bool_value(
            overrides.get("upload_artifacts", os.environ.get("UPLOAD_ARTIFACTS", "0"))
        ),
        no_guardrails=_bool_value(
            overrides.get("no_guardrails", os.environ.get("NO_GUARDRAILS", "0"))
        ),
        signal_loss_weight=float(
            overrides.get(
                "signal_loss_weight", os.environ.get("SIGNAL_LOSS_WEIGHT", "1.0")
            )
        ),
        learning_rate=float(
            overrides.get("learning_rate", os.environ.get("LEARNING_RATE", "0.05"))
        ),
        byo_signal_converter=str(
            overrides.get("byo_signal_converter")
            or os.environ.get("BYO_SIGNAL_CONVERTER")
            or ""
        ),
        byo_trainer_command=str(
            overrides.get("byo_trainer_command")
            or os.environ.get("BYO_TRAINER_COMMAND")
            or ""
        ),
        byo_vlm_command=str(
            overrides.get("byo_vlm_command") or os.environ.get("BYO_VLM_COMMAND") or ""
        ),
        byo_eval_command=str(
            overrides.get("byo_eval_command")
            or os.environ.get("BYO_EVAL_COMMAND")
            or ""
        ),
    )


def artifact_uris(config: Sim2RealLoopConfig) -> dict[str, str]:
    """Return canonical S3 artifact URIs for the full 13-stage workflow."""

    if not config.s3_bucket:
        return {}
    root = _artifact_root_uri(config)
    return {
        "root": f"{root}/",
        "trigger_dataset": config.trigger_dataset_uri,
        "stage_01_trigger": f"{root}/stage_01_trigger/trigger.json",
        "stage_02_assets_stub": f"{root}/stage_02_assets/external_stub.json",
        "stage_03_augment": f"{root}/augment/manifest.json",
        "stage_04_envs_raw": f"{root}/envs/raw/",
        "stage_05_envs_train": f"{root}/envs/train/",
        "stage_06_tokens": f"{root}/tokens/manifest.json",
        "stage_07_actions_train": f"{root}/actions/train/",
        "stage_08_vlm_eval_train": f"{root}/vlm_eval/train/",
        "stage_09_training_signal": f"{root}/training_signal/train/",
        "inner_loop": f"{root}/inner_loop/",
        "stage_10_eval_heldout": f"{root}/eval/heldout/report.json",
        "stage_11_outer_loop": f"{root}/outer_loop/decision.json",
        "candidate_checkpoint": f"{root}/checkpoints/candidate/",
        "stage_12_external_validation_stub": f"{root}/stage_12_external_validation/external_stub.json",
        "stage_13_retrigger": f"{root}/stage_13_retrigger/retrigger.json",
        "report": f"{root}/reports/sim2real-report.json",
    }


def byo_seams(config: Sim2RealLoopConfig) -> dict[str, Any]:
    """Return every runtime-configurable plug point."""

    return {
        "s3_endpoint": config.s3_endpoint,
        "s3_bucket": config.s3_bucket,
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "assets_uri": config.assets_uri,
        "scene_spec_uri": config.scene_spec_uri,
        "augment_image": config.augment_image,
        "action_rollouts_uri": config.action_rollouts_uri,
        "train_envs_uri": config.train_envs_uri,
        "heldout_envs_uri": config.heldout_envs_uri,
        "policy_image": config.policy_image,
        "trainer_image": config.trainer_image,
        "vlm_image": config.vlm_image,
        "eval_image": config.eval_image,
        "threshold": config.threshold,
        "inner_iterations": config.inner_iterations,
        "outer_iterations": config.outer_iterations,
        "loop_of_loops_iterations": config.loop_of_loops_iterations,
        "byo_signal_converter": config.byo_signal_converter,
        "byo_trainer_command": config.byo_trainer_command,
        "byo_vlm_command": config.byo_vlm_command,
        "byo_eval_command": config.byo_eval_command,
    }


def run_full_loop(
    config: Sim2RealLoopConfig,
    *,
    upload: bool | None = None,
) -> dict[str, Any]:
    """Run the full local/executable Sim2Real loop and write all artifacts."""

    config.validate()
    local_dir = config.output_dir or Path(
        tempfile.mkdtemp(prefix=f"npa-{config.run_id}-")
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    components: list[ComponentRecord] = []
    stage_artifacts: dict[str, str] = {}
    stage_records: list[dict[str, Any]] = []

    stage_records.append(
        _write_stage(local_dir, 1, "trigger", _trigger_payload(config))
    )
    components.append(
        ComponentRecord(
            "stage_01_trigger",
            "WORKS",
            "Consumed the dedicated LeRobot dataset trigger path and resolved runtime plug points.",
            {"local": str(local_dir / "stage_01_trigger" / "trigger.json")},
        )
    )

    stage_records.append(
        _write_stage(
            local_dir,
            2,
            "assets",
            {
                "schema": "npa.sim2real.external_stub.v1",
                "stage": 2,
                "name": "external real assets and SceneSpec",
                "status": "documented_external_stub",
                "assets_uri": config.assets_uri,
                "scene_spec_uri": config.scene_spec_uri,
                "next_action": "CONTINUE",
            },
            filename="external_stub.json",
        )
    )
    components.append(
        ComponentRecord(
            "stage_02_assets",
            "SEAM",
            "External assets and SceneSpec are documented BYO inputs for this reference run.",
            {"local": str(local_dir / "stage_02_assets" / "external_stub.json")},
        )
    )

    stage_records.append(
        _write_json_artifact(
            local_dir / "augment" / "manifest.json",
            {
                "schema": "npa.sim2real.augment_manifest.v1",
                "stage": 3,
                "augment": "cosmos2-transfer",
                "image": config.augment_image or "reference-cosmos2-transfer",
                "assets_uri": config.assets_uri,
                "output_uri": "augment/",
                "status": "reference_manifest",
            },
        )
    )
    components.append(
        ComponentRecord(
            "stage_03_augment",
            "WORKS",
            "Wrote a Cosmos transfer augmentation manifest with BYO image override support.",
            {"local": str(local_dir / "augment" / "manifest.json")},
        )
    )

    raw_envs = _write_env_manifest(
        local_dir / "envs" / "raw",
        count=config.rollout_count + config.heldout_env_count,
        seed=config.seed,
    )
    train_envs, heldout_envs = _write_train_heldout_split(
        local_dir / "envs",
        raw_envs=raw_envs,
        train_count=config.rollout_count,
        heldout_count=config.heldout_env_count,
        seed=config.seed,
    )
    stage_records.extend([raw_envs, train_envs, heldout_envs])
    components.append(
        ComponentRecord(
            "stage_04_06_env_gen_split_tokens",
            "WORKS",
            "Generated deterministic raw env specs, train/heldout split, and token manifest.",
            {
                "raw_envs": str(local_dir / "envs" / "raw" / "manifest.json"),
                "train_envs": str(local_dir / "envs" / "train" / "manifest.json"),
                "heldout_envs": str(local_dir / "envs" / "heldout" / "manifest.json"),
                "tokens": str(local_dir / "tokens" / "manifest.json"),
            },
        )
    )
    _write_json_artifact(
        local_dir / "tokens" / "manifest.json",
        {
            "schema": "npa.sim2real.tokens.v1",
            "stage": 6,
            "source": "stage-a-compatible-reference",
            "train_env_count": len(train_envs["payload"]["envs"]),
            "heldout_env_count": len(heldout_envs["payload"]["envs"]),
            "status": "ready",
        },
    )

    outer_history: list[dict[str, Any]] = []
    final_inner: dict[str, Any] | None = None
    final_eval: dict[str, Any] | None = None
    final_decision: dict[str, Any] | None = None
    quality = 0.36 + rng.random() * 0.04
    for outer_iteration in range(1, config.outer_iterations + 1):
        inner = run_inner_loop(
            config,
            local_dir=local_dir,
            initial_quality=quality,
            outer_iteration=outer_iteration,
        )
        final_inner = inner
        quality = float(inner["final_quality"])
        heldout_report = run_heldout_eval(
            config,
            local_dir=local_dir,
            inner_evidence=inner,
            outer_iteration=outer_iteration,
        )
        final_eval = heldout_report
        decision = threshold_decision(
            config,
            local_dir=local_dir,
            heldout_report=heldout_report,
            outer_iteration=outer_iteration,
        )
        final_decision = decision
        outer_history.append(
            {
                "outer_iteration": outer_iteration,
                "inner_loop": inner["evidence_uri"],
                "heldout_report": heldout_report["report_uri"],
                "decision": decision,
            }
        )
        if decision["decision"] == "promote_checkpoint":
            break
        quality = min(0.95, quality + 0.12)

    if final_decision is None or final_inner is None or final_eval is None:
        raise Sim2RealLoopError("full loop did not execute an outer iteration")

    stage_records.append(
        _write_stage(
            local_dir,
            12,
            "external_validation",
            {
                "schema": "npa.sim2real.external_stub.v1",
                "stage": 12,
                "name": "external real-world validation",
                "status": "documented_external_stub",
                "input_checkpoint": final_decision["checkpoint_uri"],
                "next_action": "CONTINUE",
            },
            filename="external_stub.json",
        )
    )
    components.append(
        ComponentRecord(
            "stage_12_external_validation",
            "SEAM",
            "External real-world validation is a documented BYO gate; loop-of-loops continues through Stage 13.",
            {
                "local": str(
                    local_dir / "stage_12_external_validation" / "external_stub.json"
                )
            },
        )
    )

    retrigger = {
        "schema": "npa.sim2real.retrigger.v1",
        "stage": 13,
        "run_id": config.run_id,
        "source_decision": final_decision["decision"],
        "loop_of_loops_iteration": 1,
        "max_loop_of_loops_iterations": config.loop_of_loops_iterations,
        "target_stage": 1,
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "retrigger_condition": "real_world_lerobot_dataset_landed",
        "should_retrigger": config.loop_of_loops_iterations > 1,
    }
    stage_records.append(
        _write_json_artifact(
            local_dir / "stage_13_retrigger" / "retrigger.json", retrigger
        )
    )
    components.append(
        ComponentRecord(
            "stage_13_retrigger",
            "WORKS",
            "Wrote loop-of-loops retrigger record with max-iteration cap.",
            {"local": str(local_dir / "stage_13_retrigger" / "retrigger.json")},
        )
    )

    components.extend(
        [
            ComponentRecord(
                "vlm_byo_seam",
                "WORKS",
                "VLM image/command are runtime-configurable; default reference is npa-cosmos3-reason.",
                {"image": config.vlm_image},
            ),
            ComponentRecord(
                "trainer_byo_seam",
                "WORKS",
                "Trainer image/command are runtime-configurable; default reference consumes npa.sim2real.rl_signal.v1.",
                {"image": config.trainer_image},
            ),
            ComponentRecord(
                "eval_byo_seam",
                "WORKS",
                "Held-out eval image/command and threshold are runtime-configurable.",
                {"image": config.eval_image},
            ),
        ]
    )

    report = {
        "schema": SCHEMA_E2E_REPORT,
        "run_id": config.run_id,
        "status": "completed",
        "created_at": _utc_now(),
        "local_artifact_dir": str(local_dir),
        "s3_artifacts": artifact_uris(config),
        "config": _redacted_config(config),
        "byo_seams": byo_seams(config),
        "components": [asdict(component) for component in components],
        "stage_records": stage_records,
        "inner_loop": final_inner,
        "outer_loop": {
            "history": outer_history,
            "latest_heldout_report": final_eval,
            "latest_decision": final_decision,
        },
        "image_completeness": {
            "required": [
                config.augment_image,
                config.policy_image,
                config.vlm_image,
                config.trainer_image,
                config.eval_image,
            ],
            "all_referenced": all(
                [
                    config.augment_image,
                    config.policy_image,
                    config.vlm_image,
                    config.trainer_image,
                    config.eval_image,
                ]
            ),
        },
    }
    report_path = local_dir / "reports" / "sim2real-report.json"
    _write_json_artifact(report_path, report)
    stage_artifacts["report"] = str(report_path)
    upload_enabled = config.upload_artifacts if upload is None else upload
    if upload_enabled and config.s3_bucket:
        report["upload"] = upload_run_artifacts(config, local_dir)
        _write_json_artifact(report_path, report)
    else:
        report["upload"] = {
            "status": "skipped",
            "reason": "upload_artifacts is false or no s3_bucket configured",
        }
        _write_json_artifact(report_path, report)
    return report


def run_inner_loop(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    initial_quality: float,
    outer_iteration: int = 1,
) -> dict[str, Any]:
    """Run action generation, VLM eval, signal conversion, and policy update."""

    iteration_records: list[dict[str, Any]] = []
    reward_trend: list[float] = []
    policy_deltas: list[float] = []
    quality = float(initial_quality)
    reward_head = 0.0
    action_bias = 0.0
    for iteration in range(1, config.inner_iterations + 1):
        actions_dir = (
            local_dir
            / "actions"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        rollouts = generate_action_rollouts(
            actions_dir,
            count=config.rollout_count,
            steps_per_rollout=config.steps_per_rollout,
            seed=config.seed + outer_iteration * 100 + iteration,
            quality=quality,
        )
        eval_dir = (
            local_dir
            / "vlm_eval"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        signal_dir = (
            local_dir
            / "training_signal"
            / "train"
            / f"outer-{outer_iteration:02d}"
            / f"iter-{iteration:02d}"
        )
        evals: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []
        for rollout in rollouts:
            evaluation = evaluate_rollout_with_vlm(
                rollout,
                output_dir=eval_dir,
                config=config,
            )
            signal = convert_vlm_eval_to_rl_signal(evaluation)
            _write_json_artifact(signal_dir / f"{signal['rollout_id']}.json", signal)
            evals.append(evaluation)
            signals.append(signal)
        signal_batch_path = (
            local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / f"signals-iter-{iteration:02d}.json"
        )
        _write_json_artifact(
            signal_batch_path, {"schema": SCHEMA_RL_SIGNAL, "signals": signals}
        )
        parsed_signals = parse_vlm_signal_batch({"signals": signals})
        update = run_vlm_signal_training_step(
            parsed_signals,
            output_dir=local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "trainer"
            / f"iter-{iteration:02d}",
            learning_rate=config.learning_rate,
            signal_loss_weight=config.signal_loss_weight,
            initial_reward_head=reward_head,
            initial_action_bias=action_bias,
        )
        control = run_vlm_signal_training_step(
            parsed_signals,
            output_dir=local_dir
            / "inner_loop"
            / f"outer-{outer_iteration:02d}"
            / "control"
            / f"iter-{iteration:02d}",
            learning_rate=config.learning_rate,
            signal_loss_weight=config.signal_loss_weight,
            initial_reward_head=reward_head,
            initial_action_bias=action_bias,
            control=True,
        )
        reward_head = update.reward_head_after
        action_bias = (
            update.policy_output_after[0] if update.policy_output_after else action_bias
        )
        mean_reward = round(
            sum(_signal_mean_reward(signal) for signal in signals)
            / float(len(signals)),
            6,
        )
        reward_trend.append(mean_reward)
        delta_vs_control = max(0.0, update.policy_delta_l2 - control.policy_delta_l2)
        policy_deltas.append(round(delta_vs_control, 8))
        quality = min(
            0.98, quality + max(0.06, min(0.18, delta_vs_control * 2.0 + 0.07))
        )
        iteration_records.append(
            {
                "iteration": iteration,
                "actions_dir": str(actions_dir),
                "vlm_eval_dir": str(eval_dir),
                "signal_dir": str(signal_dir),
                "signal_batch": str(signal_batch_path),
                "mean_reward": mean_reward,
                "update": update.to_dict(),
                "no_signal_control": control.to_dict(),
                "policy_delta_vs_control": round(delta_vs_control, 8),
                "next_rollout_quality": round(quality, 6),
                "sample_vlm_eval": evals[0],
                "sample_signal": signals[0],
            }
        )

    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "outer_iteration": outer_iteration,
        "status": "closed",
        "reward_trend": reward_trend,
        "policy_delta_vs_no_signal_control": policy_deltas,
        "attribution": (
            "The reference update and no-signal control share initial adapter state. "
            "Only the VLM-derived rewards, advantages, and corrective targets produce the policy-output delta."
        ),
        "iterations": iteration_records,
        "final_quality": round(quality, 6),
    }
    evidence_path = (
        local_dir / "inner_loop" / f"outer-{outer_iteration:02d}" / "evidence.json"
    )
    _write_json_artifact(evidence_path, evidence)
    return {**evidence, "evidence_uri": str(evidence_path)}


def generate_action_rollouts(
    output_dir: Path,
    *,
    count: int,
    steps_per_rollout: int,
    seed: int,
    quality: float,
) -> list[Path]:
    """Generate small action-conditioned rollout fixtures with camera frames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rollouts: list[Path] = []
    for index in range(count):
        rollout_id = f"rollout-{index:04d}"
        rollout_dir = output_dir / rollout_id
        rollout_dir.mkdir(parents=True, exist_ok=True)
        actions: list[dict[str, Any]] = []
        for step in range(steps_per_rollout):
            drift = max(0.0, 1.0 - quality) * (1.0 + rng.random() * 0.2)
            action = [
                round(quality * 0.1 + rng.uniform(-0.02, 0.02), 5),
                round((0.5 - drift) * 0.1 + rng.uniform(-0.02, 0.02), 5),
                round((quality - 0.5) * 0.1 + rng.uniform(-0.02, 0.02), 5),
            ]
            actions.append({"step": step, "action": action})
            _write_ppm(
                rollout_dir / f"camera-{step:03d}.ppm",
                red=int(64 + 120 * quality),
                green=int(40 + 80 * (1.0 - drift)),
                blue=int(80 + step * 12),
            )
        _write_json_artifact(
            rollout_dir / "manifest.json",
            {
                "schema": "npa.sim2real.action_rollout.v1",
                "rollout_id": rollout_id,
                "quality": round(quality, 6),
                "steps": steps_per_rollout,
                "camera_observations": [
                    f"camera-{step:03d}.ppm" for step in range(steps_per_rollout)
                ],
                "actions": actions,
            },
        )
        rollouts.append(rollout_dir)
    return rollouts


def evaluate_rollout_with_vlm(
    rollout_dir: Path,
    *,
    output_dir: Path,
    config: Sim2RealLoopConfig,
) -> dict[str, Any]:
    """Invoke the configured VLM component and parse its structured judgment."""

    manifest_path = rollout_dir / "manifest.json"
    if not manifest_path.exists():
        raise Sim2RealLoopError(f"rollout manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rollout_id = str(manifest.get("rollout_id") or rollout_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{rollout_id}.json"
    env = _component_env(
        config,
        component="vlm_eval",
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_ROLLOUT_DIR": str(rollout_dir),
            "NPA_SIM2REAL_ROLLOUT_ID": rollout_id,
            "NPA_SIM2REAL_ROLLOUT_MANIFEST": str(manifest_path),
            "NPA_SIM2REAL_VLM_MODEL": config.vlm_model,
            "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
            "NPA_SIM2REAL_VLM_IMAGE": config.vlm_image,
        },
    )
    if config.byo_vlm_command.strip():
        invocation = _run_component_command(
            config.byo_vlm_command,
            cwd=rollout_dir,
            env=env,
            component="vlm_eval",
        )
    else:
        invocation = _run_image_component(
            config.vlm_image,
            component="vlm_eval",
            env=env,
            mounts=[
                (rollout_dir, "/npa/input/rollout", "ro"),
                (output_dir, "/npa/output", "rw"),
            ],
            output_json=output_path,
            container_output_json=f"/npa/output/{rollout_id}.json",
        )
    payload = _read_component_json(output_path, invocation)
    evaluation = _normalize_vlm_evaluation(
        payload,
        manifest=manifest,
        rollout_id=rollout_id,
        config=config,
        invocation=invocation,
    )
    _write_json_artifact(output_path, evaluation)
    return evaluation


def _component_env(
    config: Sim2RealLoopConfig,
    *,
    component: str,
    output_json: Path,
    extra: dict[str, str],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "NPA_SIM2REAL_COMPONENT": component,
            "NPA_SIM2REAL_RUN_ID": config.run_id,
            "NPA_SIM2REAL_OUTPUT_JSON": str(output_json),
            "NPA_SIM2REAL_S3_BUCKET": config.s3_bucket,
            "NPA_SIM2REAL_S3_PREFIX": config.s3_prefix,
            "AWS_ENDPOINT_URL": config.s3_endpoint or env.get("AWS_ENDPOINT_URL", ""),
        }
    )
    env.update(extra)
    return env


def _run_component_command(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
    component: str,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise Sim2RealLoopError(
            f"{component} command failed with exit {result.returncode}: "
            f"{_component_excerpt(result.stderr or result.stdout)}"
        )
    return {
        "mode": "command",
        "component": component,
        "command": _redact_command(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_excerpt": _component_excerpt(result.stdout),
        "stderr_excerpt": _component_excerpt(result.stderr),
    }


def _run_image_component(
    image: str,
    *,
    component: str,
    env: dict[str, str],
    mounts: list[tuple[Path, str, str]],
    output_json: Path,
    container_output_json: str,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    runtime = os.environ.get("NPA_CONTAINER_RUNTIME") or _first_available_runtime()
    if not runtime:
        raise Sim2RealLoopError(
            f"no BYO {component} command was provided and no docker/podman runtime is available for image {image}"
        )
    container_env = dict(env)
    container_env["NPA_SIM2REAL_OUTPUT_JSON"] = container_output_json
    cmd = [runtime, "run", "--rm"]
    if runtime.endswith("docker") and os.environ.get("NPA_SIM2REAL_CONTAINER_GPUS", "all") != "none":
        cmd.extend(["--gpus", os.environ.get("NPA_SIM2REAL_CONTAINER_GPUS", "all")])
    for key in sorted(container_env):
        if key.startswith("NPA_SIM2REAL") or key in {"AWS_ENDPOINT_URL", "S3_ENDPOINT_URL"}:
            cmd.extend(["-e", key])
    for host, container, mode in mounts:
        cmd.extend(["-v", f"{host}:{container}:{mode}"])
    cmd.append(image)
    result = subprocess.run(
        cmd,
        env=container_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise Sim2RealLoopError(
            f"{component} image {image} failed with exit {result.returncode}: "
            f"{_component_excerpt(result.stderr or result.stdout)}"
        )
    return {
        "mode": "image",
        "component": component,
        "image": image,
        "command": " ".join(shlex.quote(part) for part in cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_excerpt": _component_excerpt(result.stdout),
        "stderr_excerpt": _component_excerpt(result.stderr),
    }


def _first_available_runtime() -> str:
    for candidate in ("docker", "podman"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return ""


def _read_component_json(output_path: Path, invocation: dict[str, Any]) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    stdout = str(invocation.get("stdout") or "")
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    raise Sim2RealLoopError(
        f"{invocation.get('component', 'component')} did not write JSON to {output_path}"
    )


def _normalize_vlm_evaluation(
    payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    rollout_id: str,
    config: Sim2RealLoopConfig,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    if "score" not in payload:
        raise Sim2RealLoopError("VLM component output must include score")
    score = max(0.0, min(1.0, float(payload["score"])))
    success = bool(payload.get("success", score >= config.threshold))
    raw_steps = payload.get("per_step") or payload.get("steps") or []
    if not raw_steps and payload.get("critique_text"):
        raw_steps = [{"step": 0, "critique_text": payload["critique_text"], "error_tags": payload.get("error_tags", [])}]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Sim2RealLoopError("VLM component output must include non-empty per_step")
    actions = {int(item["step"]): item.get("action", []) for item in manifest.get("actions", [])}
    observations = list(manifest.get("camera_observations", []))
    per_step: list[dict[str, Any]] = []
    for raw in raw_steps:
        step = int(raw.get("step", len(per_step)))
        tags = raw.get("error_tags", raw.get("tags", [])) or ["ok"]
        if isinstance(tags, str):
            tags = [tags]
        critique = str(raw.get("critique_text") or raw.get("critique") or raw.get("text") or "")
        if not critique:
            raise Sim2RealLoopError("VLM component per_step entries must include critique text")
        camera = raw.get("camera_observation")
        if not camera and 0 <= step < len(observations):
            camera = observations[step]
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": [str(tag) for tag in tags],
                "action": raw.get("action", actions.get(step, [])),
                "camera_observation": str(camera or f"camera-{step:03d}.ppm"),
            }
        )
    return {
        "schema": SCHEMA_VLM_EVAL,
        "rollout_id": str(payload.get("rollout_id") or rollout_id),
        "success": success,
        "score": round(score, 6),
        "per_step": per_step,
        "summary": str(payload.get("summary") or payload.get("critique") or ""),
        "model": str(payload.get("model") or config.vlm_model),
        "vlm_image": config.vlm_image,
        "component_invocation": _public_invocation(invocation),
        "generated_at": _utc_now(),
    }


def _component_excerpt(text: str, limit: int = 1200) -> str:
    scrubbed = []
    for line in str(text or "").splitlines():
        if "AWS_SECRET_ACCESS_KEY" in line or "AWS_ACCESS_KEY_ID" in line:
            scrubbed.append("[redacted secret line]")
        else:
            scrubbed.append(line)
    return "\n".join(scrubbed)[-limit:]


def _redact_command(command: str) -> str:
    redacted = str(command)
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NGC_API_KEY"):
        value = os.environ.get(key)
        if value:
            redacted = redacted.replace(value, f"<{key}>")
    return redacted


def _public_invocation(invocation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in invocation.items()
        if key not in {"stdout", "stderr"}
    }


def convert_vlm_eval_to_rl_signal(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Convert structured VLM critique JSON into a dense RL signal."""

    if evaluation.get("schema") != SCHEMA_VLM_EVAL:
        raise Sim2RealLoopError(
            f"unsupported VLM eval schema: {evaluation.get('schema')}"
        )
    raw_steps = evaluation.get("per_step")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Sim2RealLoopError("VLM eval must include non-empty per_step")
    success = bool(evaluation["success"])
    step_items: list[dict[str, Any]] = []
    rewards: list[float] = []
    for raw_step in raw_steps:
        tags = [str(tag) for tag in raw_step.get("error_tags", [])] or ["ok"]
        severity = max(ERROR_SEVERITY.get(tag, 0.4) for tag in tags)
        success_bonus = 0.35 if success else -0.15
        reward = max(
            -1.0, min(1.0, success_bonus + 0.65 * (1.0 - severity) - 0.75 * severity)
        )
        rewards.append(reward)
        target = _merge_targets(tags)
        source_action = raw_step.get("action") or []
        credit = [
            round(abs(float(value)) * reward, 6)
            for value in source_action
            if isinstance(value, int | float)
        ]
        step_items.append(
            {
                "step": int(raw_step["step"]),
                "reward": round(reward, 6),
                "target": target,
                "critique_text": str(raw_step.get("critique_text") or ""),
                "error_tags": tags,
                "action_credit": {
                    "source_action": source_action,
                    "credit": credit,
                },
            }
        )
    baseline = sum(rewards) / float(len(rewards))
    for item in step_items:
        item["advantage"] = round(float(item["reward"]) - baseline, 6)
    return {
        "schema": SCHEMA_RL_SIGNAL,
        "rollout_id": str(evaluation["rollout_id"]),
        "source": "vlm",
        "success": success,
        "score": evaluation.get("score"),
        "per_step": step_items,
        "mapping_rules": signal_mapping_rules(),
    }


def signal_mapping_rules() -> dict[str, Any]:
    """Return documented VLM-critique to RL-signal conversion rules."""

    return {
        "dense_reward": (
            "reward = success_bonus + 0.65 * (1 - max_tag_severity) - "
            "0.75 * max_tag_severity, clipped to [-1, 1]."
        ),
        "success_bonus": {"success": 0.35, "failure": -0.15},
        "advantage": "per-step reward minus rollout mean reward",
        "per_action_credit": "abs(action_i) * step_reward for each source action dimension",
        "nl_corrective_targets": CORRECTIVE_TARGETS,
        "error_severity": ERROR_SEVERITY,
    }


def run_heldout_eval(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    inner_evidence: dict[str, Any],
    outer_iteration: int,
) -> dict[str, Any]:
    """Invoke the configured held-out eval component and write report.json."""

    output_dir = local_dir / "eval" / "heldout"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.json"
    inner_path = output_dir / f"inner-evidence-outer-{outer_iteration:02d}.json"
    _write_json_artifact(inner_path, inner_evidence)
    env = _component_env(
        config,
        component="heldout_eval",
        output_json=output_path,
        extra={
            "NPA_SIM2REAL_HELDOUT_ENVS_DIR": str(local_dir / "envs" / "heldout"),
            "NPA_SIM2REAL_HELDOUT_ENV_COUNT": str(config.heldout_env_count),
            "NPA_SIM2REAL_INNER_EVIDENCE_JSON": str(inner_path),
            "NPA_SIM2REAL_THRESHOLD": str(config.threshold),
            "NPA_SIM2REAL_EVAL_IMAGE": config.eval_image,
        },
    )
    if config.byo_eval_command.strip():
        invocation = _run_component_command(
            config.byo_eval_command,
            cwd=local_dir,
            env=env,
            component="heldout_eval",
        )
    else:
        invocation = _run_image_component(
            config.eval_image,
            component="heldout_eval",
            env=env,
            mounts=[
                (local_dir / "envs" / "heldout", "/npa/input/heldout_envs", "ro"),
                (output_dir, "/npa/output", "rw"),
                (inner_path, "/npa/input/inner_evidence.json", "ro"),
            ],
            output_json=output_path,
            container_output_json="/npa/output/report.json",
        )
    payload = _read_component_json(output_path, invocation)
    report = _normalize_heldout_report(
        payload,
        config=config,
        outer_iteration=outer_iteration,
        inner_evidence_uri=str(inner_path),
        invocation=invocation,
    )
    _write_json_artifact(output_path, report)
    return {**report, "report_uri": str(output_path)}


def _normalize_heldout_report(
    payload: dict[str, Any],
    *,
    config: Sim2RealLoopConfig,
    outer_iteration: int,
    inner_evidence_uri: str,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    raw_items = payload.get("per_env") or payload.get("env_scores") or payload.get("scores")
    if isinstance(raw_items, dict):
        raw_items = [
            {"env_id": key, **(value if isinstance(value, dict) else {"score": value})}
            for key, value in raw_items.items()
        ]
    if not isinstance(raw_items, list) or not raw_items:
        raise Sim2RealLoopError("held-out eval component output must include non-empty per_env/env_scores")
    per_env: list[dict[str, Any]] = []
    passed = 0
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            item = {"score": item}
        score = max(0.0, min(1.0, float(item.get("score", item.get("success_score", 0.0)))))
        success = bool(item.get("success", score >= config.threshold))
        passed += int(success)
        per_env.append(
            {
                "env_id": str(item.get("env_id") or f"heldout-{index:04d}"),
                "success": success,
                "score": round(score, 6),
                "details": item.get("details", {}),
            }
        )
    success_rate = passed / float(len(per_env))
    return {
        "schema": SCHEMA_HELDOUT_REPORT,
        "stage": 10,
        "outer_iteration": outer_iteration,
        "status": "completed",
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "per_env": per_env,
        "eval_image": config.eval_image,
        "byo_eval_command": _redact_command(config.byo_eval_command),
        "inner_evidence_uri": inner_evidence_uri,
        "component_invocation": _public_invocation(invocation),
        "generated_at": _utc_now(),
    }


def threshold_decision(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    heldout_report: dict[str, Any],
    outer_iteration: int,
) -> dict[str, Any]:
    """Apply Stage 11 threshold gate and write promote/loop-back artifacts."""

    success_rate = float(heldout_report["success_rate"])
    promoted = success_rate >= config.threshold
    checkpoint_dir = local_dir / "checkpoints" / "candidate"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_uri = str(checkpoint_dir)
    decision = {
        "schema": SCHEMA_THRESHOLD_DECISION,
        "stage": 11,
        "outer_iteration": outer_iteration,
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "decision": "promote_checkpoint" if promoted else "loop_back_to_inner_loop",
        "checkpoint_uri": checkpoint_uri,
        "max_outer_iterations": config.outer_iterations,
        "remaining_outer_iterations": max(0, config.outer_iterations - outer_iteration),
    }
    if promoted:
        _write_json_artifact(
            checkpoint_dir / "candidate.json",
            {
                "schema": "npa.sim2real.candidate_checkpoint.v1",
                "run_id": config.run_id,
                "source": "vlm-rl-reference-update",
                "heldout_success_rate": round(success_rate, 6),
                "threshold": config.threshold,
                "promoted_at": _utc_now(),
            },
        )
    else:
        _write_json_artifact(
            local_dir / "outer_loop" / "loopback.json",
            {
                "schema": "npa.sim2real.loopback.v1",
                "from_stage": 11,
                "to_stage": 7,
                "reason": "heldout threshold not met",
                "decision": decision,
            },
        )
    path = local_dir / "outer_loop" / "decision.json"
    _write_json_artifact(path, decision)
    return {**decision, "decision_uri": str(path)}


def upload_run_artifacts(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Upload the run artifact tree to S3-compatible storage."""

    if not config.s3_bucket:
        return {"status": "skipped", "reason": "s3_bucket is not configured"}
    try:
        client = StorageClient.from_environment(endpoint_url=config.s3_endpoint)
        destination = f"{_artifact_root_uri(config)}/"
        uploaded = client.upload_directory(str(local_dir), destination)
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"S3 upload failed: {exc}",
            "next_action": "CONTINUE",
        }
    return {"status": "uploaded", "uri": uploaded}


def main(argv: list[str] | None = None) -> int:
    """Module CLI for raw SkyPilot YAML and local smoke runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser(
        "full-loop", help="Run the full Stage 1-13 Sim2Real workflow."
    )
    _add_common_args(full)
    inner = subparsers.add_parser(
        "inner-loop", help="Run only the VLM-to-RL inner loop."
    )
    _add_common_args(inner)
    convert = subparsers.add_parser(
        "convert-signal", help="Convert one VLM eval JSON to RL signal JSON."
    )
    convert.add_argument("--vlm-json", type=Path, required=True)
    convert.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "convert-signal":
        payload = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        _write_json_artifact(args.output_json, convert_vlm_eval_to_rl_signal(payload))
        return 0

    config = build_config_from_env(
        run_id=args.run_id,
        output_dir=args.output_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint=args.s3_endpoint,
        trigger_dataset_uri=args.trigger_dataset_uri,
        trigger_dataset_id=args.trigger_dataset_id,
        action_rollouts_uri=args.action_rollouts_uri,
        train_envs_uri=args.train_envs_uri,
        heldout_envs_uri=args.heldout_envs_uri,
        assets_uri=args.assets_uri,
        scene_spec_uri=args.scene_spec_uri,
        augment_image=args.augment_image,
        policy_image=args.policy_image,
        trainer_image=args.trainer_image,
        vlm_image=args.vlm_image,
        eval_image=args.eval_image,
        vlm_model=args.vlm_model,
        threshold=args.threshold,
        inner_iterations=args.inner_iterations,
        outer_iterations=args.outer_iterations,
        loop_of_loops_iterations=args.loop_of_loops_iterations,
        rollout_count=args.rollout_count,
        steps_per_rollout=args.steps_per_rollout,
        heldout_env_count=args.heldout_env_count,
        seed=args.seed,
        upload_artifacts=args.upload_artifacts,
        no_guardrails=args.no_guardrails,
        signal_loss_weight=args.signal_loss_weight,
        learning_rate=args.learning_rate,
        byo_signal_converter=args.byo_signal_converter,
        byo_trainer_command=args.byo_trainer_command,
        byo_vlm_command=args.byo_vlm_command,
        byo_eval_command=args.byo_eval_command,
    )
    if args.command == "full-loop":
        report = run_full_loop(config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "inner-loop":
        config.validate()
        local_dir = config.output_dir or Path(
            tempfile.mkdtemp(prefix=f"npa-{config.run_id}-")
        )
        evidence = run_inner_loop(config, local_dir=local_dir, initial_quality=0.38)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    return 2


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id", default=os.environ.get("NPA_SIM2REAL_RUN_ID", new_run_id())
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("NPA_SIM2REAL_BUCKET", os.environ.get("S3_BUCKET", "")),
    )
    parser.add_argument(
        "--s3-prefix", default=os.environ.get("NPA_SIM2REAL_PREFIX", DEFAULT_PREFIX)
    )
    parser.add_argument(
        "--s3-endpoint", default=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_S3_ENDPOINT)
    )
    parser.add_argument(
        "--trigger-dataset-uri",
        default=os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI", ""),
    )
    parser.add_argument(
        "--trigger-dataset-id",
        default=os.environ.get(
            "NPA_SIM2REAL_TRIGGER_DATASET_ID", DEFAULT_LEROBOT_DATASET_ID
        ),
    )
    parser.add_argument(
        "--action-rollouts-uri", default=os.environ.get("ACTION_ROLLOUTS_URI", "")
    )
    parser.add_argument(
        "--train-envs-uri", default=os.environ.get("TRAIN_ENVS_URI", "")
    )
    parser.add_argument(
        "--heldout-envs-uri", default=os.environ.get("HELDOUT_ENVS_URI", "")
    )
    parser.add_argument("--assets-uri", default=os.environ.get("ASSETS_URI", ""))
    parser.add_argument(
        "--scene-spec-uri", default=os.environ.get("SCENE_SPEC_URI", "")
    )
    parser.add_argument("--augment-image", default=os.environ.get("AUGMENT_IMAGE", ""))
    parser.add_argument("--policy-image", default=os.environ.get("POLICY_IMAGE", ""))
    parser.add_argument("--trainer-image", default=os.environ.get("TRAINER_IMAGE", ""))
    parser.add_argument("--vlm-image", default=os.environ.get("VLM_IMAGE", ""))
    parser.add_argument("--eval-image", default=os.environ.get("EVAL_IMAGE", ""))
    parser.add_argument(
        "--vlm-model", default=os.environ.get("VLM_MODEL", DEFAULT_REFERENCE_VLM_MODEL)
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("SUCCESS_THRESHOLD", DEFAULT_THRESHOLD)),
    )
    parser.add_argument(
        "--inner-iterations", type=int, default=DEFAULT_INNER_ITERATIONS
    )
    parser.add_argument(
        "--outer-iterations", type=int, default=DEFAULT_OUTER_ITERATIONS
    )
    parser.add_argument(
        "--loop-of-loops-iterations", type=int, default=DEFAULT_LOOP_OF_LOOPS_ITERATIONS
    )
    parser.add_argument("--rollout-count", type=int, default=DEFAULT_ROLLOUT_COUNT)
    parser.add_argument(
        "--steps-per-rollout", type=int, default=DEFAULT_STEPS_PER_ROLLOUT
    )
    parser.add_argument("--heldout-env-count", type=int, default=DEFAULT_HELDOUT_ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload-artifacts", action="store_true")
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--signal-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--byo-signal-converter", default="")
    parser.add_argument("--byo-trainer-command", default="")
    parser.add_argument("--byo-vlm-command", default="")
    parser.add_argument("--byo-eval-command", default="")


def _write_stage(
    local_dir: Path,
    number: int,
    name: str,
    payload: dict[str, Any],
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    path = local_dir / f"stage_{number:02d}_{name}" / (filename or f"{name}.json")
    return _write_json_artifact(path, payload)


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"path": str(path), "payload": payload}


def _write_env_manifest(root: Path, *, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    envs = [
        {
            "env_id": f"env-{index:04d}",
            "seed": rng.randrange(1, 2**31 - 1),
            "asset_ref": f"asset-{index:04d}",
            "physics": {
                "friction": round(0.5 + rng.random() * 0.5, 4),
                "mass_scale": round(0.85 + rng.random() * 0.3, 4),
                "lighting": round(0.4 + rng.random() * 0.5, 4),
            },
        }
        for index in range(count)
    ]
    return _write_json_artifact(
        root / "manifest.json",
        {"schema": "npa.sim2real.env_manifest.v1", "stage": 4, "envs": envs},
    )


def _write_train_heldout_split(
    root: Path,
    *,
    raw_envs: dict[str, Any],
    train_count: int,
    heldout_count: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    envs = list(raw_envs["payload"]["envs"])
    expected = train_count + heldout_count
    if len(envs) != expected:
        raise Sim2RealLoopError(
            f"raw env count {len(envs)} must equal train+heldout count {expected}"
        )
    rng = random.Random(seed)
    rng.shuffle(envs)
    train = envs[:train_count]
    heldout = envs[train_count:train_count + heldout_count]
    if len(train) != train_count or len(heldout) != heldout_count:
        raise Sim2RealLoopError("train/heldout split did not preserve requested counts")
    train_record = _write_json_artifact(
        root / "train" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "train",
            "envs": train,
        },
    )
    heldout_record = _write_json_artifact(
        root / "heldout" / "manifest.json",
        {
            "schema": "npa.sim2real.env_split.v1",
            "stage": 5,
            "split": "heldout",
            "envs": heldout,
        },
    )
    return train_record, heldout_record


def _trigger_payload(config: Sim2RealLoopConfig) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.trigger.v1",
        "stage": 1,
        "run_id": config.run_id,
        "created_at": _utc_now(),
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "input_format": "lerobot",
        "start_condition": "dataset_landed_in_trigger_path",
        "artifact_root": artifact_uris(config).get("root", ""),
        "byo_seams": byo_seams(config),
    }


def _tags_for_quality(quality: float, *, step: int) -> list[str]:
    if quality < 0.45:
        return ["missed_target", "unstable"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.65:
        return ["minor_alignment"] if step % 2 == 0 else ["late_grasp"]
    if quality < 0.8:
        return ["minor_alignment"]
    return ["ok"]


def _critique_for_tags(tags: list[str], *, quality: float) -> str:
    if tags == ["ok"]:
        return f"Step is stable; estimated rollout quality {quality:.2f}."
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])[
            "nl_correction"
        ]
        for tag in tags
    ]
    return " ".join(corrections)


def _merge_targets(tags: list[str]) -> dict[str, Any]:
    corrections = [
        CORRECTIVE_TARGETS.get(tag, CORRECTIVE_TARGETS["minor_alignment"])
        for tag in tags
    ]
    action_dim = max(len(item["action_delta"]) for item in corrections)
    merged = [0.0 for _ in range(action_dim)]
    for item in corrections:
        for index, value in enumerate(item["action_delta"]):
            merged[index] += float(value) / float(len(corrections))
    return {
        "nl_correction": " ".join(str(item["nl_correction"]) for item in corrections),
        "action_delta": [round(value, 6) for value in merged],
    }


def _signal_mean_reward(signal: dict[str, Any]) -> float:
    steps = signal.get("per_step") or []
    return sum(float(step["reward"]) for step in steps) / float(len(steps))


def _write_ppm(path: Path, *, red: int, green: int, blue: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 32
    height = 32
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes(
        [max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))]
    )
    path.write_bytes(header + pixel * width * height)


def _redacted_config(config: Sim2RealLoopConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else None
    return payload


def _artifact_root_uri(config: Sim2RealLoopConfig) -> str:
    parts = [part for part in (config.s3_prefix.strip("/"), config.run_id) if part]
    return f"s3://{config.s3_bucket}/{'/'.join(parts)}"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise Sim2RealLoopError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


__all__ = [
    "SCHEMA_HELDOUT_REPORT",
    "SCHEMA_RL_SIGNAL",
    "SCHEMA_THRESHOLD_DECISION",
    "SCHEMA_VLM_EVAL",
    "DEFAULT_LEROBOT_DATASET_ID",
    "Sim2RealLoopConfig",
    "Sim2RealLoopError",
    "artifact_uris",
    "build_config_from_env",
    "byo_seams",
    "convert_vlm_eval_to_rl_signal",
    "default_envgen_image",
    "default_eval_image",
    "default_policy_image",
    "default_trainer_image",
    "default_vlm_image",
    "evaluate_rollout_with_vlm",
    "generate_action_rollouts",
    "new_run_id",
    "run_full_loop",
    "run_heldout_eval",
    "run_inner_loop",
    "signal_mapping_rules",
    "threshold_decision",
]


if __name__ == "__main__":
    sys.exit(main())
