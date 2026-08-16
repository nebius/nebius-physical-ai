"""Workflow YAML generation and validation for the NPA agent UI."""

from __future__ import annotations

import re
import tempfile
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

API_VERSION_STABLE = "npa.workflow/v0.0.1"
API_VERSION_BETA = "npa.workflow/v0.0.1-beta"
API_VERSION = API_VERSION_STABLE
_SUPPORTED_API_VERSIONS = frozenset({API_VERSION_STABLE, API_VERSION_BETA})
_EMBEDDED_CANONICAL_SIM2REAL_YAML = ""

_TEMPLATES = (
    "two-step",
    "loop-gate",
    "vlm-rl-loop",
    "token-factory-gate",
    "byof",
    "rl-policy-success",
    "physical-ai-data-factory",
    "sim2real-staged",
)


class _FoldedStr(str):
    """YAML scalar rendered with folded (>) style."""


class _LiteralStr(str):
    """YAML scalar rendered with literal (|) style."""


def _real_finalize_run() -> OrderedDict[str, Any]:
    """A real S3 bookkeeping adapter shared by generated non-Sim2Real demos."""

    return OrderedDict(
        {
            "argv": [
                "python3",
                "-c",
                (
                    "from npa.workflows.data_factory_stages import finalize; "
                    "finalize('s3://{{config.bucket}}/{{config.prefix}}/', "
                    "'{{config.finalize_report_uri}}')"
                ),
            ]
        }
    )


class _WorkflowDumper(yaml.SafeDumper):
    pass


def _folded_representer(dumper: _WorkflowDumper, data: _FoldedStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style=">")


_WorkflowDumper.add_representer(_FoldedStr, _folded_representer)


def _literal_representer(dumper: _WorkflowDumper, data: _LiteralStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


_WorkflowDumper.add_representer(_LiteralStr, _literal_representer)

_TEMPLATE_ALIASES: dict[str, str] = {
    "vlm_rl_loop": "vlm-rl-loop",
    "vlm-rl": "vlm-rl-loop",
    "vlm_rl": "vlm-rl-loop",
    "token_factory_gate": "token-factory-gate",
    "gate": "token-factory-gate",
    "tokenfactory": "token-factory-gate",
    "loop_gate": "loop-gate",
    "loop": "loop-gate",
    "isaac_byof": "byof",
    "isaac-byof": "byof",
    "isaac-lab": "byof",
    "leisaac": "byof",
    "byof": "byof",
    "rl-policy": "rl-policy-success",
    "rl_policy": "rl-policy-success",
    "policy-training": "rl-policy-success",
    "policy_training": "rl-policy-success",
    "rl-training": "rl-policy-success",
    "rl_training": "rl-policy-success",
    "paidf": "physical-ai-data-factory",
    "data-factory": "physical-ai-data-factory",
    "data_factory": "physical-ai-data-factory",
    "datafactory": "physical-ai-data-factory",
    "physical-ai-data-factory": "physical-ai-data-factory",
    "physical_ai_data_factory": "physical-ai-data-factory",
    "video-augmentation": "physical-ai-data-factory",
    "video_augmentation": "physical-ai-data-factory",
    "augment-multiply": "physical-ai-data-factory",
    "augment_multiply": "physical-ai-data-factory",
    "multiply": "physical-ai-data-factory",
    "fanout-augment": "physical-ai-data-factory",
    "sim2real-staged": "sim2real-staged",
    "sim-to-real": "sim2real-staged",
    "staged-sim2real": "sim2real-staged",
    "real-sim2real": "sim2real-staged",
}

_INTENT_DEFAULT_TEMPLATE: dict[str, str] = {
    "create_workflow": "two-step",
    "create_vlm_rl_workflow": "sim2real-staged",
    "create_gate_workflow": "token-factory-gate",
    "create_loop_gate_workflow": "loop-gate",
    "create_rl_policy_workflow": "rl-policy-success",
    "create_data_factory_workflow": "physical-ai-data-factory",
}

_TEMPLATE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "byof": (
        "byof",
        "bring your own fork",
        "ubuntu",
        "base image",
        "leisaac",
        "lightwheel",
        "isaac lab",
        "isaac-lab",
        "datagen",
        "state machine",
        "synthetic demonstration",
        "demonstration data",
    ),
    "token-factory-gate": (
        "token",
        "tokenfactory",
        "scene reasoning",
        "reason scene",
        "quality gate",
        "cosmos gate",
    ),
    "vlm-rl-loop": (
        "vlm",
        "rl",
        "outer loop",
        "inner loop",
        "heldout",
        "policy rollout",
        "promote",
    ),
    "loop-gate": ("loop", "gate", "decision", "transition", "multi-step", "multistep"),
    "rl-policy-success": (
        "rl policy",
        "policy training",
        "reinforcement learning",
        "train policy",
        "simulation policy",
    ),
    "physical-ai-data-factory": (
        "paidf",
        "data factory",
        "physical ai data factory",
        "video augmentation",
        "video data augmentation",
        "augment",
        "fan out",
        "fan-out",
        "fanout",
        "multiply",
        "scenario variant",
        "scenario variants",
        "cosmos transfer",
        "amplify",
    ),
    "sim2real-staged": (
        "sim2real",
        "sim-to-real",
        "sim to real",
        "success threshold",
        "heldout",
        "held-out",
        "rollout",
        "robot preset",
        "genesis",
        "isaac task",
        "vlm-rl",
        "vlm rl",
    ),
    "two-step": ("two-step", "2-step", "simple", "minimal"),
}


def _workflow_specs() -> dict[str, dict[str, Any]]:
    return {
        "two-step": {
            "name": "sim2real-two-step",
            "description": (
                "DEMO ONLY: agent-generated Cosmos Transfer and raw-env fixture; "
                "it does not run the canonical 14-stage Sim2Real engine."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "sim2real/{{run.id}}",
                    "env_count": "1000",
                }
            ),
            "config_uri": OrderedDict(
                {
                    "trigger_uri": "s3://{{config.bucket}}/sim2real-triggers/{{run.id}}/lerobot-pusht/",
                    "augment_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "augment_manifest_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/manifest.json",
                    # `sim2real_envgen` takes the RUN ROOT and derives envs/raw,
                    # envs/train, envs/heldout and envs/manifest beneath it.
                    "envgen_root_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
                    "raw_envs_uri": "s3://{{config.bucket}}/{{config.prefix}}/envs/raw/",
                    "shard_index": "0",
                    "shard_count": "1",
                    "train_fraction": "0.8",
                    "envgen_seed": "42",
                    "augmented_frames_uri": "{{config.augment_manifest_uri}}",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}
                    ),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": (
                                "Real Cosmos Transfer augmentation conditioned on the seeded "
                                "input video."
                            ),
                            "toolRef": "workbench.cosmos2.transfer_conditioned_execute",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.trigger_uri}}",
                                        "schema": "npa.sim2real.trigger_dataset.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "next": "envgen",
                        }
                    ),
                    "envgen": OrderedDict(
                        {
                            "description": (
                                "Generate raw envs using only frame URIs declared by transfer."
                            ),
                            "needs": ["augment"],
                            "toolRef": "workbench.sim2real_envgen.raw_shard",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.raw_envs_uri}}raw-shard-00-summary.json",
                                        "schema": "npa.sim2real.raw_env_shard_summary.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "byof": {
            "name": "byof",
            "description": (
                "Generic BYOF workflow: build/push an OSS repo image on Ubuntu or Isaac Lab "
                "and run RL training or scripted datagen on live Kubernetes."
            ),
            "config_runtime": OrderedDict(
                {
                    "repo_url": "<repo-url>",
                    "repo_ref": "<repo-ref>",
                    "base_profile": "ubuntu",
                    "base_image": "",
                    "build_command": "",
                    "workload": "<workload>",
                    "smoke_command": "",
                    "solution_name": "",
                    "capability_name": "",
                    "smoke_artifact_name": "",
                    "resource_profile_yaml": "<resource-profile.yaml>",
                    "task": "<task>",
                    "iterations": 1,
                    "num_envs": 4,
                    "num_demos": 10,
                    "wait_timeout": 21600,
                    "poll_interval": 60,
                }
            ),
            "config_uri": OrderedDict(
                {
                    "output_root": "s3://{{config.bucket}}/byof/{{run.id}}",
                    "summary_uri": "{{config.output_root}}/npa_byof_summary.json",
                    "dataset_uri": "{{config.output_root}}/dataset.hdf5",
                    "checkpoint_uri": "{{config.output_root}}/npa_isaac_lab_checkpoint.pt",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}
                    ),
                }
            ),
            "initial": "byof-run",
            "states": OrderedDict(
                {
                    "byof-run": OrderedDict(
                        {
                            "description": "Build BYOF image from config.repo_url and run the selected workload.",
                            "toolRef": "workbench.byof.repo",
                            "resources": "gpu",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.summary_uri}}",
                                        "schema": "npa.workbench.byof.summary.v1",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "uri": "{{config.dataset_uri}}",
                                        "schema": "npa.workbench.byof.dataset.v1",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "uri": "{{config.checkpoint_uri}}",
                                        "schema": "npa.workbench.isaac_lab.checkpoint.v1",
                                    }
                                ),
                            ],
                            "terminal": True,
                        }
                    )
                }
            ),
        },
        "loop-gate": {
            "name": "sim2real-loop-gate-agent",
            "description": (
                "Sim2Real workflow with dynamic decision gating: augment, quality "
                "refine loop, then finalize."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "sim2real-loop/{{run.id}}",
                    "vlm_backend": "api",
                    "refinement_iterations": 3,
                    "default_decision": "loop_back",
                }
            ),
            "config_uri": OrderedDict(
                {
                    "trigger_uri": "s3://{{config.bucket}}/sim2real-triggers/{{run.id}}/lerobot-pusht/",
                    "augment_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "augment_manifest_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/manifest.json",
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/scores/",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}
                    ),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment stage.",
                            "toolRef": "workbench.cosmos2.transfer_conditioned_execute",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.trigger_uri}}",
                                        "schema": "npa.sim2real.trigger_dataset.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "next": "refine",
                        }
                    ),
                    "refine": OrderedDict(
                        {
                            "description": "Iterate critique and decision gate until promoted.",
                            "needs": ["augment"],
                            "loop": OrderedDict(
                                {
                                    "max": "{{config.refinement_iterations}}",
                                    "until": "promote_checkpoint",
                                }
                            ),
                            "sequence": ["vlm-critique", "quality-gate"],
                            "next": "publish",
                        }
                    ),
                    "vlm-critique": OrderedDict(
                        {
                            "description": "Score augmented rollouts before gate.",
                            "toolRef": "workbench.vlm_eval.run",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.rollouts_uri}}",
                                        "schema": "npa.workbench.rollout_set.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.scores_uri}}report.json",
                                        "schema": "npa.workbench.vlm_eval.report.v1",
                                    }
                                )
                            ],
                        }
                    ),
                    "quality-gate": OrderedDict(
                        {
                            "description": "Persist decision to promote or loop back.",
                            "writesDecision": True,
                            "needs": ["vlm-critique"],
                            "toolRef": "workbench.sim2real.write_decision",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.sim2real.threshold_decision.v1",
                                    }
                                )
                            ],
                            "transitions": [
                                OrderedDict(
                                    {"when": "promote_checkpoint", "goto": "publish"}
                                ),
                                OrderedDict({"when": "loop_back", "goto": "augment"}),
                            ],
                        }
                    ),
                    "publish": OrderedDict(
                        {
                            "description": "Finalize report once promoted.",
                            "needs": ["refine"],
                            "run": _real_finalize_run(),
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.finalize_report_uri}}",
                                        "schema": "npa.sim2real.e2e_report.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "vlm-rl-loop": {
            "name": "sim2real-vlm-rl",
            "description": (
                "VLM-to-RL staged loop: augment, envgen, outer loop (inner rollouts + "
                "VLM critique), held-out eval, promote/loop-back gate, finalize."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "sim2real/{{run.id}}",
                    "inner_iterations": 3,
                    "outer_iterations": 2,
                    "default_decision": "loop_back",
                    "env_count": "10000",
                    "vlm_backend": "self-hosted",
                }
            ),
            "config_uri": OrderedDict(
                {
                    "trigger_uri": "s3://{{config.bucket}}/sim2real-triggers/{{run.id}}/lerobot-pusht/",
                    "augment_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "augment_manifest_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/manifest.json",
                    # `sim2real_envgen` takes the RUN ROOT and derives envs/raw,
                    # envs/train, envs/heldout and envs/manifest beneath it.
                    "envgen_root_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
                    "raw_envs_uri": "s3://{{config.bucket}}/{{config.prefix}}/envs/raw/",
                    "shard_index": "0",
                    "shard_count": "1",
                    "train_fraction": "0.8",
                    "envgen_seed": "42",
                    "augmented_frames_uri": "{{config.augment_manifest_uri}}",
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/actions/train/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/vlm_eval/train/",
                    "heldout_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/eval/heldout/report.json",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/outer_loop/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/sim2real-report.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}
                    ),
                    "cpu": OrderedDict({"cloud": "kubernetes", "cpus": 8}),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment of LeRobot trigger data.",
                            "toolRef": "workbench.cosmos2.transfer_conditioned_execute",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.trigger_uri}}",
                                        "schema": "npa.sim2real.trigger_dataset.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "next": "envgen",
                        }
                    ),
                    "envgen": OrderedDict(
                        {
                            "description": "Generate raw environment shard catalog on object storage.",
                            "needs": ["augment"],
                            "toolRef": "workbench.sim2real_envgen.raw_shard",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.raw_envs_uri}}raw-shard-00-summary.json",
                                        "schema": "npa.sim2real.raw_env_shard_summary.v1",
                                    }
                                )
                            ],
                            "next": "outer",
                        }
                    ),
                    "outer": OrderedDict(
                        {
                            "description": "Outer loop: inner train pass, held-out eval, threshold gate.",
                            "needs": ["envgen"],
                            "loop": OrderedDict(
                                {
                                    "max": "{{config.outer_iterations}}",
                                    "until": "promote_checkpoint",
                                }
                            ),
                            "sequence": ["inner", "heldout", "decide"],
                            "next": "finalize",
                        }
                    ),
                    "inner": OrderedDict(
                        {
                            "description": "Inner loop: rollouts and VLM critique per iteration.",
                            "loop": OrderedDict({"max": "{{config.inner_iterations}}"}),
                            "sequence": ["rollouts", "vlm-score"],
                        }
                    ),
                    "rollouts": OrderedDict(
                        {
                            "description": "Policy action rollouts on train envs.",
                            "run": OrderedDict(
                                {"shell": "false # retired stub template"}
                            ),
                            "resources": "gpu",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.rollouts_uri}}",
                                        "schema": "npa.sim2real.action_rollout.v1",
                                    }
                                )
                            ],
                        }
                    ),
                    "vlm-score": OrderedDict(
                        {
                            "description": "VLM evaluation over train rollouts.",
                            "needs": ["rollouts"],
                            "toolRef": "workbench.vlm_eval.run",
                            "resources": "gpu",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.scores_uri}}report.json",
                                        "schema": "npa.workbench.vlm_eval.report.v1",
                                    }
                                )
                            ],
                        }
                    ),
                    "heldout": OrderedDict(
                        {
                            "description": "Held-out simulation evaluation report.",
                            "run": OrderedDict(
                                {"shell": "false # retired stub template"}
                            ),
                            "resources": "gpu",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.heldout_report_uri}}",
                                        "schema": "npa.sim2real.heldout_eval.v1",
                                    }
                                )
                            ],
                        }
                    ),
                    "decide": OrderedDict(
                        {
                            "description": "Threshold decision: promote_checkpoint or loop_back.",
                            "writesDecision": True,
                            "needs": ["heldout"],
                            "toolRef": "workbench.sim2real.write_decision",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.sim2real.threshold_decision.v1",
                                    }
                                )
                            ],
                            "transitions": [
                                OrderedDict(
                                    {"when": "promote_checkpoint", "goto": "finalize"}
                                ),
                                OrderedDict({"when": "loop_back", "goto": "outer"}),
                            ],
                        }
                    ),
                    "finalize": OrderedDict(
                        {
                            "description": "Report upload and visualization artifacts.",
                            "needs": ["outer"],
                            "run": _real_finalize_run(),
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.finalize_report_uri}}",
                                        "schema": "npa.sim2real.e2e_report.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "token-factory-gate": {
            "name": "tokenfactory-cosmos-gate",
            "description": (
                "Token Factory scene reasoning, Cosmos Transfer augment, and a VLM "
                "quality gate loop until the synthetic batch is promoted."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "tokenfactory-cosmos-gate/{{run.id}}",
                    "vlm_backend": "api",
                    "refinement_iterations": 3,
                    "default_decision": "loop_back",
                }
            ),
            "config_uri": OrderedDict(
                {
                    "scene_uri": "s3://{{config.bucket}}/{{config.prefix}}/scene/",
                    "plan_uri": "s3://{{config.bucket}}/{{config.prefix}}/plan/",
                    "trigger_uri": "s3://{{config.bucket}}/{{config.prefix}}/scene/",
                    "augment_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "augment_manifest_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/manifest.json",
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/scores/",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "accelerators": "RTXPRO6000:1",
                            "cpus": 16,
                            "memory": "80Gi",
                        }
                    ),
                }
            ),
            "initial": "reason-scene",
            "states": OrderedDict(
                {
                    "reason-scene": OrderedDict(
                        {
                            "description": "Token Factory reasoner over captured scene frames.",
                            "toolRef": "workbench.token_factory.reason",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.scene_uri}}",
                                        "schema": "npa.token_factory.scene.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.plan_uri}}plan.json",
                                        "schema": "npa.token_factory.plan.v1",
                                    }
                                )
                            ],
                            "next": "augment-scene",
                        }
                    ),
                    "augment-scene": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment driven by the scene plan.",
                            "needs": ["reason-scene"],
                            "toolRef": "workbench.cosmos2.transfer_conditioned_execute",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.trigger_uri}}",
                                        "schema": "npa.token_factory.scene.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_manifest_uri}}",
                                        "schema": "npa.cosmos2.transfer.v1",
                                    }
                                )
                            ],
                            "next": "refine",
                        }
                    ),
                    "refine": OrderedDict(
                        {
                            "description": "VLM critique loop with promote versus re-augment gate.",
                            "needs": ["augment-scene"],
                            "loop": OrderedDict(
                                {
                                    "max": "{{config.refinement_iterations}}",
                                    "until": "promote_checkpoint",
                                }
                            ),
                            "sequence": ["vlm-critique", "quality-gate"],
                            "next": "publish",
                        }
                    ),
                    "vlm-critique": OrderedDict(
                        {
                            "description": "Score augmented frames before the quality gate.",
                            "toolRef": "workbench.vlm_eval.run",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.rollouts_uri}}",
                                        "schema": "npa.workbench.rollout_set.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.scores_uri}}report.json",
                                        "schema": "npa.workbench.vlm_eval.report.v1",
                                    }
                                )
                            ],
                        }
                    ),
                    "quality-gate": OrderedDict(
                        {
                            "description": "Promote good batches or loop back for another augment pass.",
                            "writesDecision": True,
                            "needs": ["vlm-critique"],
                            "toolRef": "workbench.sim2real.write_decision",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.sim2real.threshold_decision.v1",
                                    }
                                )
                            ],
                            "transitions": [
                                OrderedDict(
                                    {"when": "promote_checkpoint", "goto": "publish"}
                                ),
                                OrderedDict(
                                    {"when": "loop_back", "goto": "augment-scene"}
                                ),
                            ],
                        }
                    ),
                    "publish": OrderedDict(
                        {
                            "description": "Write final report when the batch is promoted.",
                            "needs": ["refine"],
                            "run": _real_finalize_run(),
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.finalize_report_uri}}",
                                        "schema": "npa.sim2real.e2e_report.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "gpu-cross-region": {
            "name": "sim2real-gpu-cross-region",
            "description": (
                "Tenant-scoped GPU workflow that runs stages across primary and "
                "secondary project/region targets with containerized glue transforms."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "sim2real-cross-region/{{run.id}}",
                    "tenant_id": "tenant-example",
                    "project_primary": "project-primary",
                    "project_secondary": "project-secondary",
                    "region_primary": "us-central1",
                    "region_secondary": "eu-north1",
                    "improvement_local_path": "/tmp/{{run.id}}-improvement.json",
                }
            ),
            "config_uri": OrderedDict(
                {
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/rollouts/primary/",
                    "normalized_rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/rollouts/normalized/",
                    "heldout_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/eval/secondary/report.json",
                    "improvement_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/improvement.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu-primary": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "accelerators": "RTXPRO6000:1",
                            "project_alias": "{{config.project_primary}}",
                            "region": "{{config.region_primary}}",
                        }
                    ),
                    "gpu-secondary": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "accelerators": "RTXPRO6000:1",
                            "project_alias": "{{config.project_secondary}}",
                            "region": "{{config.region_secondary}}",
                        }
                    ),
                    "container-glue": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "cpus": 4,
                            "memory": "16Gi",
                            "image": "python:3.11-slim",
                            "project_alias": "{{config.project_secondary}}",
                            "region": "{{config.region_secondary}}",
                        }
                    ),
                }
            ),
            "initial": "primary-rollout",
            "states": OrderedDict(
                {
                    "primary-rollout": OrderedDict(
                        {
                            "description": "Run primary GPU rollout workload.",
                            "run": OrderedDict(
                                {"shell": "false # retired stub template"}
                            ),
                            "resources": "gpu-primary",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.rollouts_uri}}manifest.json",
                                        "schema": "npa.sim2real.action_rollout.v1",
                                    }
                                )
                            ],
                            "next": "transform-rollouts",
                        }
                    ),
                    "transform-rollouts": OrderedDict(
                        {
                            "description": (
                                "Contract adapter/validator stage that normalizes rollout artifacts "
                                "across project/region boundaries."
                            ),
                            "resources": "container-glue",
                            "toolRef": "workbench.data_transform.rollout_contract",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.rollouts_uri}}manifest.json",
                                        "schema": "npa.sim2real.action_rollout.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.normalized_rollouts_uri}}manifest.json",
                                        "schema": "npa.sim2real.rollout_manifest.v1",
                                    }
                                )
                            ],
                            "next": "secondary-eval",
                        }
                    ),
                    "secondary-eval": OrderedDict(
                        {
                            "description": "Run secondary GPU held-out evaluation workload.",
                            "run": OrderedDict(
                                {"shell": "false # retired stub template"}
                            ),
                            "resources": "gpu-secondary",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.normalized_rollouts_uri}}manifest.json",
                                        "schema": "npa.sim2real.rollout_manifest.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.heldout_report_uri}}",
                                        "schema": "npa.sim2real.heldout_eval.v1",
                                    }
                                )
                            ],
                            "next": "summarize-improvement",
                        }
                    ),
                    "summarize-improvement": OrderedDict(
                        {
                            "description": (
                                "Compute and validate cross-region improvement contract payload "
                                "for downstream reporting."
                            ),
                            "resources": "container-glue",
                            "toolRef": "workbench.data_transform.improvement_summary",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.improvement_report_uri}}",
                                        "schema": "npa.sim2real.improvement_report.v1",
                                    }
                                )
                            ],
                            "next": "finalize",
                        }
                    ),
                    "finalize": OrderedDict(
                        {
                            "description": "Finalize tenant-scoped cross-region run report.",
                            "run": _real_finalize_run(),
                            "resources": "gpu-secondary",
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.finalize_report_uri}}",
                                        "schema": "npa.sim2real.e2e_report.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "rl-policy-success": {
            "name": "rl-policy-training-sim-success",
            "description": (
                "Simulation RL policy workflow with explicit train/eval success gating "
                "and publish-or-fail terminal outcomes."
            ),
            "config_runtime": OrderedDict(
                {
                    "prefix": "rl-policy/{{run.id}}",
                    "task_name": "Isaac-Cartpole-v0",
                    "train_steps": 500000,
                    "learning_rate": 0.0003,
                    # `workbench.rl.policy_train` passes this as Isaac Lab's real
                    # `--num-envs` (the vectorized rollout batch dimension); there
                    # is no `--batch-size` option on that CLI.
                    "num_envs": 256,
                    "eval_episodes": 50,
                    "success_threshold": 0.85,
                }
            ),
            "config_uri": OrderedDict(
                {
                    "train_dataset_uri": "s3://{{config.bucket}}/{{config.prefix}}/inputs/train/",
                    "checkpoint_uri": "s3://{{config.bucket}}/{{config.prefix}}/artifacts/policy/latest.ckpt",
                    "eval_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/metrics/eval.json",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json",
                    "release_uri": "s3://{{config.bucket}}/{{config.prefix}}/artifacts/policy/release/",
                    "release_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/release.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "trainer-gpu": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "accelerators": "RTXPRO6000:1",
                        }
                    ),
                    "eval-gpu": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "accelerators": "RTXPRO6000:1",
                        }
                    ),
                    "control-cpu": OrderedDict(
                        {
                            "cloud": "kubernetes",
                            "cpus": 4,
                            "memory": "8Gi",
                            "image": "python:3.11-slim",
                        }
                    ),
                }
            ),
            "initial": "train-policy",
            "states": OrderedDict(
                {
                    "train-policy": OrderedDict(
                        {
                            "description": "Train RL policy on simulator task.",
                            "toolRef": "workbench.rl.policy_train",
                            "resources": "trainer-gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.train_dataset_uri}}",
                                        "schema": "npa.rl.training_dataset.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.checkpoint_uri}}",
                                        "schema": "npa.rl.policy_checkpoint.v1",
                                    }
                                )
                            ],
                            "next": "eval-policy",
                        }
                    ),
                    "eval-policy": OrderedDict(
                        {
                            "description": "Run held-out simulation evaluation for trained checkpoint.",
                            "needs": ["train-policy"],
                            "toolRef": "workbench.rl.evaluate_policy",
                            "resources": "eval-gpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.checkpoint_uri}}",
                                        "schema": "npa.rl.policy_checkpoint.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.eval_report_uri}}",
                                        "schema": "npa.rl.eval_report.v1",
                                    }
                                )
                            ],
                            "next": "success-gate",
                        }
                    ),
                    "success-gate": OrderedDict(
                        {
                            "description": (
                                "Write promote-or-loop decision from eval metrics and success threshold."
                            ),
                            "writesDecision": True,
                            "needs": ["eval-policy"],
                            "toolRef": "workbench.rl.write_success_decision",
                            "resources": "control-cpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.eval_report_uri}}",
                                        "schema": "npa.rl.eval_report.v1",
                                    }
                                )
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.rl.training_decision.v1",
                                    }
                                )
                            ],
                            "transitions": [
                                OrderedDict(
                                    {
                                        "when": "promote_checkpoint",
                                        "goto": "publish-policy",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "when": "loop_back",
                                        "goto": "training-not-success",
                                    }
                                ),
                            ],
                        }
                    ),
                    "publish-policy": OrderedDict(
                        {
                            "description": "Publish promoted checkpoint with release manifest.",
                            "needs": ["success-gate"],
                            "toolRef": "workbench.rl.publish_policy",
                            "resources": "control-cpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.checkpoint_uri}}",
                                        "schema": "npa.rl.policy_checkpoint.v1",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.rl.training_decision.v1",
                                    }
                                ),
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.release_uri}}manifest.json",
                                        "schema": "npa.rl.policy_release.v1",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "uri": "{{config.release_report_uri}}",
                                        "schema": "npa.rl.training_success_report.v1",
                                    }
                                ),
                            ],
                            "terminal": True,
                        }
                    ),
                    "training-not-success": OrderedDict(
                        {
                            "description": (
                                "Record explicit non-promotion outcome when success threshold is not met."
                            ),
                            "needs": ["success-gate"],
                            "toolRef": "workbench.rl.report_failure",
                            "resources": "control-cpu",
                            "inputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.eval_report_uri}}",
                                        "schema": "npa.rl.eval_report.v1",
                                    }
                                ),
                                OrderedDict(
                                    {
                                        "uri": "{{config.decision_uri}}",
                                        "schema": "npa.rl.training_decision.v1",
                                    }
                                ),
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.release_report_uri}}",
                                        "schema": "npa.rl.training_failure_report.v1",
                                    }
                                )
                            ],
                            "terminal": True,
                        }
                    ),
                }
            ),
        },
        "physical-ai-data-factory": _data_factory_spec(),
    }


def _data_factory_spec() -> dict[str, Any]:
    """Physical AI Data Factory blueprint (annotate -> augment+multiply -> grade
    loop -> re-label -> curate -> visualize -> finalize).

    Chat requests parameterize what to augment (``augment_subject``, surfaced as
    the input-conditioning prompt hint) and how many scenario variants to fan out
    (``n_augmentations``). The augment stage runs one real Cosmos Transfer 2.5
    inference per sampled combo and fans them across the GPU pod's devices, so an
    N-augmentation config on an ``RTXPRO6000:G`` pod amplifies N scenarios across
    G GPUs.
    """
    return {
        "name": "physical-ai-data-factory",
        "description": (
            "Physical AI Data Factory (NVIDIA blueprint on Nebius + SkyPilot, no OSMO): "
            "annotate (Token Factory VLM) -> Cosmos Transfer 2.5 augment & MULTIPLY "
            "(one GPU inference per sampled scenario, fanned across the pod's GPUs) -> "
            "VLM evaluate/validate gate loop -> re-label -> FiftyOne curation -> Rerun "
            "visualize -> finalize. Pure composition of existing workbench tools."
        ),
        "config_runtime": OrderedDict(
            {
                "prefix": "physical-ai-data-factory/{{run.id}}",
                "seed_fixture": "false",
                "seed_default_input": "false",
                # What to augment: a free-form hint surfaced as the augment prompt /
                # input-conditioning subject (the run's real input clips are the base).
                "augment_subject": "the input robot clips",
                # Fan-out: N sampled appearance combos -> N scenario variants.
                "n_augmentations": "4",
                # Concurrency of the multiply fan-out (one variant per GPU); align
                # with the gpu resource accelerator count for full utilization.
                "variant_parallelism": "4",
                # Conditioning shape: edge (on-the-fly Canny) by default, or seg to
                # condition on a GroundingDINO+SAM2 segmentation of the same input.
                # The mask keys restrict the control to one segmented region.
                "augment_control": "edge",
                "augment_control_weight": "1.0",
                "augment_control_prompt": "",
                "augment_control_asset_uri": "",
                "augment_mask_prompt": "",
                "augment_mask_asset_uri": "",
                "refinement_iterations": "2",
                "grade_threshold": "0.75",
                "default_decision": "loop_back",
                "temporal_consistency_mode": "advisory",
                "temporal_consistency_threshold": "0.8",
                "temporal_noise_floor": "0.25",
                "temporal_blur_ksize": "7",
                "temporal_regions_json": "",
                "appearance_fidelity_mode": "advisory",
                "appearance_fidelity_threshold": "0.8",
                "appearance_luminance_tolerance": "18.0",
                "appearance_global_chroma_tolerance": "8.0",
                "appearance_local_chroma_tolerance": "6.0",
                "appearance_chroma_instability_tolerance": "4.0",
                "appearance_blur_ksize": "7",
                "appearance_max_dimension": "256",
                "appearance_regions_json": "",
                "caption_model": "Qwen/Qwen2.5-VL-72B-Instruct",
                "vlm_backend": "api",
                "max_images": "8",
                "max_tokens": "512",
                "curator_clip_len_s": "3",
                "curator_min_clip_len_s": "1",
                "curator_motion_filter": "score-only",
                "fiftyone_dedup_threshold": "0.10",
            }
        ),
        "config_uri": OrderedDict(
            {
                "input_uri": "s3://{{config.bucket}}/{{config.prefix}}/input/",
                "images_uri": "s3://{{config.bucket}}/{{config.prefix}}/input/",
                "configs_uri": "s3://{{config.bucket}}/{{config.prefix}}/configs/",
                "captions_uri": "s3://{{config.bucket}}/{{config.prefix}}/labeled_original/",
                # Mandatory: managed Cosmos Transfer fails closed unless this
                # prefix contains a supported input video.
                "trigger_uri": "s3://{{config.bucket}}/{{config.prefix}}/input/",
                "augment_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "augment_manifest_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/manifest.json",
                # Sibling of the augmented clips, never nested inside them: the
                # evaluator treats every child of cosmos_augmented/ as a variant.
                "augment_control_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_control/",
                "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/grade/",
                "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/grade/decision.json",
                "quality_disposition_uri": "s3://{{config.bucket}}/{{config.prefix}}/grade/quality_disposition.json",
                "augmented_frames_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "labeled_augmented_uri": "s3://{{config.bucket}}/{{config.prefix}}/labeled_augmented/",
                "lance_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "curated_clips_uri": "s3://{{config.bucket}}/{{config.prefix}}/curation/cosmos_curator/",
                "curator_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/curation/cosmos_curator.json",
                "curation_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/curation/report.json",
                "run_root_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
                "rrd_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/sim2real.rrd",
                "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
            }
        ),
        "resources": OrderedDict(
            {
                "gpu": OrderedDict(
                    {
                        "cloud": "kubernetes",
                        "accelerators": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:4",
                        "cpus": 16,
                        "memory": "128Gi",
                    }
                ),
                "cpu": OrderedDict(
                    {"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"}
                ),
            }
        ),
        "initial": "generate-configs",
        "states": OrderedDict(
            {
                "generate-configs": OrderedDict(
                    {
                        "description": (
                            "Stage 1 - Config Generation. Sample n_augmentations appearance "
                            "combos and write the per-run config manifest that drives the "
                            "multiply fan-out."
                        ),
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "argv": [
                                    "python3",
                                    "-c",
                                    (
                                        "import sys; from npa.workflows.data_factory_stages "
                                        "import generate_configs; generate_configs(*sys.argv[1:])"
                                    ),
                                    "{{config.configs_uri}}",
                                    "{{config.n_augmentations}}",
                                    "{{run.id}}",
                                    "{{config.images_uri}}",
                                    "{{config.seed_default_input}}",
                                    "{{config.seed_fixture}}",
                                    "{{config.augment_subject}}",
                                ]
                            }
                        ),
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.configs_uri}}manifest.json",
                                    "schema": "npa.data_factory.configs.v1",
                                }
                            )
                        ],
                        "next": "annotate-original",
                    }
                ),
                "annotate-original": OrderedDict(
                    {
                        "description": (
                            "Stage 2a - Understand & Annotate. Dense-caption the source frames "
                            "with a hosted Token Factory VLM (zero-GPU)."
                        ),
                        "needs": ["generate-configs"],
                        "toolRef": "workbench.token_factory.caption",
                        "resources": "cpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.images_uri}}",
                                    "schema": "npa.data_factory.frames.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.captions_uri}}captions.json",
                                    "schema": "npa.token_factory.captions.v1",
                                }
                            )
                        ],
                        "next": "grade",
                    }
                ),
                "augment": OrderedDict(
                    {
                        "description": (
                            "Stage 2b - Augment & Multiply. Cosmos Transfer 2.5 runs ONE GPU "
                            "inference per sampled combo (config.n_augmentations) and fans them "
                            "across the pod's GPUs (config.variant_parallelism), so N combos -> "
                            "N input-conditioned scenario variants amplifying config.augment_subject. "
                            "A supported video under config.trigger_uri is mandatory. Member of the "
                            "grade refinement loop, so loop_back genuinely re-renders."
                        ),
                        "needs": ["annotate-original"],
                        "toolRef": "workbench.cosmos2.transfer_execute",
                        "resources": "gpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.trigger_uri}}",
                                    "schema": "npa.data_factory.frames.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.augment_manifest_uri}}",
                                    "schema": "npa.cosmos2.transfer.v1",
                                }
                            )
                        ],
                    }
                ),
                "grade": OrderedDict(
                    {
                        "description": (
                            "Augment & Evaluate refinement loop: augment (GPU multiply) -> "
                            "Cosmos Evaluator -> quality-gate. Loops back to RE-AUGMENT on "
                            "failure, up to refinement_iterations, and breaks on promote."
                        ),
                        "needs": ["annotate-original"],
                        "loop": OrderedDict(
                            {
                                "max": "{{config.refinement_iterations}}",
                                "until": "promote_checkpoint",
                            }
                        ),
                        "sequence": ["augment", "evaluate", "quality-gate"],
                        "next": "quality-disposition",
                    }
                ),
                "evaluate": OrderedDict(
                    {
                        "description": (
                            "Evaluate with the real NVIDIA Cosmos Evaluator: Token Factory "
                            "attribute verification plus hallucinated-motion comparison against "
                            "the input-conditioned source clip, source-relative temporal "
                            "consistency, and protected-appearance fidelity."
                        ),
                        "toolRef": "workbench.cosmos_evaluator.evaluate",
                        "resources": "cpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.rollouts_uri}}",
                                    "schema": "npa.cosmos2.transfer.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.scores_uri}}cosmos_evaluator.json",
                                    "schema": "npa.cosmos_evaluator.report.v1",
                                }
                            )
                        ],
                    }
                ),
                "quality-gate": OrderedDict(
                    {
                        "description": (
                            "Read the VLM attribute score and write a promote_checkpoint / "
                            "loop_back decision that drives the grade loop."
                        ),
                        "writesDecision": True,
                        "needs": ["evaluate"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "shell": (
                                    'python3 -c "from npa.workflows.data_factory_stages import '
                                    "grade_gate; grade_gate('{{config.scores_uri}}', "
                                    "'{{config.decision_uri}}', '{{config.grade_threshold}}')\""
                                )
                            }
                        ),
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.decision_uri}}",
                                    "schema": "npa.sim2real.threshold_decision.v1",
                                }
                            )
                        ],
                    }
                ),
                "quality-disposition": OrderedDict(
                    {
                        "description": (
                            "Fail closed after the refinement loop: persist an auditable "
                            "accepted/rejected disposition before rejecting a degraded, "
                            "below-threshold, or hard-check-failing batch."
                        ),
                        "needs": ["grade"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "argv": [
                                    "python3",
                                    "-c",
                                    (
                                        "import sys; from npa.workflows.data_factory_stages "
                                        "import enforce_quality_disposition; "
                                        "enforce_quality_disposition(*sys.argv[1:])"
                                    ),
                                    "{{config.scores_uri}}",
                                    "{{config.quality_disposition_uri}}",
                                    "{{config.grade_threshold}}",
                                ]
                            }
                        ),
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.quality_disposition_uri}}",
                                    "schema": "npa.data_factory.quality_disposition.v1",
                                }
                            )
                        ],
                        "next": "annotate-augmented",
                    }
                ),
                "annotate-augmented": OrderedDict(
                    {
                        "description": (
                            "Stage 3 - Pseudo-Label Augmented. Re-caption the promoted augmented "
                            "clips with the same hosted VLM so the amplified set ships labeled."
                        ),
                        "needs": ["quality-disposition"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "shell": (
                                    "npa workbench token-factory caption "
                                    '--input-path "{{config.augmented_frames_uri}}" '
                                    '--output-path "{{config.labeled_augmented_uri}}" '
                                    '--model "{{config.caption_model}}" '
                                    '--max-images "{{config.max_images}}" '
                                    '--max-tokens "{{config.max_tokens}}" --output json'
                                )
                            }
                        ),
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.labeled_augmented_uri}}captions.json",
                                    "schema": "npa.token_factory.captions.v1",
                                }
                            )
                        ],
                        "next": "cosmos-curate",
                    }
                ),
                "cosmos-curate": OrderedDict(
                    {
                        "description": (
                            "Stage 4a - Cosmos Curator. Run the real NVIDIA Cosmos Curator "
                            "split, transcode, motion-score, and catalog stages."
                        ),
                        "needs": ["annotate-augmented"],
                        "toolRef": "workbench.cosmos_curate.curate",
                        # Cosmos Curator prefers NVENC whenever the NVIDIA runtime
                        # exposes the node's GPU to nvidia-smi.  A CPU-only pod on a
                        # GPU node can pass that probe while CUDA remains blocked by
                        # the pod's device allocation, silently producing empty
                        # clips.  Give the real transcode stage the declared GPU.
                        "resources": "gpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.augment_uri}}",
                                    "schema": "npa.cosmos2.transfer.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.curator_report_uri}}",
                                    "schema": "npa.cosmos_curate.curation.v1",
                                }
                            )
                        ],
                        "next": "curate",
                    }
                ),
                "curate": OrderedDict(
                    {
                        "description": (
                            "Stage 4 - Curation (Voxel51 / FiftyOne). Run real FiftyOne Brain "
                            "curation over the augmented + graded variants (uniqueness + "
                            "near-duplicate detection + keep/drop) when run in the npa-fiftyone "
                            "image, else degrade to the report-only counts path."
                        ),
                        "needs": ["cosmos-curate"],
                        "toolRef": "workbench.fiftyone.curate_augmented",
                        "resources": "cpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.lance_uri}}",
                                    "schema": "npa.cosmos2.transfer.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.curation_report_uri}}",
                                    "schema": "npa.fiftyone.curation.v1",
                                }
                            )
                        ],
                        "next": "visualize",
                    }
                ),
                "visualize": OrderedDict(
                    {
                        "description": (
                            "Build reports/sim2real.rrd from the run's input + augmented frames, "
                            "captions, and per-stage docs for the NPA agent's embedded Rerun viewer."
                        ),
                        "needs": ["curate"],
                        "resources": "cpu",
                        "toolRef": "workbench.nurec.visualize",
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.rrd_uri}}",
                                    "schema": "npa.sim2real.rerun.v1",
                                }
                            )
                        ],
                        "next": "finalize",
                    }
                ),
                "finalize": OrderedDict(
                    {
                        "description": (
                            "Aggregate the run's stage artifacts into a final augmented-and-graded "
                            "dataset report."
                        ),
                        "needs": ["visualize"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "shell": (
                                    'python3 -c "from npa.workflows.data_factory_stages import '
                                    "finalize; finalize('{{config.run_root_uri}}', "
                                    "'{{config.finalize_report_uri}}')\""
                                )
                            }
                        ),
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.finalize_report_uri}}",
                                    "schema": "npa.sim2real.e2e_report.v1",
                                }
                            )
                        ],
                        "terminal": True,
                    }
                ),
            }
        ),
    }


def _normalize_template(template: str) -> str:
    value = str(template or "two-step").strip().lower()
    return _TEMPLATE_ALIASES.get(value, value if value in _TEMPLATES else "two-step")


def choose_workflow_template(
    *,
    user_text: str = "",
    intent: str = "",
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the best workflow template from user intent and capability hints."""
    text = str(user_text or "").lower()
    scores = {name: 0 for name in _TEMPLATES}
    default_template = _INTENT_DEFAULT_TEMPLATE.get(
        str(intent or "").strip(), "two-step"
    )
    scores[default_template] += 3
    if str(intent or "").strip() == "create_vlm_rl_workflow":
        # Explicit VLM-RL / outer+inner loop language reaches the loop template;
        # generic Sim2Real authoring stays on the maintained staged engine.
        if re.search(r"\bvlm\s*[/_-]?\s*rl\b", text) or (
            re.search(r"\bouter[\s-]+loop\b", text)
            and re.search(r"\binner[\s-]+loop\b", text)
        ):
            scores["vlm-rl-loop"] += 12
        else:
            scores["sim2real-staged"] += 10
    for template, keywords in _TEMPLATE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[template] += 2
    # Explicit BYOF language must beat RL/Isaac bleed from the full tool catalog.
    byof_explicit = any(
        token in text
        for token in (
            "byof",
            "bring your own fork",
            "leisaac",
            "lightwheel",
            "bring-your-own",
        )
    )
    if byof_explicit:
        scores["byof"] += 10
    if "outer loop" in text and "inner loop" in text:
        scores["vlm-rl-loop"] += 5
    data_factory_explicit = any(
        token in text
        for token in (
            "paidf",
            "data factory",
            "physical ai data factory",
            "video data augmentation",
        )
    )
    if data_factory_explicit:
        scores["physical-ai-data-factory"] += 10
    if ("augment" in text or "cosmos transfer" in text) and any(
        token in text
        for token in (
            "fan out",
            "fan-out",
            "fanout",
            "multiply",
            "scenario",
            "variant",
            "amplify",
        )
    ):
        scores["physical-ai-data-factory"] += 6
    if re.search(
        r"\b(?:sim2real|sim[\s-]?to[\s-]?real|sim\s*[- ]?2\s*[- ]?real)\b", text
    ):
        scores["sim2real-staged"] += 6
    if re.search(r"\b(?:2[\s-]?step|two[\s-]?step)\b", text):
        scores["two-step"] += 10
    if "rl" in text and ("policy" in text or "training" in text or "isaac" in text):
        if not byof_explicit:
            scores["rl-policy-success"] += 5
    if capabilities and not byof_explicit:
        capabilities_text = " ".join(
            f"{k}:{v}" for k, v in sorted(capabilities.items())
        ).lower()
        if "token" in capabilities_text:
            scores["token-factory-gate"] += 2
        if "vlm" in capabilities_text and "rl" in capabilities_text:
            scores["vlm-rl-loop"] += 2
        if any(k in capabilities_text for k in ("isaac", "policy", "training", "rl")):
            scores["rl-policy-success"] += 2
        if any(k in capabilities_text for k in ("loop", "gate", "transition")):
            scores["loop-gate"] += 1
    selected = sorted(
        scores.items(), key=lambda item: (item[1], item[0]), reverse=True
    )[0][0]
    return {"template": selected, "scores": scores}


_DATA_FACTORY_COUNT_RE = re.compile(
    r"(\d+)\s*(?:different\s+|distinct\s+)?"
    r"(?:scenario\s*(?:variant)?s?|variant|variants|augmentation|augmentations|"
    r"combos?|renders?|versions?|scenes?)",
    re.IGNORECASE,
)
_DATA_FACTORY_FANOUT_RE = re.compile(
    r"(?:fan[\s-]?out|multiply|amplify|fanout)\D{0,20}(\d+)"
    r"|(\d+)[\s-]*(?:way|x)\s*(?:fan[\s-]?out|multiply)",
    re.IGNORECASE,
)
_DATA_FACTORY_GPU_RE = re.compile(
    r"(\d+)\s*(?:x\s*)?gpus?"
    r"|(?:at\s*least|min(?:imum)?|up\s*to)\s*(\d+)\s*(?:x\s*)?gpus?"
    r"|gpus?\D{0,12}(\d+)",
    re.IGNORECASE,
)
_DATA_FACTORY_SUBJECT_RE = re.compile(
    r"augment(?:ing|ed)?\s+(?:my\s+|the\s+|our\s+|these\s+|this\s+)?(.+?)"
    r"(?:\s+(?:and|to|so|then|,|;|\.|with)\b|$)",
    re.IGNORECASE,
)
_PERCENT_RE = r"-?(?:\d+(?:\.\d+)?)\s*%?"

_DATA_FACTORY_MAX_AUGMENTATIONS = 64
_DATA_FACTORY_MAX_GPUS = 8


class WorkflowParameterError(ValueError):
    """A chat parameter was explicit but unsafe or outside supported bounds."""


def _named_number(
    text: str,
    labels: str,
    *,
    integer: bool = False,
    fraction: bool = False,
) -> float | int | None:
    number = r"\d+" if integer else _PERCENT_RE
    patterns = (
        rf"(?:{labels})\b\s*(?:of|to|=|:)?\s*({number})",
        rf"({number})\s*(?:for\s+)?(?:{labels})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).strip()
        if integer:
            return int(raw)
        value = float(raw.rstrip("%"))
        return value / 100.0 if fraction and raw.endswith("%") else value
    return None


def _named_text(text: str, labels: str) -> str:
    # Stop before a following named or numeric parameter clause. This keeps
    # identifiers exact even when two text-valued knobs are adjacent, e.g.
    # ``isaac task X and trigger dataset id Y and 3 rollouts``.
    clause = (
        r"(?=\s*(?:,\s*)?(?:(?:and|with|using|on)\s+)?(?:"
        r"-?\d+(?:\.\d+)?%?\s+(?:environments?|envs?|rollouts?|steps?|gpus?|shards?|iterations?)"
        r"|(?:trigger\s+)?dataset\s+id|isaac(?:\s+lab)?\s+task)\b|[,;\n]|$)"
    )
    match = re.search(
        rf"(?:{labels})\b(?:\s+(?:is|of|to)\b\s*|\s*[=:]\s*|\s+)"
        rf"[\"'`]?(.+?)[\"'`]?{clause}",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else ""


def _named_uri(text: str, labels: str) -> str:
    match = re.search(
        rf"(?:{labels})\s*(?:is|of|to|=|:)?\s*[\"'`]?((?:s3|https?)://[^\s,;\"'`]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).rstrip(".") if match else ""


def _requested_accelerator(text: str) -> str:
    match = re.search(
        r"\b(RTX\s*PRO\s*6000|RTXPRO6000|L40S|H100|H200|B200|B300)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    compact = re.sub(r"[\s_-]+", "", match.group(1)).upper()
    return "RTXPRO6000" if compact in {"RTXPRO6000", "RTX6000PRO"} else compact


def _first_int(match: re.Match[str] | None) -> int | None:
    if not match:
        return None
    for group in match.groups():
        if group:
            try:
                return int(group)
            except ValueError:
                continue
    return None


def extract_data_factory_params(user_text: str) -> dict[str, Any]:
    """Parse a chat request into Physical AI Data Factory knobs.

    Recognizes the fan-out count ("fan out 4 scenarios", "6 variants",
    "multiply by 8"), a GPU count ("use at least 4 GPUs"), and a free-form
    augmentation subject ("augment my warehouse robot clips ..."). Missing
    values are simply omitted so the template defaults apply.
    """
    text = str(user_text or "").strip()
    params: dict[str, Any] = {}
    if not text:
        return params

    count = _first_int(_DATA_FACTORY_FANOUT_RE.search(text))
    if count is None:
        count = _first_int(_DATA_FACTORY_COUNT_RE.search(text))
    gpus = _first_int(_DATA_FACTORY_GPU_RE.search(text))

    if count is not None and count > _DATA_FACTORY_MAX_AUGMENTATIONS:
        raise WorkflowParameterError(
            f"requested {count} augmentations exceeds the supported ceiling "
            f"of {_DATA_FACTORY_MAX_AUGMENTATIONS}"
        )
    if gpus is not None and gpus > _DATA_FACTORY_MAX_GPUS:
        raise WorkflowParameterError(
            f"requested {gpus} GPUs exceeds the supported ceiling of {_DATA_FACTORY_MAX_GPUS}"
        )

    if count is not None and count > 0:
        params["n_augmentations"] = count
    if gpus is not None and gpus > 0:
        params["gpu_count"] = gpus

    refinement = _named_number(
        text, r"refinement(?:\s+iterations?|\s+passes?)", integer=True
    )
    threshold = _named_number(
        text, r"(?:grade|quality)(?:\s+score)?\s+threshold", fraction=True
    )
    clip_len = _named_number(text, r"(?:curator\s+)?clip(?:\s+length|\s+len)?")
    min_clip_len = _named_number(text, r"minimum\s+clip(?:\s+length|\s+len)?")
    max_images = _named_number(text, r"max(?:imum)?\s+images?", integer=True)
    max_tokens = _named_number(text, r"max(?:imum)?\s+tokens?", integer=True)
    if max_images is None:
        max_images = _first_int(
            re.search(r"\bmax(?:imum)?\s+(\d+)\s+images?\b", text, re.I)
        )
    if max_tokens is None:
        max_tokens = _first_int(
            re.search(r"\bmax(?:imum)?\s+(\d+)\s+tokens?\b", text, re.I)
        )
    if isinstance(refinement, int) and refinement > 0:
        params["refinement_iterations"] = refinement
    if isinstance(threshold, float):
        if not 0 <= threshold <= 1:
            raise WorkflowParameterError(
                "grade threshold must be between 0 and 1 (or 0% and 100%)"
            )
        params["grade_threshold"] = threshold
    if isinstance(clip_len, (int, float)) and clip_len > 0:
        params["curator_clip_len_s"] = clip_len
    if isinstance(min_clip_len, (int, float)) and min_clip_len > 0:
        params["curator_min_clip_len_s"] = min_clip_len
    if isinstance(max_images, int) and max_images > 0:
        params["max_images"] = max_images
    if isinstance(max_tokens, int) and max_tokens > 0:
        params["max_tokens"] = max_tokens
    accelerator = _requested_accelerator(text)
    if accelerator:
        params["accelerator"] = accelerator
    if re.search(
        r"\bmotion\s+filter(?:ing)?\b.{0,20}\b(?:off|none|disabled)\b", text, re.I
    ):
        params["curator_motion_filter"] = "disabled"
    elif re.search(
        r"\bmotion\s+filter(?:ing)?\b.{0,20}\b(?:score|score-only)\b", text, re.I
    ):
        params["curator_motion_filter"] = "score-only"

    subject_match = _DATA_FACTORY_SUBJECT_RE.search(text)
    if subject_match:
        subject = subject_match.group(1).strip(" .,\"'`")
        # Drop trailing fan-out/scenario phrasing that leaked into the subject.
        subject = re.sub(
            r"\s+(?:and\s+)?(?:fan[\s-]?out|multiply|amplify).*$",
            "",
            subject,
            flags=re.IGNORECASE,
        ).strip(" .,\"'`")
        if subject and len(subject) <= 120:
            params["augment_subject"] = subject
    return params


_SIM2REAL_URI_FIELDS: tuple[tuple[str, str], ...] = (
    ("trigger_dataset_uri", r"(?:trigger|input)(?:\s+dataset)?\s+uri"),
    ("assets_uri", r"assets?\s+uri"),
    ("scene_spec_uri", r"scene(?:\s+spec)?\s+uri"),
    ("cameras_uri", r"cameras?\s+uri"),
    ("robot_spec_uri", r"robot(?:\s+spec)?\s+uri"),
)


def extract_sim2real_params(user_text: str) -> dict[str, Any]:
    """Extract knobs supported by the maintained staged Sim2Real CLI."""
    text = str(user_text or "").strip()
    params: dict[str, Any] = {}
    if not text:
        return params
    integer_fields = {
        "inner_iterations": r"inner(?:\s+loop)?\s+iterations?",
        "outer_iterations": r"outer(?:\s+loop)?\s+iterations?",
        "loop_of_loops_iterations": r"loop[-\s]+of[-\s]+loops?\s+iterations?",
        "rollout_count": r"(?:train\s+)?rollouts?",
        "steps_per_rollout": r"(?:steps?\s+per\s+rollout|rollout\s+length|(?:train(?:ing)?)\s+steps?)",
        "heldout_env_count": r"held[-\s]?out\s+env(?:ironment)?s?",
        "env_count": r"(?:generated\s+)?(?:environments?|envs?)",
        "envgen_shard_count": r"env(?:ironment)?(?:gen)?\s+shards?",
        "action_env_limit": r"action\s+env(?:ironment)?s?",
        "seed": r"(?:random\s+)?seed",
    }
    for field, labels in integer_fields.items():
        value = _named_number(text, labels, integer=True)
        if isinstance(value, int) and (value > 0 or field == "seed"):
            params[field] = value
    threshold = _named_number(
        text, r"(?:(?:success|held[-\s]?out|evaluation)\s+)?threshold", fraction=True
    )
    train_fraction = _named_number(text, r"train(?:ing)?\s+fraction", fraction=True)
    if isinstance(threshold, float):
        if not 0 <= threshold <= 1:
            raise WorkflowParameterError(
                "success threshold must be between 0 and 1 (or 0% and 100%)"
            )
        params["success_threshold"] = threshold
    if isinstance(train_fraction, float):
        if not 0 < train_fraction < 1:
            raise WorkflowParameterError(
                "training fraction must be greater than 0 and less than 1"
            )
        params["train_fraction"] = train_fraction
    backend = re.search(r"\b(isaac|genesis)\b(?:\s+(?:sim|backend))?", text, re.I)
    if backend:
        params["sim_backend"] = backend.group(1).lower()
    robot = re.search(r"\b(franka|ur5e|ur10e|flexiv)\b", text, re.I)
    if robot:
        params["robot_preset"] = robot.group(1).lower()
    accelerator = _requested_accelerator(text)
    if accelerator:
        params["accelerator"] = accelerator
    gpu_count = _first_int(_DATA_FACTORY_GPU_RE.search(text))
    if gpu_count and gpu_count > 0:
        params["gpu_count"] = gpu_count
    for field, labels in _SIM2REAL_URI_FIELDS:
        uri_value = _named_uri(text, labels)
        if uri_value:
            params[field] = uri_value
    task = _named_text(text, r"isaac(?:\s+lab)?\s+task")
    if task:
        params["isaac_task"] = task
    dataset_id = _named_text(text, r"(?:trigger\s+)?dataset\s+id")
    if dataset_id:
        params["trigger_dataset_id"] = dataset_id
    return params


_DATA_FACTORY_DEFAULT_GPUS = 4


def _data_factory_gpu_count(
    config: OrderedDict[str, Any], params: dict[str, Any]
) -> int:
    """Resolve the GPU accelerator count for a paidf run (>=1, <=8)."""
    gpus = params.get("gpu_count")
    if gpus:
        return max(1, int(gpus))
    return _DATA_FACTORY_DEFAULT_GPUS


def _apply_data_factory_params(
    config: OrderedDict[str, Any], params: dict[str, Any]
) -> None:
    """Overlay parsed chat knobs onto a paidf config (in place).

    Keeps ``variant_parallelism`` <= the GPU accelerator count so the fan-out
    never pins a variant to a GPU the pod does not have.
    """
    n_aug = params.get("n_augmentations")
    subject = params.get("augment_subject")
    if n_aug:
        config["n_augmentations"] = str(int(n_aug))
    if subject:
        config["augment_subject"] = str(subject)
    for key in (
        "refinement_iterations",
        "grade_threshold",
        "curator_clip_len_s",
        "curator_min_clip_len_s",
        "curator_motion_filter",
        "max_images",
        "max_tokens",
    ):
        if key in params:
            value = params[key]
            config[key] = (
                str(int(value))
                if isinstance(value, float) and value.is_integer()
                else str(value)
            )
    resolved_variants = int(str(config.get("n_augmentations") or "1"))
    gpu_count = _data_factory_gpu_count(config, params)
    config["variant_parallelism"] = str(max(1, min(resolved_variants, gpu_count)))


def _apply_sim2real_params(
    config: OrderedDict[str, Any], params: dict[str, Any]
) -> None:
    """Overlay only knobs exposed by the canonical compositional spec."""

    if "trigger_dataset_uri" in params and "trigger_uri" in config:
        config["trigger_uri"] = str(params["trigger_dataset_uri"])
    if "seed" in params and "envgen_seed" in config:
        config["envgen_seed"] = str(params["seed"])
    if "success_threshold" in params and "threshold" in config:
        config["threshold"] = str(params["success_threshold"])
    if "heldout_env_count" in params:
        for key in ("validation_count", "gold_count"):
            if key in config:
                config[key] = str(params["heldout_env_count"])
    if "envgen_shard_count" in params and "shard_count" in config:
        config["shard_count"] = str(params["envgen_shard_count"])
    for key in (
        "trigger_dataset_uri",
        "trigger_dataset_id",
        "assets_uri",
        "scene_spec_uri",
        "cameras_uri",
        "robot_spec_uri",
        "robot_source",
        "robot_preset",
        "sim_backend",
        "isaac_task",
        "vlm_model",
        "success_threshold",
        "inner_iterations",
        "outer_iterations",
        "loop_of_loops_iterations",
        "rollout_count",
        "steps_per_rollout",
        "heldout_env_count",
        "env_count",
        "train_fraction",
        "envgen_shard_count",
        "action_env_limit",
        "seed",
    ):
        if key in params and key in config:
            config[key] = str(params[key])


def _infra_entry_key(
    entry: dict[str, Any], project: str = ""
) -> tuple[int, int, int, str, str]:
    """Stable preference: explicit default, usable kubeconfig/context, then name."""
    name = str(entry.get("cluster_name") or entry.get("name") or "")
    project_match = bool(project and project.lower() in name.lower())
    return (
        0
        if bool(
            entry.get("selected") or entry.get("default") or entry.get("is_default")
        )
        else 1,
        0 if project_match else 1,
        0 if str(entry.get("context") or entry.get("kubeconfig") or "").strip() else 1,
        name,
        str(entry.get("context") or ""),
    )


def resolve_workflow_infrastructure(
    infrastructure: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select deterministic, real, non-secret workflow placement facts.

    Configured backends take precedence over locally cached kubeconfigs, which
    take precedence over cloud discovery. Every fallback is based on an actual
    inventory entry; this function never synthesizes a cluster identifier.
    """
    payload = infrastructure if isinstance(infrastructure, dict) else {}
    project = str(payload.get("project") or "").strip()
    configured = payload.get("configured")
    entries = (
        [item for item in configured if isinstance(item, dict)]
        if isinstance(configured, list)
        else []
    )
    source = "configured"
    reason = ""
    if entries:
        entries.sort(key=lambda item: _infra_entry_key(item, project))
        entry = entries[0]
        reason = f"selected configured backend deterministically from {len(entries)} candidate(s)"
    else:
        local = payload.get("local_clusters")
        local_entries = (
            [
                item
                for item in local
                if isinstance(item, dict) and bool(item.get("kubeconfig_exists"))
            ]
            if isinstance(local, list)
            else []
        )
        if local_entries:
            local_entries.sort(key=lambda item: _infra_entry_key(item, project))
            entry = local_entries[0]
            source = "local"
            reason = f"selected usable cached kubeconfig from {len(local_entries)} candidate(s)"
        else:
            cloud = payload.get("cloud_clusters")
            cloud_entries = (
                [item for item in cloud if isinstance(item, dict)]
                if isinstance(cloud, list)
                else []
            )
            cloud_entries.sort(key=lambda item: _infra_entry_key(item, project))
            entry = cloud_entries[0] if cloud_entries else {}
            source = "cloud" if entry else "none"
            reason = (
                "identified cloud cluster from live discovery; configure its context/kubeconfig before submit"
                if entry
                else "no configured, local, or cloud Kubernetes backend is available"
            )
    raw_value = entry.get("raw")
    raw: dict[str, Any] = raw_value if isinstance(raw_value, dict) else {}
    available_raw = raw.get("available_accelerators")
    available = (
        [str(value).strip() for value in available_raw if str(value).strip()]
        if isinstance(available_raw, list)
        else []
    )
    raw_accelerators = raw.get("accelerators")
    accelerator = str(
        raw.get("gpu_accelerator")
        or (raw_accelerators if isinstance(raw_accelerators, str) else "")
        or raw.get("gpu_product")
        or (available[0] if len(available) == 1 else "")
        or ""
    ).strip()
    profile = (
        str(entry.get("gpu_profile") or raw.get("gpu_profile") or "").strip().lower()
    )
    if not accelerator and profile in {"rtxpro", "rtx6000", "rtx-pro"}:
        accelerator = "RTXPRO6000"
    return {
        "project": project,
        "cluster_name": str(
            entry.get("cluster_name") or entry.get("name") or ""
        ).strip(),
        "context": str(entry.get("context") or "").strip(),
        "kubeconfig": str(entry.get("kubeconfig") or "").strip(),
        "accelerator": accelerator,
        "available_accelerators": available,
        "gpu_profile": profile,
        "source": source,
        "selection_reason": reason,
        "k8s_namespace": str(
            raw.get("namespace") or raw.get("k8s_namespace") or ""
        ).strip(),
        "k8s_service_account": str(
            raw.get("service_account") or raw.get("k8s_service_account") or ""
        ).strip(),
        "k8s_image_pull_secrets": str(
            raw.get("image_pull_secrets") or raw.get("k8s_image_pull_secrets") or ""
        ).strip(),
        "k8s_env_secret_names": str(
            raw.get("env_secret_names") or raw.get("k8s_env_secret_names") or ""
        ).strip(),
        "k8s_gpu_product": str(
            raw.get("gpu_product") or raw.get("k8s_gpu_product") or ""
        ).strip(),
        "augment_image": str(raw.get("augment_image") or "").strip(),
        "envgen_image": str(raw.get("envgen_image") or "").strip(),
        "policy_image": str(raw.get("policy_image") or "").strip(),
        "trainer_image": str(raw.get("trainer_image") or "").strip(),
        "vlm_image": str(raw.get("vlm_image") or "").strip(),
        "eval_image": str(raw.get("eval_image") or "").strip(),
        "isaac_image": str(raw.get("isaac_image") or "").strip(),
    }


def _gpu_product_for_accelerator(accelerator: str) -> str:
    family = _accelerator_family(accelerator)
    if family == "RTXPRO6000":
        return "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
    if family == "L40S":
        return "NVIDIA-L40S"
    return ""


def _accelerator_with_count(accelerator: str, count: int) -> str:
    base = str(accelerator or "").strip().split(":", 1)[0]
    # SkyPilot derives this requestable name from the live NVIDIA product label;
    # keep the compact family spelling for validation and use the discovered
    # Kubernetes catalog spelling at the scheduling boundary.
    if _accelerator_family(base) == "RTXPRO6000":
        base = "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
    return f"{base}:{max(1, int(count))}" if base else ""


def _accelerator_family(accelerator: str) -> str:
    compact = re.sub(
        r"[\s_-]+",
        "",
        str(accelerator or "").split(":", 1)[0],
    ).upper()
    if (
        compact == "RTX6000"
        or compact.startswith("RTXPRO6000")
        or "RTX6000BLACKWELL" in compact
    ):
        return "RTXPRO6000"
    return compact


def _apply_workflow_infrastructure(
    resources: OrderedDict[str, Any],
    *,
    template: str,
    params: dict[str, Any],
    infrastructure: dict[str, Any] | None,
) -> dict[str, str]:
    resolved = resolve_workflow_infrastructure(infrastructure)
    gpu = resources.get("gpu")
    if not isinstance(gpu, dict):
        return resolved
    current = str(gpu.get("accelerators") or "")
    current_count = 1
    if ":" in current:
        try:
            current_count = int(current.rsplit(":", 1)[1])
        except ValueError:
            current_count = 1
    count = int(params.get("gpu_count") or current_count)
    requested = str(params.get("accelerator") or "").strip()
    configured = str(resolved.get("accelerator") or "").strip()
    accelerator = (
        requested or configured or (current if params.get("gpu_count") else "")
    )
    if accelerator:
        gpu["accelerators"] = _accelerator_with_count(accelerator, count)
    elif infrastructure is not None:
        placeholder = (
            "<configure-rt-core-accelerator>"
            if template == "sim2real-staged"
            else "<configure-gpu-accelerator>"
        )
        gpu["accelerators"] = f"{placeholder}:{count}"
    cluster_name = str(resolved.get("cluster_name") or "").strip()
    context = str(resolved.get("context") or "").strip()
    project = str(resolved.get("project") or "").strip()
    if cluster_name or context:
        directive: OrderedDict[str, Any] = OrderedDict()
        directive["clusterName"] = cluster_name or context
        if context:
            directive["context"] = context
        if project:
            directive["project"] = project
        directive["skipS3"] = True
        gpu["deployIfAbsent"] = directive
    return resolved


def _build_spec(
    template: str,
    *,
    bucket: str,
    name: str | None,
    params: dict[str, Any] | None = None,
    infrastructure: dict[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    normalized = _normalize_template(template)
    if normalized in {"vlm-rl-loop", "sim2real-staged"}:
        canonical = _canonical_sim2real_spec(bucket=bucket, name=name)
        if params:
            _apply_sim2real_params(canonical["config"], params)
        return canonical
    catalog = _workflow_specs()
    spec = catalog[normalized]
    metadata_name = str(name or spec["name"])
    description = _FoldedStr(str(spec["description"]))
    config = OrderedDict({"bucket": str(bucket)})
    config.update(spec["config_runtime"])
    config.update(spec["config_uri"])
    if normalized == "physical-ai-data-factory" and params:
        _apply_data_factory_params(config, params)
    if normalized in {"sim2real-staged", "two-step", "vlm-rl-loop"} and params:
        _apply_sim2real_params(config, params)
    states = OrderedDict()
    for state_name, state_spec in spec["states"].items():
        state_payload: OrderedDict[str, Any] = OrderedDict()
        for key, value in state_spec.items():
            if key == "description":
                state_payload[key] = _FoldedStr(str(value))
            elif key == "run" and isinstance(value, dict):
                run_payload: OrderedDict[str, Any] = OrderedDict()
                for run_key, run_value in value.items():
                    if (
                        run_key == "shell"
                        and isinstance(run_value, str)
                        and "\n" in run_value
                    ):
                        run_payload[run_key] = _LiteralStr(run_value)
                    else:
                        run_payload[run_key] = run_value
                state_payload[key] = run_payload
            else:
                state_payload[key] = value
        states[state_name] = state_payload
    root: OrderedDict[str, Any] = OrderedDict()
    root["apiVersion"] = API_VERSION
    root["kind"] = "Workflow"
    root["metadata"] = OrderedDict({"name": metadata_name, "description": description})
    root["config"] = config
    resources = deepcopy(spec["resources"])
    if normalized == "physical-ai-data-factory" and params:
        gpu_count = _data_factory_gpu_count(config, params)
        gpu_res = resources.get("gpu")
        if isinstance(gpu_res, dict) and gpu_count != _DATA_FACTORY_DEFAULT_GPUS:
            gpu_res["accelerators"] = (
                f"RTXPRO-6000-BLACKWELL-SERVER-EDITION:{gpu_count}"
            )
    resolved_infra = _apply_workflow_infrastructure(
        resources,
        template=normalized,
        params=params or {},
        infrastructure=infrastructure,
    )
    if normalized == "sim2real-staged":
        accelerator = str(
            (params or {}).get("accelerator") or resolved_infra.get("accelerator") or ""
        )
        for key in (
            "k8s_namespace",
            "k8s_service_account",
            "k8s_image_pull_secrets",
            "k8s_env_secret_names",
            "augment_image",
            "envgen_image",
            "policy_image",
            "trainer_image",
            "vlm_image",
            "eval_image",
            "isaac_image",
        ):
            if resolved_infra.get(key):
                config[key] = resolved_infra[key]
        config["k8s_gpu_product"] = resolved_infra.get(
            "k8s_gpu_product"
        ) or _gpu_product_for_accelerator(accelerator)
    root["resources"] = resources
    root["initial"] = spec["initial"]
    root["states"] = states
    return root


def _canonical_sim2real_spec(*, bucket: str, name: str | None) -> OrderedDict[str, Any]:
    """Load the one operator Sim2Real graph instead of generating a stub twin."""

    if _EMBEDDED_CANONICAL_SIM2REAL_YAML:
        payload = yaml.safe_load(_EMBEDDED_CANONICAL_SIM2REAL_YAML)
    else:
        here = Path(__file__).resolve()
        candidates = (
            here.parents[3]
            / "workflows"
            / "workbench"
            / "npa-workflows"
            / "sim2real.yaml",
            here.parents[1]
            / "workflows"
            / "workbench"
            / "npa-workflows"
            / "sim2real.yaml",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is None:
            raise FileNotFoundError("canonical packaged sim2real.yaml is missing")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["metadata"]["name"] = str(name or "sim2real")
    payload["config"]["bucket"] = str(bucket)
    return OrderedDict(payload)


def _insert_config_spacing(yaml_text: str) -> str:
    lines = yaml_text.splitlines()
    first_uri_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s{2}[A-Za-z0-9_-]*_uri:\s", line):
            first_uri_idx = idx
            break
    if (
        first_uri_idx is not None
        and first_uri_idx > 0
        and lines[first_uri_idx - 1].strip()
    ):
        lines.insert(first_uri_idx, "")
    return "\n".join(lines).rstrip() + "\n"


def _render_spec_yaml(spec: OrderedDict[str, Any]) -> str:
    rendered = yaml.dump(
        _to_builtin(spec), Dumper=_WorkflowDumper, sort_keys=False, width=96
    )
    return _insert_config_spacing(rendered)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, OrderedDict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def generate_workflow_yaml(
    template: str = "two-step", *, bucket: str = "example-bucket"
) -> str:
    """Render npa.workflow YAML from a declarative template catalog."""
    normalized = _normalize_template(template)
    spec = _build_spec(normalized, bucket=bucket, name=None)
    return _render_spec_yaml(spec)


def generate_workflow_draft(
    *,
    user_text: str = "",
    intent: str = "",
    template: str = "",
    bucket: str = "example-bucket",
    name: str = "",
    capabilities: dict[str, Any] | None = None,
    tool_refs: frozenset[str] | None = None,
    infrastructure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Draft workflow YAML by selecting a template from intent/capabilities."""
    if template:
        selected_template = _normalize_template(template)
        selection = {"template": selected_template, "scores": {selected_template: 1}}
    else:
        selection = choose_workflow_template(
            user_text=user_text, intent=intent, capabilities=capabilities
        )
        selected_template = str(selection["template"])
    parameter_errors: list[str] = []
    try:
        if selected_template == "physical-ai-data-factory":
            params = extract_data_factory_params(user_text)
        elif selected_template in {"sim2real-staged", "two-step", "vlm-rl-loop"}:
            params = extract_sim2real_params(user_text)
        else:
            params = None
    except WorkflowParameterError as exc:
        params = None
        parameter_errors.append(str(exc))
    resolved_infra = resolve_workflow_infrastructure(infrastructure)
    warnings: list[str] = []
    context_errors: list[str] = list(parameter_errors)
    resolved_bucket = str(bucket or "").strip()
    if not resolved_bucket:
        resolved_bucket = "<configure-s3-bucket>"
        warnings.append(
            "No agent S3 bucket is configured; config.bucket is an explicit placeholder."
        )
    requested_accel = str((params or {}).get("accelerator") or "").strip()
    configured_accel = str(resolved_infra.get("accelerator") or "").strip()
    available_accels = {
        _accelerator_family(str(value))
        for value in (resolved_infra.get("available_accelerators") or [])
        if str(value).strip()
    }
    if requested_accel and available_accels:
        requested_base = _accelerator_family(requested_accel)
        if requested_base not in available_accels:
            context_errors.append(
                f"requested accelerator {requested_base} is unavailable on the selected "
                f"backend (available: {', '.join(sorted(available_accels))})"
            )
    if requested_accel and configured_accel:
        requested_base = _accelerator_family(requested_accel)
        configured_base = _accelerator_family(configured_accel)
        if requested_base != configured_base:
            context_errors.append(
                f"requested accelerator {requested_base} is not the configured profile {configured_base}"
            )
    if (
        selected_template == "sim2real-staged"
        and str((params or {}).get("sim_backend") or "isaac") == "isaac"
    ):
        selected_accel = (requested_accel or configured_accel).upper()
        if selected_accel and any(
            name in selected_accel for name in ("H100", "H200", "B200", "B300")
        ):
            context_errors.append(
                "Isaac Sim2Real requires an RT-core accelerator (L40S or RTX PRO 6000)"
            )
    if infrastructure is not None and not bool((infrastructure or {}).get("has_infra")):
        warnings.append(
            "No Kubernetes backend is currently configured; provision or select one before submit."
        )
    elif infrastructure is not None and not configured_accel and not requested_accel:
        warnings.append(
            "The configured Kubernetes backend does not declare an accelerator; "
            "the generated resource uses an explicit placeholder."
        )
    if selected_template == "sim2real-staged" and infrastructure is not None:
        if not bool((infrastructure or {}).get("has_infra")):
            context_errors.append(
                "a configured Kubernetes backend is required before Sim2Real submit"
            )
        elif not configured_accel and not requested_accel:
            context_errors.append(
                "the selected Kubernetes backend must declare an RT-core accelerator"
            )
    selection_reason = str(resolved_infra.get("selection_reason") or "").strip()
    if selection_reason:
        warnings.append(selection_reason)
    spec = _build_spec(
        selected_template,
        bucket=resolved_bucket,
        name=name or None,
        params=params,
        infrastructure=infrastructure,
    )
    yaml_text = _render_spec_yaml(spec)
    unresolved = sorted(set(re.findall(r"<[^<>\n]+>", yaml_text)))
    if unresolved:
        context_errors.append(
            "unresolved configuration placeholders: " + ", ".join(unresolved)
        )
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=tool_refs)
    plan: dict[str, Any]
    if validation.get("ok"):
        plan = plan_workflow_yaml_text(
            yaml_text,
            run_id=f"draft-{selected_template}",
            tool_refs=tool_refs,
        )
    else:
        plan = {
            "ok": False,
            "error": str(validation.get("error") or "validation failed"),
        }
    runnable = bool(validation.get("ok") and plan.get("ok") and not context_errors)
    return {
        "template": selected_template,
        "selection": selection,
        "yaml": yaml_text,
        "validation": validation,
        "plan": plan,
        "runnable": runnable,
        "parameters": params or {},
        "infrastructure": resolved_infra,
        "warnings": warnings,
        "context_errors": context_errors,
    }


_CONFIG_TOKEN_RE = re.compile(r"\{\{\s*config\.([a-zA-Z0-9_.-]+)\s*\}\}")
_STEP_COUNT_RE = re.compile(
    r"\b(\d+)[\s-]?step\b|\b(one|two|three|four|five|six)[\s-]?step\b", re.IGNORECASE
)
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_WORKFLOW_NAME_RE = re.compile(
    r"(?i)\b(?:name\s+it|named|called|name\s*[:=])\s+[\"'`]?([a-zA-Z][a-zA-Z0-9_.-]{0,63})[\"'`]?"
)
_AUTHOR_STOPWORDS = frozenset(
    {
        "write",
        "me",
        "a",
        "an",
        "the",
        "step",
        "steps",
        "npa",
        "yaml",
        "spec",
        "workflow",
        "pipeline",
        "that",
        "uses",
        "use",
        "using",
        "with",
        "and",
        "for",
        "to",
        "of",
        "create",
        "generate",
        "build",
        "make",
        "draft",
        "compose",
        "please",
        "give",
        "show",
        "new",
        "simple",
        "minimal",
        "example",
    }
)


def _slugify_workflow_name(raw: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:64]


def extract_workflow_name(goal: str) -> str:
    """Extract an explicit workflow name from goal text (e.g. 'name it cosmos-video-aug')."""
    match = _WORKFLOW_NAME_RE.search(str(goal or ""))
    if not match:
        return ""
    return _slugify_workflow_name(match.group(1))


def _infer_stage_count_from_goal(goal: str) -> int:
    """Infer how many stages the operator described (arrows / then / commas)."""
    text = str(goal or "").strip()
    if not text:
        return 0
    # Prefer arrow-separated pipelines: "generate → augment → ingest"
    if re.search(r"[→⟶➨]|->", text):
        parts = re.split(r"\s*(?:[→⟶➨]|->)\s*", text)
        stages = [p.strip() for p in parts if p.strip()]
        if len(stages) >= 2:
            return max(1, min(len(stages), 6))
    # "A then B then C"
    then_parts = re.split(r"\bthen\b", text, flags=re.IGNORECASE)
    if len(then_parts) >= 3:
        return max(1, min(len(then_parts), 6))
    return 0


def _desired_step_count(goal: str, default: int = 2) -> int:
    match = _STEP_COUNT_RE.search(str(goal or ""))
    if match:
        if match.group(1):
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                value = default
        else:
            value = _WORD_NUM.get((match.group(2) or "").lower(), default)
        return max(1, min(value, 6))
    inferred = _infer_stage_count_from_goal(goal)
    if inferred:
        return inferred
    return default


def _author_goal_keywords(goal: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", str(goal or "").lower())
    return [w for w in words if w not in _AUTHOR_STOPWORDS and len(w) > 2]


def _select_author_tool_refs(
    goal: str, tool_refs: frozenset[str] | set[str] | list[str] | None, n: int
) -> tuple[list[str], list[str]]:
    """Pick ``n`` real catalog toolRefs, preferring goal-keyword matches.

    Composes only from the live catalog (never an inline copy of tool names):
    tools are ranked by how many goal keywords appear in the toolRef string, then
    padded deterministically from the remaining catalog so the spec has n states.
    """
    catalog = sorted(str(t) for t in (tool_refs or []))
    keywords = _author_goal_keywords(goal)

    def _score(ref: str) -> int:
        low = ref.lower()
        return sum(1 for kw in keywords if kw in low)

    matched = [ref for ref in catalog if _score(ref) > 0]
    matched.sort(key=lambda ref: (-_score(ref), ref))
    selected = list(matched[:n])
    if len(selected) < n:
        for ref in catalog:
            if ref not in selected:
                selected.append(ref)
                if len(selected) >= n:
                    break
    return selected[:n], matched


_AUTHOR_SEMANTIC_TERMS: dict[str, tuple[str, ...]] = {
    "curate": ("curate", "curation", "slice", "filter", "refine", "clean"),
    "train": ("train", "training", "fit", "fitting"),
    "eval": (
        "eval",
        "evaluate",
        "evaluation",
        "benchmark",
        "benchmarking",
        "score",
    ),
    "ingest": ("ingest", "import", "load"),
    "augment": ("augment", "augmentation", "transfer"),
    "generate": ("generate", "generation", "synthesize"),
}


def _author_semantic_stages(goal: str, n_steps: int | None = None) -> list[str]:
    """Return ordered semantic stages without requiring count/clause equality."""
    text = str(goal or "").strip()
    if re.search(r"[→⟶➨]|->", text):
        parts = re.split(r"\s*(?:[→⟶➨]|->)\s*", text)
    elif re.search(r"\bthen\b", text, flags=re.IGNORECASE):
        parts = re.split(r"\bthen\b", text, flags=re.IGNORECASE)
    else:
        mentions: list[tuple[int, str]] = []
        lowered = text.lower()
        for canonical, terms in _AUTHOR_SEMANTIC_TERMS.items():
            positions = [
                match.start()
                for term in terms
                if (match := re.search(rf"\b{re.escape(term)}\b", lowered))
            ]
            if positions:
                mentions.append((min(positions), canonical))
        mentions.sort()
        return (
            [canonical for _, canonical in mentions][:6] if len(mentions) >= 2 else []
        )
    stages = [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]
    return stages[:6]


def _semantic_stage_score(stage: str, tool_ref: str) -> int:
    """Score one live-catalog entry against one ordered goal clause."""
    keywords = set(_author_goal_keywords(stage))
    stage_words = set(re.findall(r"[a-z0-9_]+", str(stage or "").lower()))
    for canonical, terms in _AUTHOR_SEMANTIC_TERMS.items():
        if canonical in stage_words or any(term in stage_words for term in terms):
            keywords.update(terms)
    ref = str(tool_ref or "").lower()
    ref_words = set(re.findall(r"[a-z0-9_]+", ref))
    description = _describe_tool_ref(tool_ref).lower()
    description_words = set(re.findall(r"[a-z0-9_]+", description))
    score = 0
    for keyword in keywords:
        if keyword in ref_words:
            score += 12
        elif keyword in ref:
            score += 7
        if keyword in description_words:
            score += 4
        elif keyword in description:
            score += 2
    return score


def _flow_binding_kind(key: str, flag: str, role: str) -> str:
    text = f"{key} {flag}".lower()
    if (
        "checkpoint" in text
        or "policy" in text
        or (role == "output" and "train" in text)
    ):
        return "checkpoint"
    if any(term in text for term in ("eval", "report", "metric", "score", "benchmark")):
        return "report"
    if any(term in text for term in ("dataset", "data_", "data-", "manifest", "curat")):
        return "dataset"
    if any(term in text for term in ("rollout", "episode", "trajectory")):
        return "rollout"
    if any(term in text for term in ("image", "video", "frame")):
        return "media"
    return "artifact"


def _tool_flow_bindings(tool_ref: str) -> list[dict[str, str]]:
    """Read artifact-like input/output config bindings from a catalog argv."""
    from npa.orchestration.npa_workflow.catalog import argv_for_tool

    try:
        argv = argv_for_tool(tool_ref)
    except Exception:  # noqa: BLE001
        return []
    bindings: list[dict[str, str]] = []
    for index, token in enumerate(argv):
        match = re.fullmatch(r"\{\{\s*config\.([a-zA-Z0-9_.-]+)\s*\}\}", str(token))
        if not match:
            continue
        key = match.group(1)
        previous = str(argv[index - 1]) if index else ""
        flag = previous.lower() if previous.startswith("--") else ""
        output_flag = (
            "output" in flag
            or flag.startswith("--out-")
            or flag
            in {
                "--artifacts-s3-uri",
                "--checkpoint-s3-uri",
                "--rollouts-s3-uri",
                "--destination",
                "--save-path",
            }
        )
        input_flag = any(
            marker in flag
            for marker in (
                "input",
                "source",
                "checkpoint",
                "dataset",
                "data-path",
                "artifact",
                "manifest",
                "view",
            )
        )
        role = "output" if output_flag else "input" if input_flag else ""
        if not role:
            continue
        bindings.append(
            {
                "key": key,
                "flag": flag,
                "role": role,
                "kind": _flow_binding_kind(key, flag, role),
            }
        )
    return bindings


def _best_flow_link(upstream: str, downstream: str) -> dict[str, Any]:
    outputs = [
        binding
        for binding in _tool_flow_bindings(upstream)
        if binding["role"] == "output"
    ]
    inputs = [
        binding
        for binding in _tool_flow_bindings(downstream)
        if binding["role"] == "input"
    ]
    best: dict[str, Any] = {}
    best_score = 0
    for output in outputs:
        for input_binding in inputs:
            score = 0
            if output["kind"] == input_binding["kind"]:
                score += 24
            if output["key"] == input_binding["key"]:
                score += 48
            if not score:
                continue
            if score > best_score:
                best_score = score
                best = {
                    "from_tool": upstream,
                    "to_tool": downstream,
                    "output_config": output["key"],
                    "input_config": input_binding["key"],
                    "artifact_kind": output["kind"],
                    "score": score,
                }
    return best


def _flow_transition_score(upstream: str, downstream: str) -> int:
    link = _best_flow_link(upstream, downstream)
    score = int(link.get("score") or -20)
    if upstream.rsplit(".", 1)[0] == downstream.rsplit(".", 1)[0]:
        score += 18
    return score


def _select_semantic_tool_refs(
    goal: str, catalog: frozenset[str], n_steps: int
) -> list[str]:
    """Choose the highest-scoring coherent path for ordered stage clauses."""
    stages = _author_semantic_stages(goal, n_steps)
    if not stages:
        return []
    candidates: list[list[tuple[int, str]]] = []
    for stage in stages:
        ranked = [(_semantic_stage_score(stage, ref), ref) for ref in catalog]
        ranked = sorted(
            (item for item in ranked if item[0] > 0),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked:
            return []
        candidates.append(ranked[:24])

    paths: list[tuple[int, list[str]]] = [
        (score, [ref]) for score, ref in candidates[0]
    ]
    for stage_candidates in candidates[1:]:
        next_paths: list[tuple[int, list[str]]] = []
        for path_score, path in paths:
            for stage_score, ref in stage_candidates:
                if ref in path:
                    continue
                if not _best_flow_link(path[-1], ref):
                    continue
                score = path_score + stage_score + _flow_transition_score(path[-1], ref)
                next_paths.append((score, [*path, ref]))
        if not next_paths:
            return []
        next_paths.sort(key=lambda item: (-item[0], item[1]))
        paths = next_paths[:96]
    return paths[0][1] if paths else []


def _workflow_flow_links(selected: list[str]) -> list[dict[str, Any]]:
    return [
        link
        for upstream, downstream in zip(selected, selected[1:])
        if (link := _best_flow_link(upstream, downstream))
    ]


def _author_placeholder_for(key: str) -> str:
    low = key.lower()
    if low == "bucket":
        return "example-bucket"
    if low == "prefix":
        return "npa-workflow/{{run.id}}"
    if (
        low.endswith("_uri")
        or low.endswith("_path")
        or "uri" in low
        or low == "output_root"
    ):
        return "s3://{{config.bucket}}/{{config.prefix}}/" + key + "/"
    if any(
        tok in low
        for tok in (
            "count",
            "iterations",
            "num",
            "size",
            "timeout",
            "interval",
            "episodes",
            "steps",
        )
    ):
        return "1"
    return "<" + key + ">"


def _state_name_for(tool_ref: str, index: int, taken: set[str]) -> str:
    tail = str(tool_ref or "").split(".")[-1] or f"step{index + 1}"
    base = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-") or f"step{index + 1}"
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}-{suffix}"
        suffix += 1
    taken.add(name)
    return name


def _build_authored_spec(
    selected: list[str],
    config_keys: list[str],
    *,
    bucket: str,
    name: str,
    matched: set[str] | frozenset[str] | None = None,
    flow_links: list[dict[str, Any]] | None = None,
) -> OrderedDict[str, Any]:
    matched_set = set(matched or ())
    resolved_links = list(flow_links or [])

    config: OrderedDict[str, Any] = OrderedDict()
    config["bucket"] = str(bucket)
    config["prefix"] = "npa-workflow/{{run.id}}"
    for key in config_keys:
        if key not in config:
            config[key] = _author_placeholder_for(key)
    for link in resolved_links:
        output_key = str(link.get("output_config") or "")
        input_key = str(link.get("input_config") or "")
        if output_key and input_key and output_key != input_key:
            config[input_key] = "{{config." + output_key + "}}"

    taken: set[str] = set()
    state_names = [_state_name_for(ref, idx, taken) for idx, ref in enumerate(selected)]
    states: OrderedDict[str, Any] = OrderedDict()
    for idx, (ref, state_name) in enumerate(zip(selected, state_names)):
        try:
            entry_desc = _describe_tool_ref(ref)
        except Exception:  # noqa: BLE001
            entry_desc = f"Run {ref}."
        # Steps that did not match a goal keyword are padding to reach the
        # requested step count — flag them so the operator replaces them rather
        # than mistaking an arbitrary catalog tool for an intended step.
        if ref not in matched_set:
            entry_desc = f"[placeholder — no goal match; replace with the intended tool] {entry_desc}"
        state: OrderedDict[str, Any] = OrderedDict()
        state["description"] = _FoldedStr(entry_desc)
        if idx > 0:
            state["needs"] = [state_names[idx - 1]]
        state["toolRef"] = ref
        incoming = next(
            (link for link in resolved_links if link.get("to_tool") == ref), None
        )
        outgoing = next(
            (link for link in resolved_links if link.get("from_tool") == ref), None
        )
        if incoming:
            upstream_key = str(incoming.get("output_config") or "")
            if upstream_key:
                state["inputs"] = [
                    OrderedDict({"uri": "{{config." + upstream_key + "}}"})
                ]
        output_key = str((outgoing or {}).get("output_config") or "")
        if not output_key:
            output_key = next(
                (
                    binding["key"]
                    for binding in _tool_flow_bindings(ref)
                    if binding["role"] == "output"
                ),
                "",
            )
        if output_key:
            state["outputs"] = [OrderedDict({"uri": "{{config." + output_key + "}}"})]
        if idx < len(selected) - 1:
            state["next"] = state_names[idx + 1]
        else:
            state["terminal"] = True
        states[state_name] = state

    root: OrderedDict[str, Any] = OrderedDict()
    root["apiVersion"] = API_VERSION
    root["kind"] = "Workflow"
    root["metadata"] = OrderedDict(
        {
            "name": str(name),
            "description": _FoldedStr(
                f"Authored {len(selected)}-state npa.workflow composed from the live tool catalog."
            ),
        }
    )
    root["config"] = config
    root["initial"] = state_names[0]
    root["states"] = states
    return root


def _describe_tool_ref(tool_ref: str) -> str:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    entry = TOOL_CATALOG.get(tool_ref)
    desc = str(getattr(entry, "description", "") or "").strip() if entry else ""
    return desc or f"Run {tool_ref}."


def _config_tokens_for(selected: list[str]) -> list[str]:
    from npa.orchestration.npa_workflow.catalog import argv_for_tool

    keys: list[str] = []
    for ref in selected:
        try:
            argv = argv_for_tool(ref)
        except Exception:  # noqa: BLE001
            continue
        for token in argv:
            for key in _CONFIG_TOKEN_RE.findall(str(token)):
                if key not in keys:
                    keys.append(key)
    return keys


def author_workflow_from_goal(
    goal: str,
    *,
    tool_refs: frozenset[str] | set[str] | list[str] | None,
    bucket: str = "example-bucket",
    name: str = "",
    max_repairs: int = 2,
) -> dict[str, Any]:
    """Author a runnable npa.workflow spec from a goal using the LIVE catalog.

    Selects real toolRefs from ``tool_refs`` matching the goal, composes a
    structural spec (no hardcoded pipeline templates or tool names), then
    self-checks with ``validate_workflow_yaml_text`` + ``plan_workflow_yaml_text``
    and repairs missing ``{{config.X}}`` tokens surfaced by the planner, bounded
    by ``max_repairs``. Returns ``{ok, yaml, validation, plan, runnable, ...}``;
    ``yaml`` is only reported runnable when validate + plan both pass.
    """
    catalog = frozenset(str(t) for t in (tool_refs or []))
    if not catalog:
        return {
            "ok": False,
            "runnable": False,
            "yaml": "",
            "error": "no toolRefs available in the live catalog",
            "tool_refs": [],
        }
    explicit_refs = list(
        dict.fromkeys(re.findall(r"\bworkbench\.[A-Za-z0-9_.-]+", str(goal or "")))
    )
    unknown_explicit = [ref for ref in explicit_refs if ref not in catalog]
    if unknown_explicit:
        return {
            "ok": False,
            "runnable": False,
            "yaml": "",
            "error": "unknown explicit toolRef(s): " + ", ".join(unknown_explicit),
            "tool_refs": explicit_refs,
        }
    # Prefer explicit N-step / arrow / then counts. Only raise to matched-tool
    # count when the operator did not pin an explicit step count.
    n_steps = _desired_step_count(goal)
    explicit_step_count = bool(_STEP_COUNT_RE.search(str(goal or "")))
    semantic_stages = _author_semantic_stages(goal, n_steps)
    _, pre_matched = _select_author_tool_refs(goal, catalog, min(6, max(n_steps, 6)))
    if (not explicit_step_count) and not semantic_stages and len(pre_matched) > n_steps:
        n_steps = max(1, min(len(pre_matched), 6))
    semantic_selected = _select_semantic_tool_refs(goal, catalog, n_steps)
    if explicit_refs:
        selected = explicit_refs
        matched = explicit_refs
        n_steps = len(explicit_refs)
    elif semantic_selected:
        selected = list(semantic_selected)
        matched = list(semantic_selected)
        target_steps = max(n_steps, len(selected))
        for ref in sorted(catalog):
            if len(selected) >= target_steps:
                break
            if ref not in selected:
                selected.append(ref)
        n_steps = target_steps
    else:
        selected, matched = _select_author_tool_refs(goal, catalog, n_steps)
    if not selected:
        return {
            "ok": False,
            "runnable": False,
            "yaml": "",
            "error": "could not select any toolRef from the catalog",
            "tool_refs": [],
        }
    resolved_name = (
        str(name or "").strip() or extract_workflow_name(goal) or "authored-workflow"
    )
    resolved_name = _slugify_workflow_name(resolved_name) or "authored-workflow"
    described = _infer_stage_count_from_goal(goal) or n_steps
    dropped_note = ""
    if described > len(selected):
        dropped_note = (
            f"Requested about {described} stages but composed {len(selected)} "
            f"(catalog match / 1–6 bound); some requested stages may be missing."
        )
    config_keys = _config_tokens_for(selected)
    flow_links = _workflow_flow_links(selected)

    matched_set = set(matched)
    padded = [ref for ref in selected if ref not in matched_set]
    validation: dict[str, Any] = {"ok": False}
    plan: dict[str, Any] = {"ok": False}
    yaml_text = ""
    for _attempt in range(max(1, int(max_repairs) + 1)):
        spec = _build_authored_spec(
            selected,
            config_keys,
            bucket=bucket,
            name=resolved_name,
            matched=matched_set,
            flow_links=flow_links,
        )
        yaml_text = _render_spec_yaml(spec)
        validation = validate_workflow_yaml_text(yaml_text, tool_refs=catalog)
        if validation.get("ok"):
            plan = plan_workflow_yaml_text(
                yaml_text, run_id="authored-workflow-plan", tool_refs=catalog
            )
        else:
            plan = {
                "ok": False,
                "error": str(validation.get("error") or "validation failed"),
            }
        if validation.get("ok") and plan.get("ok"):
            break
        # Repair: add any config token the planner/validator flagged as missing.
        missing = _missing_config_tokens(validation, plan)
        new_keys = [key for key in missing if key not in config_keys]
        if not new_keys:
            break
        config_keys.extend(new_keys)

    unresolved = sorted(set(re.findall(r"<[^<>\n]+>", yaml_text)))
    runnable = bool(
        validation.get("ok") and plan.get("ok") and not unresolved and not padded
    )
    return {
        "ok": runnable,
        "runnable": runnable,
        "template": "catalog-composed",
        # Preserve a structurally valid composed chain for operator repair even
        # when unresolved values make it intentionally non-runnable. A spec
        # that itself failed validation is never returned as usable YAML.
        "yaml": yaml_text if validation.get("ok") else "",
        "validation": validation,
        "plan": plan,
        "tool_refs": selected,
        "matched_tool_refs": matched,
        "padded_tool_refs": padded,
        "states": validation.get("states") or [],
        "name": resolved_name,
        "dropped_stages_note": dropped_note,
        "desired_steps": n_steps,
        "data_flow": flow_links,
        "context_errors": (
            (
                ["unresolved configuration placeholders: " + ", ".join(unresolved)]
                if unresolved
                else []
            )
            + (
                ["catalog composition contains unmatched placeholder stages"]
                if padded
                else []
            )
        ),
    }


def _missing_config_tokens(*payloads: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    pattern = re.compile(r"config\.([a-zA-Z0-9_.-]+)")
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        error = str(payload.get("error") or "")
        if "config token" not in error and "config." not in error:
            continue
        for key in pattern.findall(error):
            if key not in keys:
                keys.append(key)
    return keys


def generate_sim2real_two_step_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real-two-step",
    user_text: str = "",
    infrastructure: dict[str, Any] | None = None,
) -> str:
    """Compatibility wrapper for two-step template generation."""
    params = extract_sim2real_params(user_text) if user_text else None
    return _render_spec_yaml(
        _build_spec(
            "two-step",
            bucket=bucket,
            name=name,
            params=params,
            infrastructure=infrastructure,
        )
    )


def generate_sim2real_staged_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real",
    user_text: str = "",
    infrastructure: dict[str, Any] | None = None,
) -> str:
    """Render the canonical compositional Sim2Real workflow with chat overlays."""
    params = extract_sim2real_params(user_text) if user_text else None
    return _render_spec_yaml(
        _build_spec(
            "sim2real-staged",
            bucket=bucket,
            name=name,
            params=params,
            infrastructure=infrastructure,
        )
    )


def generate_sim2real_loop_gate_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real-loop-gate-agent",
) -> str:
    """Compatibility wrapper for Sim2Real loop-gate template generation."""
    return _render_spec_yaml(_build_spec("loop-gate", bucket=bucket, name=name))


def generate_vlm_rl_loop_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real",
) -> str:
    """Compatibility wrapper for VLM-RL loop template generation."""
    return _render_spec_yaml(_build_spec("vlm-rl-loop", bucket=bucket, name=name))


def generate_token_factory_gate_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "tokenfactory-cosmos-gate",
) -> str:
    """Compatibility wrapper for token-factory gate template generation."""
    return _render_spec_yaml(
        _build_spec("token-factory-gate", bucket=bucket, name=name)
    )


def generate_isaac_byof_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "byof",
) -> str:
    """Compatibility wrapper for generic BYOF template generation."""
    return _render_spec_yaml(_build_spec("byof", bucket=bucket, name=name))


def generate_byof_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "byof",
) -> str:
    """Render generic BYOF workflow YAML."""
    return generate_isaac_byof_yaml(bucket=bucket, name=name)


def generate_gpu_cross_region_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "sim2real-gpu-cross-region",
) -> str:
    """Reject the retired stub template instead of advertising fake GPU work."""
    del bucket, name
    raise ValueError(
        "gpu-cross-region was a demo with stub Sim2Real components and is retired; "
        "author a generic npa.workflow with real solution toolRefs instead"
    )


def generate_rl_policy_training_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "rl-policy-training-sim-success",
) -> str:
    """Compatibility wrapper for RL policy training template generation."""
    return _render_spec_yaml(_build_spec("rl-policy-success", bucket=bucket, name=name))


def generate_data_factory_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "physical-ai-data-factory",
    user_text: str = "",
    infrastructure: dict[str, Any] | None = None,
) -> str:
    """Render Physical AI Data Factory (paidf) workflow YAML.

    ``user_text`` (optional) parameterizes the fan-out count, GPU count, and
    augmentation subject from a natural-language chat request.
    """
    params = extract_data_factory_params(user_text) if user_text else None
    return _render_spec_yaml(
        _build_spec(
            "physical-ai-data-factory",
            bucket=bucket,
            name=name,
            params=params,
            infrastructure=infrastructure,
        )
    )


def validate_workflow_yaml_text(
    yaml_text: str,
    *,
    tool_refs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Validate workflow YAML; prefers npa.orchestration when available."""
    text = str(yaml_text or "").strip()
    if not text:
        return {"ok": False, "status": "invalid", "error": "empty workflow YAML"}
    try:
        return _validate_with_npa(text)
    except ImportError:
        return _validate_lightweight(text, tool_refs=tool_refs)


def plan_workflow_yaml_text(
    yaml_text: str,
    *,
    run_id: str = "",
    assume_decision: str = "",
    tool_refs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Expand workflow YAML into a dry-run plan."""
    text = str(yaml_text or "").strip()
    if not text:
        return {"ok": False, "error": "empty workflow YAML"}
    try:
        return _plan_with_npa(text, run_id=run_id, assume_decision=assume_decision)
    except ImportError:
        return _plan_lightweight(text, run_id=run_id, tool_refs=tool_refs)


def format_workflow_chat_reply(
    yaml_text: str,
    validation: dict[str, Any],
    *,
    template: str = "two-step",
    plan: dict[str, Any] | None = None,
    runnable: bool | None = None,
    dropped_stages_note: str = "",
    warnings: list[str] | None = None,
) -> str:
    """Markdown reply for chat when a workflow YAML is generated."""
    name = str(validation.get("name") or "unnamed")
    status = str(
        validation.get("status") or ("valid" if validation.get("ok") else "invalid")
    )
    states = validation.get("states") or []
    state_label = (
        ", ".join(str(s) for s in states) if isinstance(states, list) else str(states)
    )
    resolved_plan = plan if isinstance(plan, dict) else {}
    plan_ok = bool(resolved_plan.get("ok"))
    resolved_runnable = (
        bool(runnable)
        if runnable is not None
        else bool(validation.get("ok") and plan_ok)
    )
    plan_step_count = (
        len(resolved_plan.get("steps") or [])
        if isinstance(resolved_plan.get("steps"), list)
        else 0
    )
    _desc_map = {
        "vlm-rl-loop": "VLM-RL outer/inner loop with promote/loop-back gate",
        "token-factory-gate": "Token Factory scene→augment→VLM quality gate loop",
        "loop-gate": "Sim2Real loop + decision gate pipeline",
        "byof": "Generic BYOF workflow (OSS repo → Ubuntu/Isaac base image → workload on Kubernetes)",
        "rl-policy-success": "Simulation RL policy training with success gate and publish/fail outcomes",
        "physical-ai-data-factory": (
            "Physical AI Data Factory: annotate → Cosmos Transfer augment & multiply "
            "(fan out scenarios across GPUs) → Cosmos Evaluator gate → Cosmos Curator "
            "→ FiftyOne review → Rerun visualize"
        ),
        "sim2real-staged": (
            "Canonical compositional 14-stage Sim2Real VLM-RL workflow"
        ),
        "two-step": "2-step Sim2Real pipeline",
    }
    t = str(template or "two-step").strip().lower()
    state_list = [str(s) for s in states] if isinstance(states, list) else []
    if t == "catalog-composed" or t not in _desc_map:
        if state_list:
            arrow = "→".join(state_list[:6])
            desc = f"{len(state_list)}-step pipeline: {arrow}"
        else:
            desc = "catalog-composed pipeline"
    else:
        desc = _desc_map[t]
    lines = [
        f"**Generated {API_VERSION} spec** ({desc}):",
        f"- **name**: `{name}`",
        f"- **validation**: `{status}`",
        f"- **runnable**: `{'yes' if resolved_runnable else 'no'}`",
        f"- **plan steps**: `{plan_step_count}`",
        f"- **states**: `{state_label or 'n/a'}`",
        "",
        "Edit in the **Workflow YAML** panel, then **Validate**, **Plan**, or **Submit**.",
        "- **Submit** on the agent = scheduler **plan-only** (does not execute tool steps on K8s).",
        "- Real execute: `npa workbench workflow run-spec <spec.yaml> --execute` on the operator machine.",
        "",
        "```yaml",
        yaml_text.rstrip(),
        "```",
    ]
    drop_note = str(dropped_stages_note or "").strip()
    if not drop_note and isinstance(validation, dict):
        drop_note = str(validation.get("dropped_stages_note") or "").strip()
    if drop_note:
        lines.insert(6, f"- **note**: {drop_note}")
    for warning in reversed(
        [str(item) for item in (warnings or []) if str(item).strip()]
    ):
        lines.insert(6, f"- **infrastructure**: {warning}")
    if not validation.get("ok"):
        err = str(validation.get("error") or "validation failed")
        lines.insert(6, f"- **error**: `{err}`")
    elif resolved_plan and not plan_ok:
        plan_err = str(resolved_plan.get("error") or "plan failed")
        lines.insert(6, f"- **plan_error**: `{plan_err}`")
    return "\n".join(lines)


def _npa_compatible_yaml(yaml_text: str) -> str:
    """Translate beta apiVersion to stable for orchestration loaders."""
    return re.sub(
        r"(?m)^(\s*apiVersion:\s*)npa\.workflow/v0\.0\.1-beta(\s*)$",
        r"\1npa.workflow/v0.0.1\2",
        str(yaml_text or ""),
    )


def _validate_with_npa(yaml_text: str) -> dict[str, Any]:
    from npa.orchestration.npa_workflow import NpaWorkflowError, load_spec

    path = _write_temp_yaml(_npa_compatible_yaml(yaml_text))
    try:
        spec = load_spec(path)
    except NpaWorkflowError as exc:
        return {"ok": False, "status": "invalid", "error": str(exc)}
    return {
        "ok": True,
        "status": "valid",
        "apiVersion": spec.api_version,
        "name": spec.name,
        "states": sorted(spec.states),
        "initial": spec.initial,
    }


def _plan_with_npa(
    yaml_text: str, *, run_id: str, assume_decision: str
) -> dict[str, Any]:
    from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan, load_spec

    path = _write_temp_yaml(_npa_compatible_yaml(yaml_text))
    try:
        spec = load_spec(path)
        resolved_run_id = run_id or f"{spec.name}-plan"
        plan = build_plan(spec, run_id=resolved_run_id, assume_decision=assume_decision)
    except NpaWorkflowError as exc:
        return {"ok": False, "error": str(exc)}
    payload = plan.to_dict()
    payload["ok"] = True
    payload["run_id"] = resolved_run_id
    return payload


def _validate_lightweight(
    yaml_text: str, *, tool_refs: frozenset[str] | None
) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        return {"ok": False, "status": "invalid", "error": f"invalid YAML: {exc}"}
    if not isinstance(data, dict):
        return {
            "ok": False,
            "status": "invalid",
            "error": "workflow spec must be a mapping",
        }

    api_version = str(data.get("apiVersion") or "")
    if api_version not in _SUPPORTED_API_VERSIONS:
        return {
            "ok": False,
            "status": "invalid",
            "error": (
                f"unsupported apiVersion {api_version!r} "
                f"(expected one of {sorted(_SUPPORTED_API_VERSIONS)!r})"
            ),
        }

    metadata = data.get("metadata") or {}
    name = (
        str(metadata.get("name") or "unnamed")
        if isinstance(metadata, dict)
        else "unnamed"
    )
    states_raw = data.get("states") or {}
    if not isinstance(states_raw, dict) or not states_raw:
        return {
            "ok": False,
            "status": "invalid",
            "error": "states must be a non-empty mapping",
        }

    initial = str(data.get("initial") or next(iter(states_raw)))
    if initial not in states_raw:
        return {
            "ok": False,
            "status": "invalid",
            "error": f"initial state {initial!r} not found",
        }

    catalog = tool_refs or frozenset()
    for state_name, entry in states_raw.items():
        if not isinstance(entry, dict):
            return {
                "ok": False,
                "status": "invalid",
                "error": f"state {state_name!r} must be a mapping",
            }
        tool_ref = str(entry.get("toolRef") or "").strip()
        if tool_ref and catalog and tool_ref not in catalog:
            return {
                "ok": False,
                "status": "invalid",
                "error": f"unknown toolRef {tool_ref!r}",
            }
        for edge in _state_edges(entry):
            if edge not in states_raw:
                return {
                    "ok": False,
                    "status": "invalid",
                    "error": f"state {state_name!r} references missing state {edge!r}",
                }

    return {
        "ok": True,
        "status": "valid",
        "apiVersion": api_version,
        "name": name,
        "states": sorted(str(k) for k in states_raw),
        "initial": initial,
        "mode": "lightweight",
    }


def _plan_lightweight(
    yaml_text: str, *, run_id: str, tool_refs: frozenset[str] | None
) -> dict[str, Any]:
    validation = _validate_lightweight(yaml_text, tool_refs=tool_refs)
    if not validation.get("ok"):
        return {
            "ok": False,
            "error": str(validation.get("error") or "validation failed"),
        }

    import yaml

    data = yaml.safe_load(yaml_text) or {}
    api_version = str(data.get("apiVersion") or API_VERSION)
    states_raw = data.get("states") or {}
    metadata = data.get("metadata") or {}
    name = (
        str(metadata.get("name") or "unnamed")
        if isinstance(metadata, dict)
        else "unnamed"
    )
    initial = str(data.get("initial") or next(iter(states_raw)))
    resolved_run_id = run_id or f"{name}-plan"

    steps: list[dict[str, Any]] = []
    visited: set[str] = set()
    queue: list[str] = [initial]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        entry = states_raw[current]
        steps.append(
            {
                "state": current,
                "iteration": None,
                "tool_ref": str(entry.get("toolRef") or ""),
                "resources": str(entry.get("resources") or "default"),
            }
        )
        for edge in _state_edges(entry):
            if edge not in visited:
                queue.append(edge)

    return {
        "ok": True,
        "workflow": name,
        "api_version": api_version,
        "initial": initial,
        "run_id": resolved_run_id,
        "steps": steps,
        "mode": "lightweight",
    }


def _state_edges(entry: dict[str, Any]) -> list[str]:
    edges: list[str] = []
    nxt = str(entry.get("next") or "").strip()
    if nxt:
        edges.append(nxt)

    transitions = entry.get("transitions")
    if isinstance(transitions, dict):
        for target in transitions.values():
            label = str(target or "").strip()
            if label:
                edges.append(label)
    elif isinstance(transitions, list):
        for item in transitions:
            if isinstance(item, dict):
                label = str(
                    item.get("next") or item.get("target") or item.get("goto") or ""
                ).strip()
                if label:
                    edges.append(label)

    sequence = entry.get("sequence")
    if isinstance(sequence, list):
        for item in sequence:
            if isinstance(item, dict):
                label = str(item.get("state") or item.get("next") or "").strip()
                if label:
                    edges.append(label)
            elif isinstance(item, str) and item.strip():
                edges.append(item.strip())
    return edges


def _write_temp_yaml(yaml_text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".yaml", delete=False
    )
    try:
        handle.write(yaml_text)
        handle.flush()
        return Path(handle.name)
    finally:
        handle.close()
