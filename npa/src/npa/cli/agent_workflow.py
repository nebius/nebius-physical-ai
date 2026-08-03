"""Workflow YAML generation and validation for the NPA agent UI."""

from __future__ import annotations

import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

API_VERSION_STABLE = "npa.workflow/v0.0.1"
API_VERSION_BETA = "npa.workflow/v0.0.1-beta"
API_VERSION = API_VERSION_BETA
_SUPPORTED_API_VERSIONS = frozenset({API_VERSION_STABLE, API_VERSION_BETA})

_TEMPLATES = (
    "two-step",
    "loop-gate",
    "vlm-rl-loop",
    "token-factory-gate",
    "byof",
    "gpu-cross-region",
    "rl-policy-success",
    "physical-ai-data-factory",
)


class _FoldedStr(str):
    """YAML scalar rendered with folded (>) style."""


class _LiteralStr(str):
    """YAML scalar rendered with literal (|) style."""


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
    "gpu_cross_region": "gpu-cross-region",
    "multi_region": "gpu-cross-region",
    "cross_region": "gpu-cross-region",
    "multi-region": "gpu-cross-region",
    "cross-region": "gpu-cross-region",
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
}

_INTENT_DEFAULT_TEMPLATE: dict[str, str] = {
    "create_workflow": "two-step",
    "create_vlm_rl_workflow": "vlm-rl-loop",
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
    "gpu-cross-region": (
        "multi-region",
        "cross-region",
        "two regions",
        "2 regions",
        "cross project",
        "multi project",
        "gpu workflow",
        "tenant",
    ),
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
    "two-step": ("two-step", "2-step", "simple", "minimal"),
}


def _workflow_specs() -> dict[str, dict[str, Any]]:
    return {
        "two-step": {
            "name": "sim2real-two-step",
            "description": (
                "Two-step Sim2Real pipeline: Cosmos Transfer augment, then raw "
                "environment generation."
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
                    # `sim2real_envgen` takes the RUN ROOT and derives envs/raw,
                    # envs/train, envs/heldout and envs/manifest beneath it.
                    "envgen_root_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
                    "raw_envs_uri": "s3://{{config.bucket}}/{{config.prefix}}/envs/raw/",
                    "shard_index": "0",
                    "shard_count": "1",
                    "train_fraction": "0.8",
                    "envgen_seed": "42",
                    "augmented_frames_uri": "",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict({"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment of LeRobot trigger data.",
                            "toolRef": "workbench.cosmos2.transfer",
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
                                        "uri": "{{config.augment_uri}}manifest.json",
                                        "schema": "npa.sim2real.augment.v1",
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
                                        "uri": "{{config.augment_uri}}manifest.json",
                                        "schema": "npa.sim2real.augment.v1",
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
                    "gpu": OrderedDict({"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}),
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
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/scores/",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict({"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment stage.",
                            "toolRef": "workbench.cosmos2.transfer",
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
                                        "uri": "{{config.augment_uri}}manifest.json",
                                        "schema": "npa.sim2real.augment.v1",
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
                                {"max": "{{config.refinement_iterations}}", "until": "promote_checkpoint"}
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
                                OrderedDict({"uri": "{{config.rollouts_uri}}", "schema": "npa.workbench.rollout_set.v1"})
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
                                OrderedDict({"when": "promote_checkpoint", "goto": "publish"}),
                                OrderedDict({"when": "loop_back", "goto": "augment"}),
                            ],
                        }
                    ),
                    "publish": OrderedDict(
                        {
                            "description": "Finalize report once promoted.",
                            "needs": ["refine"],
                            "toolRef": "workbench.sim2real.finalize",
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
                    # `sim2real_envgen` takes the RUN ROOT and derives envs/raw,
                    # envs/train, envs/heldout and envs/manifest beneath it.
                    "envgen_root_uri": "s3://{{config.bucket}}/{{config.prefix}}/",
                    "raw_envs_uri": "s3://{{config.bucket}}/{{config.prefix}}/envs/raw/",
                    "shard_index": "0",
                    "shard_count": "1",
                    "train_fraction": "0.8",
                    "envgen_seed": "42",
                    "augmented_frames_uri": "",
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/actions/train/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/vlm_eval/train/",
                    "heldout_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/eval/heldout/report.json",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/outer_loop/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/sim2real-report.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict({"cloud": "kubernetes", "accelerators": "RTXPRO6000:1"}),
                    "cpu": OrderedDict({"cloud": "kubernetes", "cpus": 8}),
                }
            ),
            "initial": "augment",
            "states": OrderedDict(
                {
                    "augment": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment of LeRobot trigger data.",
                            "toolRef": "workbench.cosmos2.transfer",
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
                                        "uri": "{{config.augment_uri}}manifest.json",
                                        "schema": "npa.sim2real.augment.v1",
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
                            "loop": OrderedDict({"max": "{{config.outer_iterations}}", "until": "promote_checkpoint"}),
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
                            "toolRef": "workbench.sim2real.policy_rollouts",
                            "resources": "gpu",
                            "outputs": [
                                OrderedDict({"uri": "{{config.rollouts_uri}}", "schema": "npa.sim2real.action_rollout.v1"})
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
                            "toolRef": "workbench.sim2real.heldout_eval",
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
                                OrderedDict({"when": "promote_checkpoint", "goto": "finalize"}),
                                OrderedDict({"when": "loop_back", "goto": "outer"}),
                            ],
                        }
                    ),
                    "finalize": OrderedDict(
                        {
                            "description": "Report upload and visualization artifacts.",
                            "needs": ["outer"],
                            "toolRef": "workbench.sim2real.finalize",
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
                    "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/augment/",
                    "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/scores/",
                    "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json",
                    "finalize_report_uri": "s3://{{config.bucket}}/{{config.prefix}}/reports/final.json",
                }
            ),
            "resources": OrderedDict(
                {
                    "gpu": OrderedDict(
                        {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1", "cpus": 16, "memory": "80Gi"}
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
                                OrderedDict({"uri": "{{config.scene_uri}}", "schema": "npa.token_factory.scene.v1"})
                            ],
                            "outputs": [
                                OrderedDict({"uri": "{{config.plan_uri}}plan.json", "schema": "npa.token_factory.plan.v1"})
                            ],
                            "next": "augment-scene",
                        }
                    ),
                    "augment-scene": OrderedDict(
                        {
                            "description": "Cosmos Transfer augment driven by the scene plan.",
                            "needs": ["reason-scene"],
                            "toolRef": "workbench.cosmos2.transfer",
                            "resources": "gpu",
                            "inputs": [
                                OrderedDict({"uri": "{{config.trigger_uri}}", "schema": "npa.token_factory.scene.v1"})
                            ],
                            "outputs": [
                                OrderedDict(
                                    {
                                        "uri": "{{config.augment_uri}}manifest.json",
                                        "schema": "npa.sim2real.augment.v1",
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
                                {"max": "{{config.refinement_iterations}}", "until": "promote_checkpoint"}
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
                                OrderedDict({"uri": "{{config.rollouts_uri}}", "schema": "npa.workbench.rollout_set.v1"})
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
                                OrderedDict({"when": "promote_checkpoint", "goto": "publish"}),
                                OrderedDict({"when": "loop_back", "goto": "augment-scene"}),
                            ],
                        }
                    ),
                    "publish": OrderedDict(
                        {
                            "description": "Write final report when the batch is promoted.",
                            "needs": ["refine"],
                            "toolRef": "workbench.sim2real.finalize",
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
                            "toolRef": "workbench.sim2real.policy_rollouts",
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
                            "toolRef": "workbench.sim2real.heldout_eval",
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
                            "toolRef": "workbench.sim2real.finalize",
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
                                OrderedDict({"when": "promote_checkpoint", "goto": "publish-policy"}),
                                OrderedDict({"when": "loop_back", "goto": "training-not-success"}),
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
                # What to augment: a free-form hint surfaced as the augment prompt /
                # input-conditioning subject (the run's real input clips are the base).
                "augment_subject": "the input robot clips",
                # Fan-out: N sampled appearance combos -> N scenario variants.
                "n_augmentations": "4",
                # Concurrency of the multiply fan-out (one variant per GPU); align
                # with the gpu resource accelerator count for full utilization.
                "variant_parallelism": "4",
                "refinement_iterations": "2",
                "grade_threshold": "0.5",
                "default_decision": "promote_checkpoint",
                "caption_model": "Qwen/Qwen2.5-VL-72B-Instruct",
                "vlm_backend": "api",
                "max_images": "8",
                "max_tokens": "512",
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
                "rollouts_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "scores_uri": "s3://{{config.bucket}}/{{config.prefix}}/grade/",
                "decision_uri": "s3://{{config.bucket}}/{{config.prefix}}/grade/decision.json",
                "augmented_frames_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
                "labeled_augmented_uri": "s3://{{config.bucket}}/{{config.prefix}}/labeled_augmented/",
                "lance_uri": "s3://{{config.bucket}}/{{config.prefix}}/cosmos_augmented/",
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
                        "accelerators": "RTXPRO6000:4",
                        "cpus": 16,
                        "memory": "80Gi",
                    }
                ),
                "cpu": OrderedDict({"cloud": "kubernetes", "cpus": 4, "memory": "16Gi"}),
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
                                "shell": (
                                    "python3 -c \"from npa.workflows.data_factory_stages import "
                                    "generate_configs; generate_configs('{{config.configs_uri}}', "
                                    "'{{config.n_augmentations}}', '{{run.id}}')\""
                                )
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
                                    "uri": "{{config.augment_uri}}manifest.json",
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
                            "attribute-verify (VLM) -> quality-gate. Loops back to RE-AUGMENT on "
                            "failure, up to refinement_iterations, and breaks on promote."
                        ),
                        "needs": ["annotate-original"],
                        "loop": OrderedDict(
                            {
                                "max": "{{config.refinement_iterations}}",
                                "until": "promote_checkpoint",
                            }
                        ),
                        "sequence": ["augment", "attribute-verify", "quality-gate"],
                        "next": "annotate-augmented",
                    }
                ),
                "attribute-verify": OrderedDict(
                    {
                        "description": (
                            "VLM-based attribute verification of the augmented clips "
                            "(Token Factory, --backend api)."
                        ),
                        "toolRef": "workbench.vlm_eval.run",
                        "resources": "cpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.rollouts_uri}}",
                                    "schema": "npa.sim2real.augment.v1",
                                }
                            )
                        ],
                        "outputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.scores_uri}}vlm_eval_stub.json",
                                    "schema": "npa.workbench.vlm_eval.report.v1",
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
                        "needs": ["attribute-verify"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "shell": (
                                    "python3 -c \"from npa.workflows.data_factory_stages import "
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
                "annotate-augmented": OrderedDict(
                    {
                        "description": (
                            "Stage 3 - Pseudo-Label Augmented. Re-caption the promoted augmented "
                            "clips with the same hosted VLM so the amplified set ships labeled."
                        ),
                        "needs": ["grade"],
                        "resources": "cpu",
                        "run": OrderedDict(
                            {
                                "shell": (
                                    "npa workbench token-factory caption "
                                    "--input-path \"{{config.augmented_frames_uri}}\" "
                                    "--output-path \"{{config.labeled_augmented_uri}}\" "
                                    "--model \"{{config.caption_model}}\" "
                                    "--max-images \"{{config.max_images}}\" "
                                    "--max-tokens \"{{config.max_tokens}}\" --output json"
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
                        "needs": ["annotate-augmented"],
                        "resources": "cpu",
                        "inputs": [
                            OrderedDict(
                                {
                                    "uri": "{{config.lance_uri}}",
                                    "schema": "npa.cosmos2.transfer.v1",
                                }
                            )
                        ],
                        "run": OrderedDict(
                            {
                                "shell": (
                                    "python3 -c \"from npa.workflows.data_factory_stages import "
                                    "curate; curate('{{config.augment_uri}}', "
                                    "'{{config.curation_report_uri}}')\""
                                )
                            }
                        ),
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
                        "run": OrderedDict(
                            {
                                "shell": (
                                    "python3 -c \"from npa.workflows.data_factory_viz import "
                                    "build_run_rrd; print(build_run_rrd('{{config.run_root_uri}}', "
                                    "'{{config.rrd_uri}}'))\""
                                )
                            }
                        ),
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
                                    "python3 -c \"from npa.workflows.data_factory_stages import "
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
    default_template = _INTENT_DEFAULT_TEMPLATE.get(str(intent or "").strip(), "two-step")
    scores[default_template] += 3
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
        for token in ("paidf", "data factory", "physical ai data factory", "video data augmentation")
    )
    if data_factory_explicit:
        scores["physical-ai-data-factory"] += 10
    if ("augment" in text or "cosmos transfer" in text) and any(
        token in text for token in ("fan out", "fan-out", "fanout", "multiply", "scenario", "variant", "amplify")
    ):
        scores["physical-ai-data-factory"] += 6
    if "gpu" in text and ("region" in text or "project" in text):
        scores["gpu-cross-region"] += 5
    if "rl" in text and ("policy" in text or "training" in text or "isaac" in text):
        if not byof_explicit:
            scores["rl-policy-success"] += 5
    if capabilities and not byof_explicit:
        capabilities_text = " ".join(f"{k}:{v}" for k, v in sorted(capabilities.items())).lower()
        if "token" in capabilities_text:
            scores["token-factory-gate"] += 2
        if "vlm" in capabilities_text and "rl" in capabilities_text:
            scores["vlm-rl-loop"] += 2
        if any(k in capabilities_text for k in ("isaac", "policy", "training", "rl")):
            scores["rl-policy-success"] += 2
        if any(k in capabilities_text for k in ("loop", "gate", "transition")):
            scores["loop-gate"] += 1
        if any(k in capabilities_text for k in ("tenant", "project", "region")):
            scores["gpu-cross-region"] += 2
    selected = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
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

    if count is not None and count > 0:
        params["n_augmentations"] = min(count, 64)
    if gpus is not None and gpus > 0:
        params["gpu_count"] = min(gpus, 8)

    subject_match = _DATA_FACTORY_SUBJECT_RE.search(text)
    if subject_match:
        subject = subject_match.group(1).strip(" .,\"'`")
        # Drop trailing fan-out/scenario phrasing that leaked into the subject.
        subject = re.sub(
            r"\s+(?:and\s+)?(?:fan[\s-]?out|multiply|amplify).*$", "", subject, flags=re.IGNORECASE
        ).strip(" .,\"'`")
        if subject and len(subject) <= 120:
            params["augment_subject"] = subject
    return params


_DATA_FACTORY_DEFAULT_GPUS = 4


def _data_factory_gpu_count(config: OrderedDict[str, Any], params: dict[str, Any]) -> int:
    """Resolve the GPU accelerator count for a paidf run (>=1, <=8)."""
    gpus = params.get("gpu_count")
    if gpus:
        return max(1, min(int(gpus), 8))
    return _DATA_FACTORY_DEFAULT_GPUS


def _apply_data_factory_params(config: OrderedDict[str, Any], params: dict[str, Any]) -> None:
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
    resolved_variants = int(str(config.get("n_augmentations") or "1"))
    gpu_count = _data_factory_gpu_count(config, params)
    config["variant_parallelism"] = str(max(1, min(resolved_variants, gpu_count)))


def _build_spec(
    template: str,
    *,
    bucket: str,
    name: str | None,
    params: dict[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    catalog = _workflow_specs()
    normalized = _normalize_template(template)
    spec = catalog[normalized]
    metadata_name = str(name or spec["name"])
    description = _FoldedStr(str(spec["description"]))
    config = OrderedDict({"bucket": str(bucket)})
    config.update(spec["config_runtime"])
    config.update(spec["config_uri"])
    if normalized == "physical-ai-data-factory" and params:
        _apply_data_factory_params(config, params)
    states = OrderedDict()
    for state_name, state_spec in spec["states"].items():
        state_payload: OrderedDict[str, Any] = OrderedDict()
        for key, value in state_spec.items():
            if key == "description":
                state_payload[key] = _FoldedStr(str(value))
            elif key == "run" and isinstance(value, dict):
                run_payload: OrderedDict[str, Any] = OrderedDict()
                for run_key, run_value in value.items():
                    if run_key == "shell" and isinstance(run_value, str) and "\n" in run_value:
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
    resources = spec["resources"]
    if normalized == "physical-ai-data-factory" and params:
        gpu_count = _data_factory_gpu_count(config, params)
        gpu_res = resources.get("gpu")
        if isinstance(gpu_res, dict) and gpu_count != _DATA_FACTORY_DEFAULT_GPUS:
            gpu_res["accelerators"] = f"RTXPRO6000:{gpu_count}"
    root["resources"] = resources
    root["initial"] = spec["initial"]
    root["states"] = states
    return root


def _insert_config_spacing(yaml_text: str) -> str:
    lines = yaml_text.splitlines()
    first_uri_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s{2}[A-Za-z0-9_-]*_uri:\s", line):
            first_uri_idx = idx
            break
    if first_uri_idx is not None and first_uri_idx > 0 and lines[first_uri_idx - 1].strip():
        lines.insert(first_uri_idx, "")
    return "\n".join(lines).rstrip() + "\n"


def _render_spec_yaml(spec: OrderedDict[str, Any]) -> str:
    rendered = yaml.dump(_to_builtin(spec), Dumper=_WorkflowDumper, sort_keys=False, width=96)
    return _insert_config_spacing(rendered)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, OrderedDict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


def generate_workflow_yaml(template: str = "two-step", *, bucket: str = "example-bucket") -> str:
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
) -> dict[str, Any]:
    """Draft workflow YAML by selecting a template from intent/capabilities."""
    if template:
        selected_template = _normalize_template(template)
        selection = {"template": selected_template, "scores": {selected_template: 1}}
    else:
        selection = choose_workflow_template(user_text=user_text, intent=intent, capabilities=capabilities)
        selected_template = str(selection["template"])
    params = (
        extract_data_factory_params(user_text)
        if selected_template == "physical-ai-data-factory"
        else None
    )
    spec = _build_spec(selected_template, bucket=bucket, name=name or None, params=params)
    yaml_text = _render_spec_yaml(spec)
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=tool_refs)
    plan: dict[str, Any]
    if validation.get("ok"):
        plan = plan_workflow_yaml_text(
            yaml_text,
            run_id=f"draft-{selected_template}",
            tool_refs=tool_refs,
        )
    else:
        plan = {"ok": False, "error": str(validation.get("error") or "validation failed")}
    runnable = bool(validation.get("ok") and plan.get("ok"))
    return {
        "template": selected_template,
        "selection": selection,
        "yaml": yaml_text,
        "validation": validation,
        "plan": plan,
        "runnable": runnable,
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
        "write", "me", "a", "an", "the", "step", "steps", "npa", "yaml", "spec",
        "workflow", "pipeline", "that", "uses", "use", "using", "with", "and",
        "for", "to", "of", "create", "generate", "build", "make", "draft",
        "compose", "please", "give", "show", "new", "simple", "minimal", "example",
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


def _author_placeholder_for(key: str) -> str:
    low = key.lower()
    if low == "bucket":
        return "example-bucket"
    if low == "prefix":
        return "npa-workflow/{{run.id}}"
    if low.endswith("_uri") or low.endswith("_path") or "uri" in low or low == "output_root":
        return "s3://{{config.bucket}}/{{config.prefix}}/" + key + "/"
    if any(tok in low for tok in ("count", "iterations", "num", "size", "timeout", "interval", "episodes", "steps")):
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
) -> OrderedDict[str, Any]:
    matched_set = set(matched or ())

    config: OrderedDict[str, Any] = OrderedDict()
    config["bucket"] = str(bucket)
    config["prefix"] = "npa-workflow/{{run.id}}"
    for key in config_keys:
        if key not in config:
            config[key] = _author_placeholder_for(key)

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
        if matched_set and ref not in matched_set:
            entry_desc = f"[placeholder — no goal match; replace with the intended tool] {entry_desc}"
        state: OrderedDict[str, Any] = OrderedDict()
        state["description"] = _FoldedStr(entry_desc)
        if idx > 0:
            state["needs"] = [state_names[idx - 1]]
        state["toolRef"] = ref
        if idx < len(selected) - 1:
            state["next"] = state_names[idx + 1]
        else:
            state["terminal"] = True
        states[state_name] = state

    root: OrderedDict[str, Any] = OrderedDict()
    root["apiVersion"] = API_VERSION
    root["kind"] = "Workflow"
    root["metadata"] = OrderedDict(
        {"name": str(name), "description": _FoldedStr(f"Authored {len(selected)}-state npa.workflow composed from the live tool catalog.")}
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
        return {"ok": False, "runnable": False, "yaml": "", "error": "no toolRefs available in the live catalog", "tool_refs": []}
    # Prefer explicit N-step / arrow / then counts. Only raise to matched-tool
    # count when the operator did not pin an explicit step count.
    n_steps = _desired_step_count(goal)
    explicit_step_count = bool(_STEP_COUNT_RE.search(str(goal or "")))
    _, pre_matched = _select_author_tool_refs(goal, catalog, min(6, max(n_steps, 6)))
    if (not explicit_step_count) and len(pre_matched) > n_steps:
        n_steps = max(1, min(len(pre_matched), 6))
    selected, matched = _select_author_tool_refs(goal, catalog, n_steps)
    if not selected:
        return {"ok": False, "runnable": False, "yaml": "", "error": "could not select any toolRef from the catalog", "tool_refs": []}
    resolved_name = (
        str(name or "").strip()
        or extract_workflow_name(goal)
        or "authored-workflow"
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

    matched_set = set(matched)
    padded = [ref for ref in selected if ref not in matched_set]
    validation: dict[str, Any] = {"ok": False}
    plan: dict[str, Any] = {"ok": False}
    yaml_text = ""
    for _attempt in range(max(1, int(max_repairs) + 1)):
        spec = _build_authored_spec(
            selected, config_keys, bucket=bucket, name=resolved_name, matched=matched_set
        )
        yaml_text = _render_spec_yaml(spec)
        validation = validate_workflow_yaml_text(yaml_text, tool_refs=catalog)
        if validation.get("ok"):
            plan = plan_workflow_yaml_text(yaml_text, run_id="authored-workflow-plan", tool_refs=catalog)
        else:
            plan = {"ok": False, "error": str(validation.get("error") or "validation failed")}
        if validation.get("ok") and plan.get("ok"):
            break
        # Repair: add any config token the planner/validator flagged as missing.
        missing = _missing_config_tokens(validation, plan)
        new_keys = [key for key in missing if key not in config_keys]
        if not new_keys:
            break
        config_keys.extend(new_keys)

    runnable = bool(validation.get("ok") and plan.get("ok"))
    return {
        "ok": runnable,
        "runnable": runnable,
        "template": "catalog-composed",
        "yaml": yaml_text,
        "validation": validation,
        "plan": plan,
        "tool_refs": selected,
        "matched_tool_refs": matched,
        "padded_tool_refs": padded,
        "states": validation.get("states") or [],
        "name": resolved_name,
        "dropped_stages_note": dropped_note,
        "desired_steps": n_steps,
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
) -> str:
    """Compatibility wrapper for two-step template generation."""
    return _render_spec_yaml(_build_spec("two-step", bucket=bucket, name=name))


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
    name: str = "sim2real-vlm-rl",
) -> str:
    """Compatibility wrapper for VLM-RL loop template generation."""
    return _render_spec_yaml(_build_spec("vlm-rl-loop", bucket=bucket, name=name))


def generate_token_factory_gate_yaml(
    *,
    bucket: str = "example-bucket",
    name: str = "tokenfactory-cosmos-gate",
) -> str:
    """Compatibility wrapper for token-factory gate template generation."""
    return _render_spec_yaml(_build_spec("token-factory-gate", bucket=bucket, name=name))


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
    """Compatibility wrapper for tenant-scoped cross-region GPU template generation."""
    return _render_spec_yaml(_build_spec("gpu-cross-region", bucket=bucket, name=name))


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
) -> str:
    """Render Physical AI Data Factory (paidf) workflow YAML.

    ``user_text`` (optional) parameterizes the fan-out count, GPU count, and
    augmentation subject from a natural-language chat request.
    """
    params = extract_data_factory_params(user_text) if user_text else None
    return _render_spec_yaml(
        _build_spec("physical-ai-data-factory", bucket=bucket, name=name, params=params)
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
) -> str:
    """Markdown reply for chat when a workflow YAML is generated."""
    name = str(validation.get("name") or "unnamed")
    status = str(validation.get("status") or ("valid" if validation.get("ok") else "invalid"))
    states = validation.get("states") or []
    state_label = ", ".join(str(s) for s in states) if isinstance(states, list) else str(states)
    resolved_plan = plan if isinstance(plan, dict) else {}
    plan_ok = bool(resolved_plan.get("ok"))
    resolved_runnable = bool(runnable) if runnable is not None else bool(validation.get("ok") and plan_ok)
    plan_step_count = len(resolved_plan.get("steps") or []) if isinstance(resolved_plan.get("steps"), list) else 0
    _desc_map = {
        "vlm-rl-loop": "VLM-RL outer/inner loop with promote/loop-back gate",
        "token-factory-gate": "Token Factory scene→augment→VLM quality gate loop",
        "loop-gate": "Sim2Real loop + decision gate pipeline",
        "byof": "Generic BYOF workflow (OSS repo → Ubuntu/Isaac base image → workload on Kubernetes)",
        "gpu-cross-region": "Tenant-scoped GPU workflow across two project/region targets",
        "rl-policy-success": "Simulation RL policy training with success gate and publish/fail outcomes",
        "physical-ai-data-factory": (
            "Physical AI Data Factory: annotate → Cosmos Transfer augment & multiply "
            "(fan out scenarios across GPUs) → VLM grade loop → curate → Rerun visualize"
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


def _plan_with_npa(yaml_text: str, *, run_id: str, assume_decision: str) -> dict[str, Any]:
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


def _validate_lightweight(yaml_text: str, *, tool_refs: frozenset[str] | None) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        return {"ok": False, "status": "invalid", "error": f"invalid YAML: {exc}"}
    if not isinstance(data, dict):
        return {"ok": False, "status": "invalid", "error": "workflow spec must be a mapping"}

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
    name = str(metadata.get("name") or "unnamed") if isinstance(metadata, dict) else "unnamed"
    states_raw = data.get("states") or {}
    if not isinstance(states_raw, dict) or not states_raw:
        return {"ok": False, "status": "invalid", "error": "states must be a non-empty mapping"}

    initial = str(data.get("initial") or next(iter(states_raw)))
    if initial not in states_raw:
        return {"ok": False, "status": "invalid", "error": f"initial state {initial!r} not found"}

    catalog = tool_refs or frozenset()
    for state_name, entry in states_raw.items():
        if not isinstance(entry, dict):
            return {"ok": False, "status": "invalid", "error": f"state {state_name!r} must be a mapping"}
        tool_ref = str(entry.get("toolRef") or "").strip()
        if tool_ref and catalog and tool_ref not in catalog:
            return {"ok": False, "status": "invalid", "error": f"unknown toolRef {tool_ref!r}"}
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


def _plan_lightweight(yaml_text: str, *, run_id: str, tool_refs: frozenset[str] | None) -> dict[str, Any]:
    validation = _validate_lightweight(yaml_text, tool_refs=tool_refs)
    if not validation.get("ok"):
        return {"ok": False, "error": str(validation.get("error") or "validation failed")}

    import yaml

    data = yaml.safe_load(yaml_text) or {}
    api_version = str(data.get("apiVersion") or API_VERSION)
    states_raw = data.get("states") or {}
    metadata = data.get("metadata") or {}
    name = str(metadata.get("name") or "unnamed") if isinstance(metadata, dict) else "unnamed"
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
                label = str(item.get("next") or item.get("target") or item.get("goto") or "").strip()
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
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False)
    try:
        handle.write(yaml_text)
        handle.flush()
        return Path(handle.name)
    finally:
        handle.close()
