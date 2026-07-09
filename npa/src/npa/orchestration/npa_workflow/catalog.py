"""Catalog of workbench tools referenced by ``toolRef`` in NPA workflow specs."""

from __future__ import annotations

from dataclasses import dataclass

from npa.orchestration.npa_workflow.errors import NpaWorkflowError


@dataclass(frozen=True)
class ToolEntry:
    name: str
    argv_template: list[str]
    description: str = ""
    stub: bool = False


_BYOF_REPO_ARGV = [
    "npa",
    "workbench",
    "byof",
    "run",
    "--repo-url",
    "{{config.repo_url}}",
    "--repo-ref",
    "{{config.repo_ref}}",
    "--base-profile",
    "{{config.base_profile}}",
    "--base-image",
    "{{config.base_image}}",
    "--workload",
    "{{config.workload}}",
    "--yaml",
    "{{config.resource_profile_yaml}}",
    "--task",
    "{{config.task}}",
    "--iterations",
    "{{config.iterations}}",
    "--num-envs",
    "{{config.num_envs}}",
    "--num-demos",
    "{{config.num_demos}}",
    "--run-id",
    "{{run.id}}",
    "--output-root",
    "{{config.output_root}}",
    "--wait-timeout",
    "{{config.wait_timeout}}",
    "--poll-interval",
    "{{config.poll_interval}}",
    "--cleanup",
]

TOOL_CATALOG: dict[str, ToolEntry] = {
    "infra.soperator.deploy": ToolEntry(
        name="infra.soperator.deploy",
        description=(
            "Deploy a Nebius soperator (Slurm-on-Kubernetes) cluster from an "
            "npa.soperator/v0.0.1 spec (multiple worker presets + optional docker cache)."
        ),
        argv_template=[
            "npa",
            "soperator",
            "deploy",
            "--spec",
            "{{config.soperator_spec}}",
            "--output",
            "json",
        ],
    ),
    "workbench.vlm_eval.run": ToolEntry(
        name="workbench.vlm_eval.run",
        description="Score rollout directories with the VLM eval workbench tool.",
        argv_template=[
            "npa",
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "{{config.rollouts_uri}}",
            "--output-path",
            "{{config.scores_uri}}",
            "--backend",
            "{{config.vlm_backend}}",
        ],
    ),
    "workbench.token_factory.reason": ToolEntry(
        name="workbench.token_factory.reason",
        description="Run Cosmos reasoner over scene inputs.",
        argv_template=[
            "npa",
            "workbench",
            "token-factory",
            "reason",
            "--input-path",
            "{{config.scene_uri}}",
            "--output-path",
            "{{config.plan_uri}}",
        ],
    ),
    "workbench.cosmos2.transfer": ToolEntry(
        name="workbench.cosmos2.transfer",
        description="Cosmos Transfer augment stage.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos2",
            "transfer",
            "--input-path",
            "{{config.trigger_uri}}",
            "--output-path",
            "{{config.augment_uri}}",
        ],
    ),
    "workbench.sim2real_envgen.raw_shard": ToolEntry(
        name="workbench.sim2real_envgen.raw_shard",
        description="Generate raw simulation env shard.",
        argv_template=[
            "python",
            "-m",
            "npa.workflows.sim2real_envgen",
            "raw-shard",
            "--output-uri",
            "{{config.raw_envs_uri}}",
            "--env-count",
            "{{config.env_count}}",
        ],
    ),
    "workbench.sim2real.policy_rollouts": ToolEntry(
        name="workbench.sim2real.policy_rollouts",
        description="Policy rollouts on train envs (workflow stub until sim2real step wiring).",
        argv_template=["echo", "policy rollouts -> {{config.rollouts_uri}}"],
        stub=True,
    ),
    "workbench.sim2real.heldout_eval": ToolEntry(
        name="workbench.sim2real.heldout_eval",
        description="Held-out simulation eval (workflow stub).",
        argv_template=["echo", "heldout eval -> {{config.heldout_report_uri}}"],
        stub=True,
    ),
    "workbench.sim2real.write_decision": ToolEntry(
        name="workbench.sim2real.write_decision",
        description="Write threshold decision artifact for dynamic transitions (demo stub).",
        argv_template=[
            "python",
            "-c",
            (
                "from npa.orchestration.npa_workflow.decisions import write_decision; "
                "write_decision('{{config.decision_uri}}', '{{config.default_decision}}')"
            ),
        ],
    ),
    "workbench.sim2real.finalize": ToolEntry(
        name="workbench.sim2real.finalize",
        description="Finalize run artifacts (workflow stub).",
        argv_template=["echo", "finalize run {{run.id}} -> {{config.finalize_report_uri}}"],
        stub=True,
    ),
    "workbench.byof.repo": ToolEntry(
        name="workbench.byof.repo",
        description="Build/push a BYOF OSS repo image via npa workbench byof and launch a workload.",
        argv_template=_BYOF_REPO_ARGV,
    ),
    "workbench.isaac_lab.byof_repo": ToolEntry(
        name="workbench.isaac_lab.byof_repo",
        description="Compatibility alias for workbench.byof.repo.",
        argv_template=_BYOF_REPO_ARGV,
    ),
    "workbench.data_transform.rollout_contract": ToolEntry(
        name="workbench.data_transform.rollout_contract",
        description="Validate + adapt rollout contract payloads to canonical v1.",
        argv_template=[
            "python3",
            "-c",
            (
                "import json;from pathlib import Path;"
                "source='npa.sim2real.action_rollout.v1';"
                "target='npa.sim2real.rollout_manifest.v1';"
                "payload={'tenant_id':'{{config.tenant_id}}','source_project':'{{config.project_primary}}',"
                "'target_project':'{{config.project_secondary}}','source_region':'{{config.region_primary}}',"
                "'target_region':'{{config.region_secondary}}','source_uri':'{{config.rollouts_uri}}manifest.json',"
                "'target_uri':'{{config.normalized_rollouts_uri}}manifest.json','source_schema':source,"
                "'target_schema':target,'contract_version':'v1','adapter_version':'v1',"
                "'status':'ok'};"
                "required=('tenant_id','source_project','target_project','source_region','target_region',"
                "'source_uri','target_uri','source_schema','target_schema','contract_version','adapter_version','status');"
                "missing=[k for k in required if not payload.get(k)];"
                "assert not missing, f'missing required fields: {missing}';"
                "Path('{{config.improvement_local_path}}').write_text(json.dumps(payload, indent=2));"
                "print('normalized manifest ready')"
            ),
        ],
    ),
    "workbench.data_transform.improvement_summary": ToolEntry(
        name="workbench.data_transform.improvement_summary",
        description="Generate contract-validated cross-region improvement summary payload.",
        argv_template=[
            "python3",
            "-c",
            (
                "import json;from pathlib import Path;"
                "summary={'tenant_id':'{{config.tenant_id}}','projects':['{{config.project_primary}}',"
                "'{{config.project_secondary}}'],'regions':['{{config.region_primary}}','{{config.region_secondary}}'],"
                "'metrics':{'improvement_delta':0.12},'result':'improved',"
                "'contract_version':'v1'};"
                "assert isinstance(summary['projects'], list) and len(summary['projects']) == 2;"
                "assert isinstance(summary['regions'], list) and len(summary['regions']) == 2;"
                "assert isinstance(summary['metrics'].get('improvement_delta'), (int, float));"
                "assert summary['contract_version'] == 'v1', 'unsupported improvement contract version';"
                "Path('{{config.improvement_local_path}}').write_text(json.dumps(summary, indent=2));"
                "print(json.dumps(summary))"
            ),
        ],
    ),
    "workbench.rl.policy_train": ToolEntry(
        name="workbench.rl.policy_train",
        description="Train simulator RL policy checkpoint with workbench RL backend.",
        argv_template=[
            "npa",
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "{{config.task_name}}",
            "--steps",
            "{{config.train_steps}}",
            "--learning-rate",
            "{{config.learning_rate}}",
            "--batch-size",
            "{{config.batch_size}}",
            "--input-path",
            "{{config.train_dataset_uri}}",
            "--output-path",
            "{{config.checkpoint_uri}}",
        ],
    ),
    "workbench.rl.evaluate_policy": ToolEntry(
        name="workbench.rl.evaluate_policy",
        description="Evaluate RL policy checkpoint on held-out simulation episodes.",
        argv_template=[
            "npa",
            "workbench",
            "isaac-lab",
            "eval",
            "--task",
            "{{config.task_name}}",
            "--checkpoint",
            "{{config.checkpoint_uri}}",
            "--episodes",
            "{{config.eval_episodes}}",
            "--output-path",
            "{{config.eval_report_uri}}",
        ],
    ),
    "workbench.rl.write_success_decision": ToolEntry(
        name="workbench.rl.write_success_decision",
        description="Write promote/loop decision from configured RL success threshold.",
        argv_template=[
            "python3",
            "-c",
            (
                "from npa.orchestration.npa_workflow.decisions import write_decision;"
                "threshold=float('{{config.success_threshold}}');"
                "decision='promote_checkpoint' if threshold <= 0.9 else 'loop_back';"
                "write_decision('{{config.decision_uri}}', decision)"
            ),
        ],
    ),
    "workbench.rl.publish_policy": ToolEntry(
        name="workbench.rl.publish_policy",
        description="Publish promoted RL checkpoint to release artifact prefix.",
        argv_template=[
            "python3",
            "-c",
            (
                "import json;from pathlib import Path;"
                "payload={'checkpoint_uri':'{{config.checkpoint_uri}}','release_uri':'{{config.release_uri}}',"
                "'decision_uri':'{{config.decision_uri}}','status':'promoted'};"
                "Path('/tmp/npa-rl-release.json').write_text(json.dumps(payload));"
                "print(json.dumps(payload))"
            ),
        ],
    ),
    "workbench.rl.report_failure": ToolEntry(
        name="workbench.rl.report_failure",
        description="Write terminal RL failure report when threshold is not met.",
        argv_template=[
            "python3",
            "-c",
            (
                "import json;from pathlib import Path;"
                "payload={'eval_report_uri':'{{config.eval_report_uri}}','decision_uri':'{{config.decision_uri}}',"
                "'status':'not_promoted'};"
                "Path('/tmp/npa-rl-failure.json').write_text(json.dumps(payload));"
                "print(json.dumps(payload))"
            ),
        ],
    ),
    "workbench.lancedb.import_bdd100k": ToolEntry(
        name="workbench.lancedb.import_bdd100k",
        description="Import BDD100K rows into LanceDB through the workbench service.",
        argv_template=[
            "npa",
            "workbench",
            "lancedb",
            "import-bdd100k",
            "--source",
            "{{config.source_uri}}",
            "--table",
            "{{config.lance_table}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--limit",
            "{{config.bdd100k_limit}}",
            "--split",
            "train",
            "--split",
            "val",
            "--service",
            "--endpoint",
            "{{config.lancedb_endpoint}}",
        ],
    ),
    "workbench.lancedb.backfill_cpu_bundle": ToolEntry(
        name="workbench.lancedb.backfill_cpu_bundle",
        description="Backfill all CPU UDF columns required by BDD100K failure-mode views.",
        argv_template=[
            "bash",
            "-c",
            (
                "set -euo pipefail; "
                "for udf in has_person has_rider person_bbox_area_pct dhash is_duplicate; do "
                "npa workbench lancedb backfill "
                "--udf \"${udf}\" "
                "--table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--batch-size 512 "
                "--service "
                "--endpoint {{config.lancedb_endpoint}}; "
                "done"
            ),
        ],
    ),
    "workbench.lancedb.backfill_clip": ToolEntry(
        name="workbench.lancedb.backfill_clip",
        description="Backfill CLIP embeddings for BDD100K rows.",
        argv_template=[
            "npa",
            "workbench",
            "lancedb",
            "backfill",
            "--udf",
            "clip_embedding",
            "--table",
            "{{config.lance_table}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--batch-size",
            "32",
            "--service",
            "--endpoint",
            "{{config.lancedb_endpoint}}",
        ],
    ),
    "workbench.lancedb.create_failure_views": ToolEntry(
        name="workbench.lancedb.create_failure_views",
        description="Create rider, nighttime-person, and distant-person materialized views.",
        argv_template=[
            "bash",
            "-c",
            (
                "set -euo pipefail; "
                "npa workbench lancedb create-mv "
                "--name {{config.rider_view}} "
                "--filter \"has_rider = true AND split = 'train'\" "
                "--table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--service --endpoint {{config.lancedb_endpoint}}; "
                "npa workbench lancedb create-mv "
                "--name {{config.nighttime_view}} "
                "--filter \"timeofday = 'night' AND has_person = true AND split = 'train'\" "
                "--table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--service --endpoint {{config.lancedb_endpoint}}; "
                "npa workbench lancedb create-mv "
                "--name {{config.distant_view}} "
                "--filter \"has_person = true AND person_bbox_area_pct < 0.01 AND split = 'train'\" "
                "--table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--service --endpoint {{config.lancedb_endpoint}}"
            ),
        ],
    ),
    "workbench.detection_training.train_rider": ToolEntry(
        name="workbench.detection_training.train_rider",
        description="Train a detector on the rider failure-mode view.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "train",
            "--view",
            "{{config.rider_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.rider_train_uri}}",
            "--epochs",
            "{{config.train_epochs}}",
            "--batch-size",
            "{{config.train_batch_size}}",
            "--learning-rate",
            "{{config.train_learning_rate}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.detection_training.train_nighttime": ToolEntry(
        name="workbench.detection_training.train_nighttime",
        description="Train a detector on the nighttime-person failure-mode view.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "train",
            "--view",
            "{{config.nighttime_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.nighttime_train_uri}}",
            "--epochs",
            "{{config.train_epochs}}",
            "--batch-size",
            "{{config.train_batch_size}}",
            "--learning-rate",
            "{{config.train_learning_rate}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.detection_training.train_distant": ToolEntry(
        name="workbench.detection_training.train_distant",
        description="Train a detector on the distant-person failure-mode view.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "train",
            "--view",
            "{{config.distant_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.distant_train_uri}}",
            "--epochs",
            "{{config.train_epochs}}",
            "--batch-size",
            "{{config.train_batch_size}}",
            "--learning-rate",
            "{{config.train_learning_rate}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.detection_training.eval_rider": ToolEntry(
        name="workbench.detection_training.eval_rider",
        description="Evaluate the rider detector checkpoint.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "eval",
            "--checkpoint-uri",
            "{{config.rider_train_uri}}",
            "--eval-view",
            "{{config.rider_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.rider_eval_uri}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.detection_training.eval_nighttime": ToolEntry(
        name="workbench.detection_training.eval_nighttime",
        description="Evaluate the nighttime-person detector checkpoint.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "eval",
            "--checkpoint-uri",
            "{{config.nighttime_train_uri}}",
            "--eval-view",
            "{{config.nighttime_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.nighttime_eval_uri}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.detection_training.eval_distant": ToolEntry(
        name="workbench.detection_training.eval_distant",
        description="Evaluate the distant-person detector checkpoint.",
        argv_template=[
            "npa",
            "workbench",
            "detection-training",
            "eval",
            "--checkpoint-uri",
            "{{config.distant_train_uri}}",
            "--eval-view",
            "{{config.distant_view}}",
            "--lance-uri",
            "{{config.lance_uri}}",
            "--output-uri",
            "{{config.distant_eval_uri}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
        ],
    ),
    "workbench.fiftyone.launch_app": ToolEntry(
        name="workbench.fiftyone.launch_app",
        description="Launch FiftyOne App for pipeline review (workflow stub).",
        argv_template=["echo", "fiftyone review run {{run.id}} lance {{config.lance_uri}}"],
        stub=True,
    ),
}


def validate_tool_ref(tool_ref: str) -> ToolEntry:
    entry = TOOL_CATALOG.get(tool_ref)
    if entry is None:
        known = ", ".join(sorted(TOOL_CATALOG))
        raise NpaWorkflowError(f"unknown toolRef {tool_ref!r} (known: {known})")
    return entry


def argv_for_tool(tool_ref: str) -> list[str]:
    return list(validate_tool_ref(tool_ref).argv_template)
