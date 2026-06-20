"""CLI for the Sim2Real VLM-to-RL workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.constants import (
    DEFAULT_ACTION_ENV_LIMIT,
    DEFAULT_ENVGEN_SHARD_COUNT,
    DEFAULT_K8S_MAX_PARALLEL_GPUS,
    DEFAULT_HELDOUT_ENVS,
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_ISAAC_TASK,
    DEFAULT_LEROBOT_DATASET_ID,
    DEFAULT_LOOP_OF_LOOPS_ITERATIONS,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_PREFIX,
    DEFAULT_REFERENCE_VLM_MODEL,
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_MODEL,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_SIM_BACKEND,
    DEFAULT_STEPS_PER_ROLLOUT,
    DEFAULT_THRESHOLD,
    DEFAULT_TRAIN_FRACTION,
    SIM_BACKENDS,
)
from npa.workflows.sim2real.engine import (
    _config_from_workflow_state,
    _read_workflow_state,
    _write_json_artifact,
    _write_workflow_state,
    convert_vlm_eval_to_rl_signal,
    run_cosmos2_transfer_component_from_s3,
    run_finalize,
    run_heldout_eval_component_from_s3,
    run_inner_loop,
    run_policy_actions_component_from_s3,
    run_preamble,
    run_single_outer_iteration,
    run_vlm_eval_component_from_s3,
)
from npa.workflows.sim2real.models import Sim2RealLoopError, new_run_id
from npa.workflows.sim2real.runner import Sim2RealWorkflow, run_full_loop
from npa.workflows.sim2real.utils import _bool_value, _utc_now

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
    parser.add_argument(
        "--cameras-uri",
        default=os.environ.get(
            "NPA_SIM2REAL_CAMERAS_URI", os.environ.get("CAMERAS_URI", "")
        ),
    )
    parser.add_argument(
        "--robot-spec-uri", default=os.environ.get("ROBOT_SPEC_URI", "")
    )
    parser.add_argument("--robot-source", default=os.environ.get("ROBOT_SOURCE", ""))
    parser.add_argument("--robot-preset", default=os.environ.get("ROBOT_PRESET", ""))
    parser.add_argument("--augment-image", default=os.environ.get("AUGMENT_IMAGE", ""))
    parser.add_argument("--envgen-image", default=os.environ.get("ENVGEN_IMAGE", ""))
    parser.add_argument(
        "--env-count", type=int, default=int(os.environ.get("NPA_ENV_COUNT", "0"))
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=float(os.environ.get("NPA_TRAIN_FRACTION", DEFAULT_TRAIN_FRACTION)),
    )
    parser.add_argument(
        "--envgen-shard-count",
        type=int,
        default=int(os.environ.get("NPA_ENVGEN_SHARD_COUNT", DEFAULT_ENVGEN_SHARD_COUNT)),
    )
    parser.add_argument(
        "--action-env-limit",
        type=int,
        default=int(os.environ.get("NPA_ACTION_ENV_LIMIT", DEFAULT_ACTION_ENV_LIMIT)),
    )
    parser.add_argument("--policy-image", default=os.environ.get("POLICY_IMAGE", ""))
    parser.add_argument("--trainer-image", default=os.environ.get("TRAINER_IMAGE", ""))
    parser.add_argument("--vlm-image", default=os.environ.get("VLM_IMAGE", ""))
    parser.add_argument("--eval-image", default=os.environ.get("EVAL_IMAGE", ""))
    parser.add_argument("--isaac-image", default=os.environ.get("ISAAC_IMAGE", ""))
    parser.add_argument(
        "--sim-backend",
        default=os.environ.get("NPA_SIM2REAL_SIM_BACKEND", DEFAULT_SIM_BACKEND),
        choices=list(SIM_BACKENDS),
    )
    parser.add_argument(
        "--isaac-task",
        default=os.environ.get("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK),
    )
    parser.add_argument(
        "--vlm-model", default=os.environ.get("VLM_MODEL", DEFAULT_REFERENCE_VLM_MODEL)
    )
    parser.add_argument(
        "--vlm-reason2-model",
        default=os.environ.get("VLM_REASON2_MODEL", DEFAULT_REASON2_MODEL),
    )
    parser.add_argument(
        "--vlm-reason3-model",
        default=os.environ.get("VLM_REASON3_MODEL", DEFAULT_REASON3_MODEL),
    )
    parser.add_argument(
        "--vlm-reason2-image", default=os.environ.get("VLM_REASON2_IMAGE", "")
    )
    parser.add_argument(
        "--vlm-reason3-image", default=os.environ.get("VLM_REASON3_IMAGE", "")
    )
    parser.add_argument(
        "--vlm-dual-reason",
        action=argparse.BooleanOptionalAction,
        default=_bool_value(os.environ.get("NPA_SIM2REAL_VLM_DUAL_REASON", "1")),
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
    parser.add_argument("--byo-rerun-command", default="")
    parser.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        default=_bool_value(os.environ.get("NPA_SIM2REAL_RERUN", "1")),
        help="Emit a Rerun .rrd visualization after the loop (default on).",
    )
    parser.add_argument(
        "--no-rerun",
        dest="rerun",
        action="store_false",
        help="Disable Rerun .rrd visualization emission.",
    )
    parser.add_argument(
        "--k8s-namespace",
        default=os.environ.get("NPA_SIM2REAL_K8S_NAMESPACE", ""),
    )
    parser.add_argument(
        "--k8s-service-account",
        default=os.environ.get("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa"),
    )
    parser.add_argument(
        "--k8s-image-pull-secrets",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_IMAGE_PULL_SECRETS",
            "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry",
        ),
    )
    parser.add_argument(
        "--k8s-env-secret-names",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_ENV_SECRET_NAMES",
            "hf-ngc-tokens,npa-storage-credentials",
        ),
    )
    parser.add_argument(
        "--k8s-gpu-resource",
        default=os.environ.get("NPA_SIM2REAL_K8S_GPU_RESOURCE", "nvidia.com/gpu"),
    )
    parser.add_argument(
        "--k8s-gpu-product",
        default=os.environ.get(
            "NPA_SIM2REAL_K8S_GPU_PRODUCT",
            "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        ),
    )
    parser.add_argument(
        "--k8s-kubeconfig",
        default=os.environ.get("NPA_SIM2REAL_KUBECONFIG", os.environ.get("KUBECONFIG", "")),
    )
    parser.add_argument(
        "--k8s-context",
        default=os.environ.get("NPA_SIM2REAL_K8S_CONTEXT", ""),
    )
    parser.add_argument(
        "--k8s-job-timeout-s",
        type=int,
        default=int(os.environ.get("NPA_SIM2REAL_K8S_JOB_TIMEOUT_S", "7200")),
    )
    parser.add_argument(
        "--k8s-max-parallel-gpus",
        type=int,
        default=int(
            os.environ.get(
                "NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS",
                DEFAULT_K8S_MAX_PARALLEL_GPUS,
            )
        ),
    )
    parser.add_argument("--source-repo", default=os.environ.get("NPA_SOURCE_REPO", ""))
    parser.add_argument("--source-ref", default=os.environ.get("NPA_SOURCE_REF", ""))
    parser.add_argument(
        "--heldout-eval-limit",
        type=int,
        default=int(os.environ.get("NPA_SIM2REAL_HELDOUT_EVAL_LIMIT", "0")),
    )

def main(argv: list[str] | None = None) -> int:
    """Module CLI for raw SkyPilot YAML and local smoke runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    full = subparsers.add_parser(
        "full-loop", help="Run the full Stage 1-13 Sim2Real workflow."
    )
    _add_common_args(full)
    preamble = subparsers.add_parser(
        "preamble", help="Run Stage 1-6 setup and persist workflow state."
    )
    _add_common_args(preamble)
    outer = subparsers.add_parser(
        "outer-iteration", help="Run one Stage 7-11 outer iteration from saved state."
    )
    _add_common_args(outer)
    outer.add_argument("--outer-iteration", type=int, required=True)
    outer.add_argument("--initial-quality", type=float, default=None)
    finalize = subparsers.add_parser(
        "finalize", help="Run Stage 12-13/report/upload from saved state."
    )
    _add_common_args(finalize)

    run_cmd = subparsers.add_parser(
        "run",
        help="Run the full workflow via Sim2RealWorkflow (canonical orchestrator).",
    )
    _add_common_args(run_cmd)
    run_cmd.add_argument(
        "--initial-quality",
        type=float,
        default=None,
        help="Override quality seed when resuming staged state.",
    )
    inner = subparsers.add_parser(
        "inner-loop", help="Run only the VLM-to-RL inner loop."
    )
    _add_common_args(inner)
    convert = subparsers.add_parser(
        "convert-signal", help="Convert one VLM eval JSON to RL signal JSON."
    )
    convert.add_argument("--vlm-json", type=Path, required=True)
    convert.add_argument("--output-json", type=Path, required=True)
    component_vlm = subparsers.add_parser(
        "component-vlm-eval", help="Run one sibling-image VLM component contract."
    )
    component_vlm.add_argument("--input-uri", required=True)
    component_vlm.add_argument("--output-uri", required=True)
    component_vlm.add_argument("--rollout-id", default="")
    component_vlm.add_argument("--model", default=DEFAULT_REFERENCE_VLM_MODEL)
    component_vlm.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    component_heldout = subparsers.add_parser(
        "component-heldout-eval",
        help="Run one sibling-image held-out eval component contract.",
    )
    component_heldout.add_argument("--heldout-envs-uri", required=True)
    component_heldout.add_argument("--inner-evidence-uri", required=True)
    component_heldout.add_argument("--output-uri", required=True)
    component_heldout.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    component_heldout.add_argument("--limit", type=int, default=0)
    component_heldout.add_argument("--scene-spec-uri", default="")
    component_heldout.add_argument("--cameras-uri", default="")
    component_heldout.add_argument("--assets-uri", default="")
    component_heldout.add_argument("--byo-mesh-uri", default="")
    component_heldout.add_argument("--robot-spec-uri", default="")
    component_heldout.add_argument("--robot-source", default="")
    component_heldout.add_argument("--robot-preset", default="")
    component_heldout.add_argument(
        "--sim-backend",
        default=os.environ.get("NPA_SIM2REAL_SIM_BACKEND", DEFAULT_SIM_BACKEND),
        choices=list(SIM_BACKENDS),
    )
    component_heldout.add_argument(
        "--isaac-task",
        default=os.environ.get("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK),
    )
    component_cosmos = subparsers.add_parser(
        "component-cosmos2-transfer",
        help="Run Cosmos Transfer 2.5 augment in a sibling GPU job.",
    )
    component_cosmos.add_argument("--input-uri", required=True)
    component_cosmos.add_argument("--output-uri", required=True)
    component_cosmos.add_argument("--augmented-frames-uri", required=True)
    component_cosmos.add_argument("--assets-uri", default="")
    component_cosmos.add_argument("--scene-spec-uri", default="")
    component_cosmos.add_argument("--image", default="")
    component_cosmos.add_argument("--run-id", default="")
    component_policy = subparsers.add_parser(
        "component-policy-actions",
        help="Run swappable LeRobot policy container for Stage 7 rollouts.",
    )
    component_policy.add_argument("--train-envs-uri", required=True)
    component_policy.add_argument("--output-uri", required=True)
    component_policy.add_argument("--policy-image", required=True)
    component_policy.add_argument("--limit", type=int, default=DEFAULT_ACTION_ENV_LIMIT)
    component_policy.add_argument("--seed", type=int, default=42)
    component_policy.add_argument("--run-id", default="")
    component_policy.add_argument("--rollout-count", type=int, default=DEFAULT_ROLLOUT_COUNT)
    component_policy.add_argument(
        "--steps-per-rollout", type=int, default=DEFAULT_STEPS_PER_ROLLOUT
    )
    status_cmd = subparsers.add_parser(
        "status",
        help="Live stage progress for a staged cluster run (S3 artifacts + kubectl).",
    )
    status_cmd.add_argument("run_id", help="Sim2Real run id.")
    status_cmd.add_argument(
        "--s3-bucket",
        default=os.environ.get("NPA_SIM2REAL_BUCKET", os.environ.get("S3_BUCKET", "")),
    )
    status_cmd.add_argument(
        "--s3-prefix",
        default=os.environ.get("NPA_SIM2REAL_PREFIX", DEFAULT_PREFIX),
    )
    status_cmd.add_argument(
        "--s3-endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_S3_ENDPOINT),
    )
    status_cmd.add_argument(
        "--k8s-context",
        default=os.environ.get("NPA_SIM2REAL_K8S_CONTEXT", ""),
    )
    status_cmd.add_argument(
        "--k8s-namespace",
        default=os.environ.get("NPA_SIM2REAL_K8S_NAMESPACE", "default"),
    )
    status_cmd.add_argument(
        "--watch",
        action="store_true",
        help="Refresh until the run reaches a terminal state.",
    )
    status_cmd.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Watch refresh interval in seconds.",
    )
    status_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    if args.command == "status":
        from npa.workflows.sim2real.monitor import watch_sim2real_status

        watch_sim2real_status(
            args.run_id,
            watch=args.watch,
            interval=args.interval,
            json_output=args.json,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint=args.s3_endpoint,
            k8s_context=args.k8s_context,
            k8s_namespace=args.k8s_namespace,
        )
        return 0
    if args.command == "convert-signal":
        payload = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        _write_json_artifact(args.output_json, convert_vlm_eval_to_rl_signal(payload))
        return 0
    if args.command == "component-vlm-eval":
        run_vlm_eval_component_from_s3(
            input_uri=args.input_uri,
            output_uri=args.output_uri,
            rollout_id=args.rollout_id,
            model=args.model,
            threshold=args.threshold,
        )
        return 0
    if args.command == "component-heldout-eval":
        run_heldout_eval_component_from_s3(
            heldout_envs_uri=args.heldout_envs_uri,
            inner_evidence_uri=args.inner_evidence_uri,
            output_uri=args.output_uri,
            threshold=args.threshold,
            limit=args.limit,
            scene_spec_uri=args.scene_spec_uri,
            cameras_uri=args.cameras_uri,
            assets_uri=args.assets_uri,
            byo_mesh_uri=args.byo_mesh_uri,
            robot_spec_uri=args.robot_spec_uri,
            robot_source=args.robot_source,
            robot_preset=args.robot_preset,
            sim_backend=args.sim_backend,
            isaac_task=args.isaac_task,
        )
        return 0
    if args.command == "component-cosmos2-transfer":
        run_cosmos2_transfer_component_from_s3(
            input_uri=args.input_uri,
            output_uri=args.output_uri,
            augmented_frames_uri=args.augmented_frames_uri,
            assets_uri=args.assets_uri,
            scene_spec_uri=args.scene_spec_uri,
            image=args.image,
            run_id=args.run_id,
        )
        return 0
    if args.command == "component-policy-actions":
        run_policy_actions_component_from_s3(
            train_envs_uri=args.train_envs_uri,
            output_uri=args.output_uri,
            policy_image=args.policy_image,
            limit=args.limit,
            seed=args.seed,
            run_id=args.run_id,
            rollout_count=args.rollout_count,
            steps_per_rollout=args.steps_per_rollout,
        )
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
        cameras_uri=args.cameras_uri,
        robot_spec_uri=args.robot_spec_uri,
        robot_source=args.robot_source,
        robot_preset=args.robot_preset,
        augment_image=args.augment_image,
        envgen_image=args.envgen_image,
        env_count=args.env_count,
        train_fraction=args.train_fraction,
        envgen_shard_count=args.envgen_shard_count,
        action_env_limit=args.action_env_limit,
        policy_image=args.policy_image,
        trainer_image=args.trainer_image,
        vlm_image=args.vlm_image,
        vlm_reason2_image=args.vlm_reason2_image,
        vlm_reason3_image=args.vlm_reason3_image,
        eval_image=args.eval_image,
        isaac_image=args.isaac_image,
        sim_backend=args.sim_backend,
        isaac_task=args.isaac_task,
        vlm_model=args.vlm_model,
        vlm_reason2_model=args.vlm_reason2_model,
        vlm_reason3_model=args.vlm_reason3_model,
        vlm_dual_reason=args.vlm_dual_reason,
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
        byo_rerun_command=args.byo_rerun_command,
        byo_policy_command=getattr(args, "byo_policy_command", ""),
        rerun_enabled=args.rerun,
        k8s_namespace=args.k8s_namespace,
        k8s_service_account=args.k8s_service_account,
        k8s_image_pull_secrets=args.k8s_image_pull_secrets,
        k8s_env_secret_names=args.k8s_env_secret_names,
        k8s_gpu_resource=args.k8s_gpu_resource,
        k8s_gpu_product=args.k8s_gpu_product,
        k8s_kubeconfig=args.k8s_kubeconfig,
        k8s_context=args.k8s_context,
        k8s_job_timeout_s=args.k8s_job_timeout_s,
        k8s_max_parallel_gpus=args.k8s_max_parallel_gpus,
        source_repo=args.source_repo,
        source_ref=args.source_ref,
        heldout_eval_limit=args.heldout_eval_limit,
    )

    if args.command == "run":
        workflow = Sim2RealWorkflow(config)
        report = workflow.run_staged(
            upload=True if args.upload_artifacts else None,
            initial_quality=getattr(args, "initial_quality", None),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        # A command that orchestrates sub-operations must exit non-zero when one
        # of them failed — a "blocked"/"failed" upload recorded in the JSON report
        # is NOT a substitute for a non-zero exit code (set -e, CI gates, and the
        # submit/monitor wrappers all rely on the exit code). rerun-serve stays a
        # best-effort warning (the engine itself records it as WARN/CONTINUE).
        upload_status = str((report.get("upload") or {}).get("status", ""))
        if upload_status in {"blocked", "failed"}:
            reason = str((report.get("upload") or {}).get("reason", "")) or "see report"
            print(
                f"ERROR: artifact upload {upload_status}: {reason}",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.command == "preamble":
        state = run_preamble(config)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if args.command == "outer-iteration":
        local_dir = config.output_dir
        if local_dir is None:
            raise Sim2RealLoopError("--output-dir is required for outer-iteration")
        state = _read_workflow_state(local_dir)
        config = _config_from_workflow_state(config, state)
        initial_quality = (
            float(args.initial_quality)
            if args.initial_quality is not None
            else float(state.get("current_quality", 0.38))
        )
        iteration = run_single_outer_iteration(
            config,
            local_dir=local_dir,
            outer_iteration=int(args.outer_iteration),
            initial_quality=initial_quality,
        )
        state["final_inner"] = iteration["inner"]
        state["final_eval"] = iteration["heldout_report"]
        state["final_decision"] = iteration["decision"]
        state.setdefault("outer_history", []).append(iteration["history_entry"])
        state["current_quality"] = iteration["next_quality"]
        state["next_outer_iteration"] = int(args.outer_iteration) + 1
        state["status"] = "outer_iteration_completed"
        state["updated_at"] = _utc_now()
        _write_workflow_state(local_dir, state)
        print(json.dumps(iteration, indent=2, sort_keys=True))
        return 0
    if args.command == "finalize":
        local_dir = config.output_dir
        if local_dir is None:
            raise Sim2RealLoopError("--output-dir is required for finalize")
        state = _read_workflow_state(local_dir)
        final_inner = state.get("final_inner")
        final_eval = state.get("final_eval")
        final_decision = state.get("final_decision")
        if not final_inner or not final_eval or not final_decision:
            raise Sim2RealLoopError(
                "cannot finalize before an outer iteration has produced decision artifacts"
            )
        report = run_finalize(
            config,
            local_dir=local_dir,
            stage_records=list(state.get("stage_records", [])),
            components=list(state.get("components", [])),
            outer_history=list(state.get("outer_history", [])),
            final_inner=dict(final_inner),
            final_eval=dict(final_eval),
            final_decision=dict(final_decision),
        )
        state["status"] = "completed"
        state["updated_at"] = _utc_now()
        state["report_path"] = str(local_dir / "reports" / "sim2real-report.json")
        _write_workflow_state(local_dir, state)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
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

if __name__ == "__main__":
    raise SystemExit(main())
