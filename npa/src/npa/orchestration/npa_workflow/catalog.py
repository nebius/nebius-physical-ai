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


# Public composable entries intentionally available to customer-authored specs,
# even though no shipped reference spec consumes them today. Everything else in
# TOOL_CATALOG must be reachable from at least one shipped spec.
PUBLIC_REUSABLE_TOOLREFS: dict[str, str] = {
    "infra.fleet.deploy": "public npa.fleet deployment primitive",
    "infra.soperator.deploy": "public npa.soperator deployment primitive",
    "workbench.cosmos2.transfer": "public Cosmos Transfer composition primitive",
    "workbench.foxglove.convert": "public recording-conversion primitive",
    "workbench.insights.record": "public lineage/metrics ingestion primitive",
    "workbench.isaac_lab.byof_repo": "public Isaac Lab BYOF primitive",
    "workbench.lerobot.eval": "public LeRobot evaluation primitive",
}


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
    "--build-command",
    "{{config.build_command}}",
    "--workload",
    "{{config.workload}}",
    "--smoke-command",
    "{{config.smoke_command}}",
    "--solution-name",
    "{{config.solution_name}}",
    "--capability-name",
    "{{config.capability_name}}",
    "--smoke-artifact-name",
    "{{config.smoke_artifact_name}}",
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
    "infra.fleet.deploy": ToolEntry(
        name="infra.fleet.deploy",
        description=(
            "Deploy a fleet of Nebius Managed Kubernetes (k8s-training) clusters "
            "across one or many projects in a tenant from an npa.fleet/v0.0.1 spec "
            "(identical and/or custom clusters; creates projects on demand). Set "
            "the spec's 'profile' to target a tenant other than the active "
            "~/.nebius profile."
        ),
        argv_template=[
            "npa",
            "fleet",
            "deploy",
            "--spec",
            "{{config.fleet_spec}}",
            # Workflow states are non-interactive: without --yes the deploy
            # confirmation prompt would block the run forever.
            "--yes",
            "--output",
            "json",
        ],
    ),
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
    # --- NuRec / NRE neural reconstruction -----------------------------------
    # Every verb is a real entrypoint in npa/src/npa/cli/nurec/__init__.py that
    # drives the real NVIDIA NRE container; none of these write a manifest stub.
    "workbench.nurec.check": ToolEntry(
        name="workbench.nurec.check",
        description=(
            "Verify NRE container pullability, real Hugging Face download "
            "authorization, and that the GPU has RT cores, before any GPU work."
        ),
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "check",
            "--image",
            "{{config.nurec_image}}",
            "--dataset",
            "{{config.dataset_id}}",
            "--scene",
            "{{config.scene}}",
            "--variant",
            "{{config.variant}}",
            "--require-gpu",
            "--output",
            "json",
        ],
    ),
    "workbench.nurec.fetch": ToolEntry(
        name="workbench.nurec.fetch",
        description=(
            "Download and unpack real NCore V4 shards and derive the rig->world "
            "pose edge NRE requires for object-centric captures."
        ),
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "fetch",
            "--dataset",
            "{{config.dataset_id}}",
            "--scene",
            "{{config.scene}}",
            "--variant",
            "{{config.variant}}",
            "--cache-dir",
            "{{config.cache_dir}}",
            "--derive-rig",
            # Each stage is its own pod, so the sequence must travel through S3.
            # `fetch` publishes under `<ncore_uri>sequence/` and `reconstruct` reads
            # `config.ncore_sequence_uri`; those two MUST agree. The relationship is
            # asserted by npa/tests/workbench/test_nurec_access.py::
            # test_spec_ncore_sequence_uri_matches_what_fetch_publishes, because a
            # spec that sets only `ncore_uri` would otherwise fail silently in a
            # different pod with an empty materialization.
            "--publish-sequence",
            "--output-uri",
            "{{config.ncore_uri}}",
            "--output",
            "json",
        ],
    ),
    "workbench.nurec.reconstruct": ToolEntry(
        name="workbench.nurec.reconstruct",
        description=(
            "Train a 3DGUT Gaussian reconstruction with NRE into a renderable "
            "USDZ, with real val metrics and exported ground-truth frames."
        ),
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "reconstruct",
            "--ncore-uri",
            "{{config.ncore_sequence_uri}}",
            "--config-name",
            "{{config.config_name}}",
            "--mode",
            "{{config.mode}}",
            "--poses-component-group",
            "{{config.poses_component_group}}",
            "--cache-dir",
            "{{config.cache_dir}}",
            "--out-dir",
            "{{config.out_dir}}",
            "--max-epochs",
            "{{config.max_epochs}}",
            "--world-size",
            "{{config.world_size}}",
            "--image",
            "{{config.nurec_image}}",
            "--export-gt",
            "--output-uri",
            "{{config.reconstruction_uri}}",
            "--input-uri",
            "{{config.input_uri}}",
            "--output",
            "json",
        ],
    ),
    "workbench.nurec.render": ToolEntry(
        name="workbench.nurec.render",
        description=(
            "Render novel views from a trained reconstruction with `nre render` "
            "using a rig offset (not the training views)."
        ),
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "render",
            "--artifact-uri",
            "{{config.reconstruction_uri}}",
            "--out-dir",
            "{{config.out_dir}}",
            "--output-dir",
            "{{config.render_dir}}",
            "--image-scale",
            "{{config.render_image_scale}}",
            "--renderer",
            "{{config.renderer}}",
            "--rig-translation-offset",
            "{{config.rig_translation_offset}}",
            "--rig-rotation-offset",
            "{{config.rig_rotation_offset}}",
            "--no-replicate-training-views",
            "--output-uri",
            "{{config.novel_views_uri}}",
            "--output",
            "json",
        ],
    ),
    "workbench.nurec.visualize": ToolEntry(
        name="workbench.nurec.visualize",
        description=(
            "Build the run's Rerun recording (reports/sim2real.rrd) so the "
            "reconstruction renders in the NPA agent's embedded viewer."
        ),
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "visualize",
            "--input-uri",
            "{{config.run_root_uri}}",
            "--output-uri",
            "{{config.rrd_uri}}",
            "--output",
            "json",
        ],
    ),
    "workbench.nurec.finalize": ToolEntry(
        name="workbench.nurec.finalize",
        description="Aggregate a NuRec run tree into a real final report.",
        argv_template=[
            "npa",
            "workbench",
            "nurec",
            "finalize",
            "--input-uri",
            "{{config.run_root_uri}}",
            "--output-uri",
            "{{config.final_report_uri}}",
            "--run-id",
            "{{run.id}}",
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
    "workbench.vlm_eval.judge_against_plan": ToolEntry(
        name="workbench.vlm_eval.judge_against_plan",
        description=(
            "Score a rollout against the plan an earlier reasoning stage produced."
        ),
        argv_template=[
            "npa",
            "workbench",
            "vlm-eval",
            "run",
            "--input-path",
            "{{config.rollouts_uri}}",
            "--output-path",
            "{{config.scores_uri}}",
            # The judge's task comes from the reasoner's artifact, not a literal: that is what
            # makes this a three-stage combo rather than two unrelated stages.
            "--task-from",
            "{{config.plan_uri}}scene_reasoning.json",
            "--backend",
            "{{config.vlm_backend}}",
            "--api-key-env",
            "NEBIUS_TOKEN_FACTORY_KEY",
            "--frame-selection",
            "{{config.vlm_frame_selection}}",
            "--max-frames",
            "{{config.vlm_max_frames}}",
            "--success-threshold",
            "{{config.vlm_success_threshold}}",
        ],
    ),
    "workbench.vlm_eval.loop": ToolEntry(
        name="workbench.vlm_eval.loop",
        description=(
            "Score every rollout under a prefix and write an aggregate task-success report."
        ),
        argv_template=[
            "npa",
            "workbench",
            "vlm-eval",
            "loop",
            "--input-path",
            "{{config.rollouts_uri}}",
            "--output-path",
            "{{config.scores_uri}}",
            "--task",
            "{{config.vlm_task}}",
            "--backend",
            "{{config.vlm_backend}}",
            "--frame-selection",
            "{{config.vlm_frame_selection}}",
            "--max-frames",
            "{{config.vlm_max_frames}}",
            "--success-threshold",
            "{{config.vlm_success_threshold}}",
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
            "--input-uri",
            "{{config.trigger_uri}}",
            "--output-uri",
            "{{config.augment_uri}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workbench.cosmos2.transfer_execute": ToolEntry(
        name="workbench.cosmos2.transfer_execute",
        description="Run the REAL Cosmos-Transfer2.5 model (GPU) and upload augmented video + frames to S3.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "{{config.trigger_uri}}",
            "--output-uri",
            "{{config.augment_uri}}",
            "--run-id",
            "{{run.id}}",
            "--configs-uri",
            "{{config.configs_uri}}",
            "--condition-on-input",
            "--execute",
        ],
    ),
    "workbench.cosmos2.transfer_conditioned_execute": ToolEntry(
        name="workbench.cosmos2.transfer_conditioned_execute",
        description=(
            "Run the REAL Cosmos-Transfer2.5 model conditioned on the input video "
            "and upload its video, exact frame list, and manifest to S3."
        ),
        argv_template=[
            "npa",
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "{{config.trigger_uri}}",
            "--output-uri",
            "{{config.augment_uri}}",
            "--run-id",
            "{{run.id}}",
            "--execute",
            "--condition-on-input",
        ],
    ),
    "workbench.cosmos3.text_to_image": ToolEntry(
        name="workbench.cosmos3.text_to_image",
        description="Generate an image from a prompt with the Cosmos3 framework and publish it.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos3",
            "text-to-image",
            "--prompt",
            "{{config.t2i_prompt}}",
            "--output-uri",
            "{{config.t2i_output_uri}}",
            "--model-id",
            "{{config.cosmos_model_id}}",
            "--checkpoint-name",
            "{{config.t2i_checkpoint_name}}",
            "--source-repo-url",
            "{{config.cosmos_source_repo}}",
            "--cache-dir",
            "{{config.cosmos_cache_dir}}",
            "--uv-group",
            "{{config.t2i_uv_group}}",
            "--seed",
            "{{config.t2i_seed}}",
            # The framework's guardrails pull further gated weights; the template defaulted them
            # off too (NPA_COSMOS3_NO_GUARDRAILS).
            "--no-guardrails",
        ],
    ),
    "workbench.cosmos.check": ToolEntry(
        name="workbench.cosmos.check",
        description="Check Cosmos source and checkpoint access before fetching them.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos",
            "check",
            "--source-repo-url",
            "{{config.cosmos_source_repo}}",
            "--model-id",
            "{{config.cosmos_model_id}}",
            "--cache-dir",
            "{{config.cosmos_cache_dir}}",
            "--hf-token-env",
            "{{config.cosmos_hf_token_env}}",
            "--reasoning-parser",
            "{{config.cosmos_reasoning_parser}}",
            "--tool-call-parser",
            "{{config.cosmos_tool_call_parser}}",
            "--output",
            "json",
        ],
    ),
    "workbench.cosmos_evaluator.evaluate": ToolEntry(
        name="workbench.cosmos_evaluator.evaluate",
        description=(
            "Grade augmented variants with the REAL NVIDIA Cosmos Evaluator checks "
            "(hallucination + VLM attribute verification, Apache-2.0) plus the "
            "NPA source-relative temporal consistency companion diagnostic."
            " It also reports source-relative protected-appearance fidelity for "
            "excessive global colour cast or localized material recolouring."
        ),
        argv_template=[
            "npa",
            "workbench",
            "cosmos-evaluator",
            "evaluate",
            "--augment-uri",
            "{{config.rollouts_uri}}",
            "--output-uri",
            "{{config.scores_uri}}",
            "--input-uri",
            "{{config.input_uri}}",
            "--configs-uri",
            "{{config.configs_uri}}",
            "--threshold",
            "{{config.grade_threshold}}",
            "--temporal-threshold",
            "{{config.temporal_consistency_threshold}}",
            "--temporal-regions-json",
            "{{config.temporal_regions_json}}",
            "--temporal-mode",
            "{{config.temporal_consistency_mode}}",
            "--temporal-noise-floor",
            "{{config.temporal_noise_floor}}",
            "--temporal-blur-ksize",
            "{{config.temporal_blur_ksize}}",
            "--appearance-threshold",
            "{{config.appearance_fidelity_threshold}}",
            "--appearance-regions-json",
            "{{config.appearance_regions_json}}",
            "--appearance-mode",
            "{{config.appearance_fidelity_mode}}",
            "--appearance-luminance-tolerance",
            "{{config.appearance_luminance_tolerance}}",
            "--appearance-global-chroma-tolerance",
            "{{config.appearance_global_chroma_tolerance}}",
            "--appearance-local-chroma-tolerance",
            "{{config.appearance_local_chroma_tolerance}}",
            "--appearance-chroma-instability-tolerance",
            "{{config.appearance_chroma_instability_tolerance}}",
            "--appearance-blur-ksize",
            "{{config.appearance_blur_ksize}}",
            "--appearance-max-dimension",
            "{{config.appearance_max_dimension}}",
            "--vlm-model",
            "{{config.caption_model}}",
            "--output",
            "json",
        ],
    ),
    "workbench.cosmos.fetch": ToolEntry(
        name="workbench.cosmos.fetch",
        description="Materialize Cosmos source and checkpoint into a local cache.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos",
            "fetch",
            "--source-repo-url",
            "{{config.cosmos_source_repo}}",
            "--model-id",
            "{{config.cosmos_model_id}}",
            "--cache-dir",
            "{{config.cosmos_cache_dir}}",
            "--hf-token-env",
            "{{config.cosmos_hf_token_env}}",
            "--reasoning-parser",
            "{{config.cosmos_reasoning_parser}}",
            "--tool-call-parser",
            "{{config.cosmos_tool_call_parser}}",
            "--output",
            "json",
        ],
    ),
    "workbench.cosmos_curate.curate": ToolEntry(
        name="workbench.cosmos_curate.curate",
        description=(
            "Curate augmented variants with the REAL NVIDIA Cosmos Curator stages "
            "(split, transcode, motion-score, canonical clip metadata; Apache-2.0)."
        ),
        argv_template=[
            "npa",
            "workbench",
            "cosmos-curate",
            "curate-augmented",
            "--augment-uri",
            "{{config.augment_uri}}",
            "--curated-uri",
            "{{config.curated_clips_uri}}",
            "--report-uri",
            "{{config.curator_report_uri}}",
            "--clip-len-s",
            "{{config.curator_clip_len_s}}",
            "--min-clip-length-s",
            "{{config.curator_min_clip_len_s}}",
            "--motion-filter",
            "{{config.curator_motion_filter}}",
            "--output",
            "json",
        ],
    ),
    "workbench.sim2real_envgen.raw_shard": ToolEntry(
        name="workbench.sim2real_envgen.raw_shard",
        description="Generate raw simulation env shard.",
        argv_template=[
            "npa",
            "workbench",
            "sim2real-envgen",
            "raw-shard",
            "--run-id",
            "{{run.id}}",
            # The module takes the RUN ROOT and derives envs/raw, envs/train,
            # envs/heldout and envs/manifest under it; the raw prefix would nest a
            # second envs/raw inside itself.
            "--output-uri",
            "{{config.envgen_root_uri}}",
            "--env-count",
            "{{config.env_count}}",
            # The retired sim2real-envgen-split.yaml drove sharding from a Kubernetes Job
            # completion index; a spec expresses it as a parallel group whose members set
            # `shard_index` in `params:`.
            "--shard-index",
            "{{config.shard_index}}",
            "--shard-count",
            "{{config.shard_count}}",
            "--seed",
            "{{config.envgen_seed}}",
            "--augmented-frames-uri",
            "{{config.augmented_frames_uri}}",
        ],
    ),
    "workbench.isaac_lab.capture_frames": ToolEntry(
        name="workbench.isaac_lab.capture_frames",
        description="Capture RGB frames from a headless Isaac Lab task and publish them.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.isaac_capture",
            "--task",
            "{{config.isaac_task}}",
            "--output-path",
            "{{config.scene_uri}}",
            "--max-steps",
            "{{config.capture_max_steps}}",
            "--max-frames",
            "{{config.capture_max_frames}}",
        ],
    ),
    "workbench.sim2real_envgen.actions": ToolEntry(
        name="workbench.sim2real_envgen.actions",
        description="Condition the train slice on a policy image's action space.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.sim2real_envgen",
            "actions",
            "--run-id",
            "{{run.id}}",
            "--output-uri",
            "{{config.envgen_root_uri}}",
            "--env-count",
            "{{config.env_count}}",
            "--seed",
            "{{config.envgen_seed}}",
            "--limit",
            "{{config.action_limit}}",
            # Recorded as provenance in actions-summary.json and salted into each env's seed;
            # the shipped generator does not run the image (see the spec's note).
            "--policy-image",
            "{{config.policy_image}}",
            # The split stage's own output, so actions conditions the same train slice rather
            # than recomputing a split of its own.
            "--train-envs-uri",
            "{{config.train_envs_uri}}",
            "--actions-uri",
            "{{config.actions_uri}}",
        ],
    ),
    "workbench.sim2real_envgen.split": ToolEntry(
        name="workbench.sim2real_envgen.split",
        description=(
            "Consume raw shards and create disjoint train, validation, and gold-heldout sets."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.sim2real_envgen",
            "split",
            "--run-id",
            "{{run.id}}",
            "--output-uri",
            "{{config.envgen_root_uri}}",
            "--env-count",
            "{{config.env_count}}",
            "--train-fraction",
            "{{config.train_fraction}}",
            "--shard-count",
            "{{config.shard_count}}",
            "--seed",
            "{{config.envgen_seed}}",
            "--augmented-frames-uri",
            "{{config.augmented_frames_uri}}",
        ],
    ),
    # Generic decision writer retained for non-Sim2Real workflow examples. The
    # canonical Sim2Real graph calls its stage adapter and does not use this
    # historical namespace.
    "workbench.sim2real.write_decision": ToolEntry(
        name="workbench.sim2real.write_decision",
        description="Write a real S3 workflow decision artifact.",
        argv_template=[
            "python3",
            "-c",
            (
                "from npa.orchestration.npa_workflow.decisions import write_decision; "
                "write_decision('{{config.decision_uri}}', '{{config.default_decision}}')"
            ),
        ],
    ),
    "workbench.byof.repo": ToolEntry(
        name="workbench.byof.repo",
        description=(
            "Build/push a BYOF OSS repo image via npa workbench byof and launch "
            "RL, datagen, container-verify, or solution smoke."
        ),
        argv_template=_BYOF_REPO_ARGV,
    ),
    "workbench.isaac_lab.byof_repo": ToolEntry(
        name="workbench.isaac_lab.byof_repo",
        description="Compatibility alias for workbench.byof.repo.",
        argv_template=_BYOF_REPO_ARGV,
    ),
    "workbench.rl.policy_train": ToolEntry(
        name="workbench.rl.policy_train",
        description=(
            "Train simulator RL policy checkpoint with workbench RL backend. "
            "Trainer hyper-parameters go through Isaac Lab's repeatable Hydra "
            "`--override KEY=VALUE`, which is what the CLI actually accepts."
        ),
        argv_template=[
            "npa",
            "workbench",
            "isaac-lab",
            "train",
            "--task",
            "{{config.task_name}}",
            "--steps",
            "{{config.train_steps}}",
            # `--num-envs` is the vectorized rollout batch dimension for on-policy
            # training; it is the real CLI flag for what the specs called
            # `batch_size` (there is no `--batch-size` option).
            "--num-envs",
            "{{config.num_envs}}",
            "--override",
            "agent.algorithm.learning_rate={{config.learning_rate}}",
            # The CLI names this `--data-path`, not `--input-path`.
            "--data-path",
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
            # The CLI names this `--num-episodes`, not `--episodes`.
            "--num-episodes",
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
    "workbench.scenario_gen.generate": ToolEntry(
        name="workbench.scenario_gen.generate",
        description=(
            "Mine a ranked adversarial scenario set that maximizes failures of a "
            "policy-under-test via a pluggable adversary backend (Isaac Lab RL "
            "intended; deterministic heuristic default, GPU-free)."
        ),
        argv_template=[
            "npa",
            "workbench",
            "scenario-gen",
            "generate",
            "--policy-uri",
            "{{config.policy_uri}}",
            "--input-path",
            "{{config.base_config_uri}}",
            "--output-path",
            "{{config.adversarial_set_uri}}",
            "--task",
            "{{config.task_name}}",
            "--num-scenarios",
            "{{config.num_scenarios}}",
            "--adversary-steps",
            "{{config.adversary_steps}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.scenario_gen.rank": ToolEntry(
        name="workbench.scenario_gen.rank",
        description="Score/rank adversarial scenarios by failure severity + diversity.",
        argv_template=[
            "npa",
            "workbench",
            "scenario-gen",
            "rank",
            "--input-path",
            "{{config.adversarial_set_uri}}manifest.json",
            "--output-path",
            "{{config.ranked_set_uri}}",
            "--top-k",
            "{{config.rank_top_k}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.scenario_gen.write_hardening_decision": ToolEntry(
        name="workbench.scenario_gen.write_hardening_decision",
        description="Write promote/loop decision from the configured failure-rate threshold.",
        argv_template=[
            "python3",
            "-c",
            (
                "from npa.orchestration.npa_workflow.decisions import write_decision;"
                "threshold=float('{{config.failure_rate_threshold}}');"
                "decision='promote_checkpoint' if threshold >= 0.5 else 'loop_back';"
                "write_decision('{{config.decision_uri}}', decision)"
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
    "workbench.dataset.ingest": ToolEntry(
        name="workbench.dataset.ingest",
        description=(
            "Ingest raw sensor data, validate against a sensor schema, normalize "
            "to canonical records, and register a versioned dataset-of-record manifest."
        ),
        argv_template=[
            "npa",
            "workbench",
            "dataset",
            "ingest",
            "--input-path",
            "{{config.raw_sensor_uri}}",
            "--output-path",
            "{{config.dataset_root_uri}}",
            "--dataset-id",
            "{{config.dataset_id}}",
            "--version",
            "{{config.dataset_version}}",
            "--source",
            "{{config.dataset_source}}",
            "--workflow-run",
            "{{run.id}}",
            # Populate the query index as records land. Without this the `query` stage has
            # nothing to find, which is what a live run showed the moment the LanceDB service
            # became reachable at all (EVIDENCE.md §R41).
            "--lancedb-endpoint",
            "{{config.lancedb_endpoint}}",
            "--lance-uri",
            "{{config.lance_uri}}",
        ],
    ),
    "workbench.dataset.validate": ToolEntry(
        name="workbench.dataset.validate",
        description="Validate a dataset manifest against schema + quality thresholds.",
        argv_template=[
            "npa",
            "workbench",
            "dataset",
            "validate",
            "--input-path",
            "{{config.manifest_uri}}",
            "--output-path",
            "{{config.validation_uri}}",
            "--completeness-min",
            "{{config.completeness_min}}",
            "--max-corruption-rate",
            "{{config.max_corruption_rate}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.dataset.curate": ToolEntry(
        name="workbench.dataset.curate",
        description="Slice a dataset version by event/location/quality with lineage.",
        argv_template=[
            "npa",
            "workbench",
            "dataset",
            "curate",
            "--input-path",
            "{{config.manifest_uri}}",
            "--output-path",
            "{{config.curated_root_uri}}",
            "--event",
            "{{config.event_of_interest}}",
            "--location",
            "{{config.location_of_interest}}",
            "--quality-metric",
            "{{config.quality_metric}}",
            "--min-quality",
            "{{config.min_quality}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.dataset.query": ToolEntry(
        name="workbench.dataset.query",
        description="Query dataset records by event/location/quality facets (LanceDB-backed).",
        argv_template=[
            "npa",
            "workbench",
            "dataset",
            "query",
            "--input-path",
            "{{config.curated_manifest_uri}}",
            "--event",
            "{{config.event_of_interest}}",
            "--location",
            "{{config.location_of_interest}}",
            "--lancedb-endpoint",
            "{{config.lancedb_endpoint}}",
            # Ingest writes one table per dataset id; a query that does not name it reads the
            # service's default and finds nothing (EVIDENCE.md §R41).
            "--lance-table",
            "{{config.dataset_id}}",
            "--lance-uri",
            "{{config.lance_uri}}",
        ],
    ),
    "workbench.dataset.write_quality_decision": ToolEntry(
        name="workbench.dataset.write_quality_decision",
        description="Write accept/reject decision from a validation quality gate.",
        argv_template=[
            "python3",
            "-c",
            (
                "from npa.orchestration.npa_workflow.decisions import write_decision;"
                "threshold=float('{{config.quality_gate}}');"
                "decision='promote_checkpoint' if threshold >= 0.5 else 'loop_back';"
                "write_decision('{{config.decision_uri}}', decision)"
            ),
        ],
    ),
    "workbench.dataset.report_rejection": ToolEntry(
        name="workbench.dataset.report_rejection",
        description="Write terminal rejection report when a dataset breaches the quality gate.",
        argv_template=[
            "python3",
            "-c",
            (
                "import json;from pathlib import Path;"
                "payload={'validation_uri':'{{config.validation_uri}}','decision_uri':'{{config.decision_uri}}',"
                "'status':'rejected'};"
                "Path('/tmp/npa-dataset-rejection.json').write_text(json.dumps(payload));"
                "print(json.dumps(payload))"
            ),
        ],
    ),
    "workbench.insights.record": ToolEntry(
        name="workbench.insights.record",
        description=(
            "Record metric emissions + lineage edges (from an upstream metrics "
            "JSON) into the append-only insights store keyed by run id."
        ),
        argv_template=[
            "npa",
            "workbench",
            "insights",
            "record",
            "--input-path",
            "{{config.metrics_input_uri}}",
            "--output-path",
            "{{config.insights_store_uri}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.foxglove.convert": ToolEntry(
        name="workbench.foxglove.convert",
        description=(
            "Pack a run's frames, JSON metrics, and logs into a real MCAP recording "
            "(Foxglove well-known schemas) for the embedded Foxglove viewer."
        ),
        argv_template=[
            "npa",
            "workbench",
            "foxglove",
            "convert-run",
            "--input-path",
            "{{config.run_artifacts_path}}",
            "--output-path",
            "{{config.mcap_output_path}}",
            "--run-id",
            "{{run.id}}",
            "--fps",
            "{{config.mcap_fps}}",
            "--output",
            "json",
        ],
    ),
    "workbench.insights.ingest_run": ToolEntry(
        name="workbench.insights.ingest_run",
        description=(
            "Non-invasively scan an S3 run prefix for known tool manifests/reports "
            "and extract their metrics + provenance into the insights store."
        ),
        argv_template=[
            "npa",
            "workbench",
            "insights",
            "ingest-run",
            "--input-path",
            "{{config.run_prefix_uri}}",
            "--output-path",
            "{{config.insights_store_uri}}",
            "--workflow",
            "{{config.workflow_name}}",
            "--workflow-run",
            "{{run.id}}",
        ],
    ),
    "workbench.insights.compare": ToolEntry(
        name="workbench.insights.compare",
        description="Compare a metric set between two runs; flag regressed/improved.",
        argv_template=[
            "npa",
            "workbench",
            "insights",
            "compare",
            "--input-path",
            "{{config.insights_store_uri}}",
            "--base-run",
            "{{config.base_run}}",
            "--candidate-run",
            "{{config.candidate_run}}",
            "--output-path",
            "{{config.comparison_uri}}",
        ],
    ),
    "workbench.insights.dashboard": ToolEntry(
        name="workbench.insights.dashboard",
        description="Emit a dashboard rollup JSON + self-contained static HTML report.",
        argv_template=[
            "npa",
            "workbench",
            "insights",
            "dashboard",
            "--input-path",
            "{{config.insights_store_uri}}",
            "--output-path",
            "{{config.dashboard_uri}}",
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
            "--synthetic",
            "{{config.synthetic_rows}}",
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
                '--udf "$udf" '
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
                "--source-table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--service --endpoint {{config.lancedb_endpoint}}; "
                "npa workbench lancedb create-mv "
                "--name {{config.nighttime_view}} "
                "--filter \"timeofday = 'night' AND has_person = true AND split = 'train'\" "
                "--source-table {{config.lance_table}} "
                "--lance-uri {{config.lance_uri}} "
                "--service --endpoint {{config.lancedb_endpoint}}; "
                "npa workbench lancedb create-mv "
                "--name {{config.distant_view}} "
                "--filter \"has_person = true AND person_bbox_area_pct < 0.01 AND split = 'train'\" "
                "--source-table {{config.lance_table}} "
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
            "--label-map",
            "{{config.detection_label_map}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
            # The retired template polled /status in bash until the run finished;
            # without the wait, eval would run against a checkpoint that does not
            # exist yet.
            "--wait",
            "--poll-seconds",
            "{{config.train_poll_seconds}}",
            "--timeout-seconds",
            "{{config.train_timeout_seconds}}",
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
            "--label-map",
            "{{config.detection_label_map}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
            # The retired template polled /status in bash until the run finished;
            # without the wait, eval would run against a checkpoint that does not
            # exist yet.
            "--wait",
            "--poll-seconds",
            "{{config.train_poll_seconds}}",
            "--timeout-seconds",
            "{{config.train_timeout_seconds}}",
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
            "--label-map",
            "{{config.detection_label_map}}",
            "--service",
            "--endpoint",
            "{{config.detection_endpoint}}",
            # The retired template polled /status in bash until the run finished;
            # without the wait, eval would run against a checkpoint that does not
            # exist yet.
            "--wait",
            "--poll-seconds",
            "{{config.train_poll_seconds}}",
            "--timeout-seconds",
            "{{config.train_timeout_seconds}}",
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
            # The retired template resolved the checkpoint from /runs and published
            # <output-uri>/metrics.json itself; --checkpoint-uri above is the
            # training output prefix to search.
            "--discover-checkpoint",
            "--write-canonical-metrics",
            # Eval must read labels the way training wrote them; BDD100K categories are
            # strings and one of them is literally "train" (EVIDENCE.md §R46).
            "--label-map",
            "{{config.detection_label_map}}",
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
            # The retired template resolved the checkpoint from /runs and published
            # <output-uri>/metrics.json itself; --checkpoint-uri above is the
            # training output prefix to search.
            "--discover-checkpoint",
            "--write-canonical-metrics",
            # Eval must read labels the way training wrote them; BDD100K categories are
            # strings and one of them is literally "train" (EVIDENCE.md §R46).
            "--label-map",
            "{{config.detection_label_map}}",
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
            # The retired template resolved the checkpoint from /runs and published
            # <output-uri>/metrics.json itself; --checkpoint-uri above is the
            # training output prefix to search.
            "--discover-checkpoint",
            "--write-canonical-metrics",
            # Eval must read labels the way training wrote them; BDD100K categories are
            # strings and one of them is literally "train" (EVIDENCE.md §R46).
            "--label-map",
            "{{config.detection_label_map}}",
        ],
    ),
    "workbench.fiftyone.launch_app": ToolEntry(
        name="workbench.fiftyone.launch_app",
        description="Launch FiftyOne App for pipeline review (workflow stub).",
        argv_template=[
            "echo",
            "fiftyone review run {{run.id}} lance {{config.lance_uri}}",
        ],
        stub=True,
    ),
    "workbench.fiftyone.curate_augmented": ToolEntry(
        name="workbench.fiftyone.curate_augmented",
        description=(
            "Run real FiftyOne Brain uniqueness, similarity, duplicate detection, "
            "and PCA review over PAIDF variants, merging the Cosmos Curator report; "
            "fail closed if the FiftyOne engine does not complete."
        ),
        argv_template=[
            "npa",
            "workbench",
            "fiftyone",
            "curate-augmented",
            "--augment-uri",
            "{{config.augment_uri}}",
            "--report-uri",
            "{{config.curation_report_uri}}",
            "--curator-report-uri",
            "{{config.curator_report_uri}}",
            "--dedup-threshold",
            "{{config.fiftyone_dedup_threshold}}",
            "--require-fiftyone",
            "--output",
            "json",
        ],
    ),
    "workbench.token_factory.caption": ToolEntry(
        name="workbench.token_factory.caption",
        description="Caption images with Nebius Token Factory (zero-GPU).",
        argv_template=[
            "npa",
            "workbench",
            "token-factory",
            "caption",
            "--input-path",
            "{{config.images_uri}}",
            "--output-path",
            "{{config.captions_uri}}",
            "--model",
            "{{config.caption_model}}",
            "--max-images",
            "{{config.max_images}}",
            "--max-tokens",
            "{{config.max_tokens}}",
            "--output",
            "json",
        ],
    ),
    "workbench.token_factory.generate": ToolEntry(
        name="workbench.token_factory.generate",
        description="Generate text completions with Nebius Token Factory (zero-GPU).",
        argv_template=[
            "npa",
            "workbench",
            "token-factory",
            "generate",
            "--input-path",
            "{{config.prompts_uri}}",
            "--output-path",
            "{{config.generations_uri}}",
            "--model",
            "{{config.generate_model}}",
            "--max-tokens",
            "{{config.max_tokens}}",
            "--output",
            "json",
        ],
    ),
    "workbench.vlm_eval.benchmark": ToolEntry(
        name="workbench.vlm_eval.benchmark",
        description="Benchmark VLM backends against a fixture or rollout set.",
        argv_template=[
            "npa",
            "workbench",
            "vlm-eval",
            "benchmark",
            "--dataset",
            "{{config.benchmark_dataset}}",
            "--output",
            "{{config.benchmark_output}}",
            "--backend",
            "{{config.vlm_backend}}",
            "--thresholds",
            "{{config.thresholds}}",
            "--rubrics",
            "{{config.rubrics}}",
            "--models",
            "{{config.vlm_models}}",
            "--format",
            "json",
        ],
    ),
    "workbench.lerobot.policy_train": ToolEntry(
        name="workbench.lerobot.policy_train",
        description=(
            "Train a LeRobot policy IN the stage's own pod, using the vendor image's LeRobot."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workbench.lerobot.policy_container",
            "train",
            "--dataset-repo-id",
            "{{config.lerobot_dataset}}",
            "--output-dir",
            "{{config.lerobot_output_dir}}",
            "--steps",
            "{{config.train_steps}}",
            "--policy-type",
            "{{config.policy_type}}",
            "--batch-size",
            "{{config.train_batch_size}}",
            "--device",
            "{{config.policy_device}}",
            # The checkpoint AND the run's textual artifacts (configs, logs, metrics) go to
            # the same prefix, so a downstream stage can read the run. The retired template
            # did the second half in a trailing inline-python block.
            "--checkpoint-s3-uri",
            "{{config.artifacts_uri}}",
            "--artifacts-s3-uri",
            "{{config.artifacts_uri}}",
        ],
    ),
    "workbench.token_factory.triage": ToolEntry(
        name="workbench.token_factory.triage",
        description=(
            "Digest a run's textual artifacts and have a hosted text model write a triage report."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.token_factory_triage",
            "run",
            "--artifacts-uri",
            "{{config.artifacts_uri}}",
            "--triage-uri",
            "{{config.triage_uri}}",
            "--job-name",
            "{{config.triage_job_name}}",
            "--model",
            "{{config.triage_model}}",
            "--max-tokens",
            "{{config.triage_max_tokens}}",
        ],
    ),
    "workbench.lerobot.policy_rollout": ToolEntry(
        name="workbench.lerobot.policy_rollout",
        description=(
            "Roll out a LeRobot policy IN the stage's own pod and publish the rendered episodes."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workbench.lerobot.policy_container",
            "eval",
            # A public Hugging Face policy id, an s3:// prefix or a local path; a stage's pod
            # starts empty, so anything remote is materialised first.
            "--checkpoint-path",
            "{{config.policy_checkpoint}}",
            "--output-dir",
            "{{config.rollout_output_dir}}",
            "--env-type",
            "{{config.rollout_env}}",
            "--episodes",
            "{{config.rollout_episodes}}",
            "--device",
            "{{config.policy_device}}",
            # The judge stage reads what this stage rendered; the retired template did the
            # upload in a trailing inline-python block.
            "--rollouts-s3-uri",
            "{{config.rollouts_uri}}",
        ],
    ),
    "workbench.lerobot.eval": ToolEntry(
        name="workbench.lerobot.eval",
        description="Evaluate a LeRobot policy checkpoint.",
        argv_template=[
            "npa",
            "workbench",
            "lerobot",
            "eval",
            "--checkpoint",
            "{{config.checkpoint_uri}}",
            "--env",
            "{{config.env}}",
            "--episodes",
            "{{config.episodes}}",
            "--output-path",
            "{{config.eval_uri}}",
            "--output",
            "json",
        ],
    ),
    "workbench.retargeting.run": ToolEntry(
        name="workbench.retargeting.run",
        description="Retarget source motion into a SONIC embodiment schema.",
        argv_template=[
            "npa",
            "workbench",
            "sonic",
            "retargeting",
            "run",
            "--input-path",
            "{{config.motion_uri}}",
            "--output-path",
            "{{config.retargeted_uri}}",
            "--embodiment",
            "{{config.embodiment}}",
            "--source-format",
            "{{config.source_format}}",
        ],
    ),
    "workbench.mjlab.eval": ToolEntry(
        name="workbench.mjlab.eval",
        description="Score a SONIC locomotion checkpoint with MJLab metrics.",
        argv_template=[
            "npa",
            "workbench",
            "mjlab",
            "eval",
            "--input-path",
            "{{config.motion_uri}}",
            "--checkpoint",
            "{{config.checkpoint_uri}}",
            "--output-path",
            "{{config.mjlab_uri}}",
            "--suite",
            "{{config.suite}}",
            "--embodiment",
            "{{config.embodiment}}",
            "--episodes",
            "{{config.episodes}}",
            "--output",
            "json",
        ],
    ),
    "workbench.sonic.train": ToolEntry(
        name="workbench.sonic.train",
        description="Train or smoke-validate a SONIC locomotion policy.",
        argv_template=[
            "npa",
            "workbench",
            "sonic",
            "train",
            "--runtime",
            "{{config.sonic_runtime}}",
            # NVIDIA's terms are the operator's to accept. `--runtime in-job` downloads Isaac
            # Sim and Isaac Lab onto the machine, and the SONIC entrypoint refuses until this
            # is set (live job 327, EVIDENCE.md §R47). A spec is where a reviewer can see it.
            "--accept-nvidia-eula",
            "{{config.sonic_accept_nvidia_eula}}",
            "--checkpoint",
            "{{config.checkpoint_uri}}",
            "--data-path",
            "{{config.data_uri}}",
            "--output-path",
            "{{config.training_uri}}",
            "--max-iterations",
            "{{config.train_iterations}}",
            "--output",
            "json",
        ],
    ),
    "workbench.groot.finetune": ToolEntry(
        name="workbench.groot.finetune",
        description=(
            "Fine-tune NVIDIA GR00T N1.7 in the stage's own GPU container and "
            "publish the vendor checkpoints plus an NPA provenance manifest."
        ),
        argv_template=[
            "npa",
            "workbench",
            "groot",
            "finetune",
            "--runtime",
            "local",
            "--data-path",
            "{{config.data_uri}}",
            "--checkpoint-s3-uri",
            "{{config.candidate_checkpoint_uri}}",
            "--base-model",
            "{{config.base_model}}",
            "--robot-embodiment",
            "{{config.robot_embodiment}}",
            "--num-gpus",
            "{{config.gpu_count}}",
            "--nccl-transport",
            "{{config.nccl_transport}}",
            "--global-batch-size",
            "{{config.global_batch_size}}",
            "--per-device-batch-size",
            "{{config.per_device_batch_size}}",
            "--gradient-accumulation-steps",
            "{{config.gradient_accumulation_steps}}",
            "--dataloader-num-workers",
            "{{config.dataloader_num_workers}}",
            "--logging-steps",
            "{{config.logging_steps}}",
            "--max-steps",
            "{{config.max_steps}}",
            "--save-steps",
            "{{config.save_steps}}",
            "--save-total-limit",
            "{{config.save_total_limit}}",
            "--save-only-model",
            "--override",
            "episode-sampling-rate=1.0",
            "--override",
            "state-dropout-prob=0.0",
            "--override",
            "tune-projector=true",
            "--override",
            "tune-diffusion-model=true",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workflow.groot.resolve_trained_checkpoint": ToolEntry(
        name="workflow.groot.resolve_trained_checkpoint",
        description=(
            "Hash and freeze the exact completed parameterized multi-GPU "
            "trainer output."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_task_performance",
            "resolve-trained-checkpoint",
            "--training-manifest-uri",
            "{{config.training_manifest_uri}}",
            "--split-manifest-uri",
            "{{config.split_manifest_uri}}",
            "--checkpoint-uri",
            "{{config.candidate_checkpoint_uri}}",
            "--baseline-checkpoint-uri",
            "{{config.baseline_checkpoint_uri}}",
            "--output-uri",
            "{{config.candidate_checkpoint_ref_uri}}",
            "--run-id",
            "{{run.id}}",
            "--expected-gpu-count",
            "{{config.gpu_count}}",
            "--expected-max-steps",
            "{{config.max_steps}}",
            "--expected-save-steps",
            "{{config.save_steps}}",
            "--expected-save-total-limit",
            "{{config.save_total_limit}}",
        ],
    ),
    "workflow.groot.preflight_rigor": ToolEntry(
        name="workflow.groot.preflight_rigor",
        description="Fail before GPU scheduling when the declared learning contract is incoherent.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "preflight-rigor",
            "--output-uri",
            "{{config.rigor_preflight_uri}}",
            "--run-id",
            "{{run.id}}",
            "--gpu-type",
            "{{config.gpu_type}}",
            "--gpu-count",
            "{{config.gpu_count}}",
            "--global-batch-size",
            "{{config.global_batch_size}}",
            "--per-device-batch-size",
            "{{config.per_device_batch_size}}",
            "--gradient-accumulation-steps",
            "{{config.gradient_accumulation_steps}}",
            "--train-episodes",
            "{{config.train_episodes}}",
            "--validation-episodes",
            "{{config.heldout_episodes}}",
            "--final-episodes",
            "{{config.final_episodes}}",
            "--max-steps",
            "{{config.max_steps}}",
            "--save-steps",
            "{{config.save_steps}}",
            "--save-total-limit",
            "{{config.save_total_limit}}",
            "--minimum-epochs",
            "{{config.minimum_epochs}}",
        ],
    ),
    "workflow.groot.prepare_split": ToolEntry(
        name="workflow.groot.prepare_split",
        description="Materialize and hash a deterministic leakage-free episode split.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "prepare-split",
            "--source-uri",
            "{{config.source_data_uri}}",
            "--train-uri",
            "{{config.train_data_uri}}",
            "--heldout-uri",
            "{{config.heldout_data_uri}}",
            "--output-uri",
            "{{config.split_manifest_uri}}",
            "--run-id",
            "{{run.id}}",
            "--train-episodes",
            "{{config.train_episodes}}",
            "--heldout-episodes",
            "{{config.heldout_episodes}}",
            "--final-uri",
            "{{config.final_data_uri}}",
            "--final-episodes",
            "{{config.final_episodes}}",
            "--seed",
            "{{config.split_seed}}",
            "--global-batch-size",
            "{{config.global_batch_size}}",
            "--max-steps",
            "{{config.max_steps}}",
            "--minimum-epochs",
            "{{config.minimum_epochs}}",
            "--minimum-effective-global-batch",
            "{{config.minimum_effective_global_batch}}",
            "--gpu-count",
            "{{config.gpu_count}}",
            "--per-device-batch-size",
            "{{config.per_device_batch_size}}",
            "--gradient-accumulation-steps",
            "{{config.gradient_accumulation_steps}}",
            "--action-representation",
            "absolute",
        ],
    ),
    "workbench.groot.baseline_eval": ToolEntry(
        name="workbench.groot.baseline_eval",
        description=(
            "Initialize the custom embodiment from the pinned base checkpoint with "
            "train-only statistics and run real held-out Gr00tPolicy forwards."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "baseline-eval",
            "--split-manifest-uri",
            "{{config.split_manifest_uri}}",
            "--output-uri",
            "{{config.offline_baseline_eval_uri}}",
            "--arrays-uri",
            "{{config.offline_baseline_arrays_uri}}",
            "--baseline-checkpoint-uri",
            "{{config.baseline_checkpoint_uri}}",
            "--base-model",
            "{{config.base_model}}",
            "--run-id",
            "{{run.id}}",
            "--action-horizon",
            "{{config.action_horizon}}",
            "--evaluation-repeats",
            "{{config.evaluation_repeats}}",
        ],
    ),
    "workbench.groot.posttrain_eval": ToolEntry(
        name="workbench.groot.posttrain_eval",
        description="Run the identical real held-out evaluation on the trained checkpoint.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "posttrain-eval",
            "--split-manifest-uri",
            "{{config.split_manifest_uri}}",
            "--checkpoint-ref-uri",
            "{{config.candidate_checkpoint_ref_uri}}",
            "--output-uri",
            "{{config.offline_candidate_eval_uri}}",
            "--arrays-uri",
            "{{config.offline_candidate_arrays_uri}}",
            "--run-id",
            "{{run.id}}",
            "--action-horizon",
            "{{config.action_horizon}}",
            "--evaluation-repeats",
            "{{config.evaluation_repeats}}",
        ],
    ),
    "workflow.groot.compare_learning": ToolEntry(
        name="workflow.groot.compare_learning",
        description="Compare aligned real evaluations, expose regressions, and gate improvement.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "compare-learning",
            "--split-manifest-uri",
            "{{config.split_manifest_uri}}",
            "--baseline-uri",
            "{{config.offline_baseline_eval_uri}}",
            "--posttrain-uri",
            "{{config.offline_candidate_eval_uri}}",
            "--training-manifest-uri",
            "{{config.training_manifest_uri}}",
            "--output-uri",
            "{{config.learning_report_uri}}",
            "--video-uri",
            "{{config.offline_comparison_video_uri}}",
            "--run-id",
            "{{run.id}}",
            "--minimum-relative-improvement",
            "{{config.minimum_relative_improvement}}",
            "--minimum-skill-score",
            "{{config.minimum_skill_score}}",
            "--repeat-noise-multiple",
            "{{config.repeat_noise_multiple}}",
            "--max-dimension-regression",
            "{{config.max_dimension_regression}}",
            "--loss-decrease-tolerance",
            "{{config.loss_decrease_tolerance}}",
        ],
    ),
    "workflow.groot.emit_learning_mcap": ToolEntry(
        name="workflow.groot.emit_learning_mcap",
        description="Emit synchronized camera, expert/predicted action, error, and metric MCAP.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "emit-mcap",
            "--report-uri",
            "{{config.learning_report_uri}}",
            "--output-uri",
            "{{config.offline_mcap_uri}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workflow.groot.emit_learning_rrd": ToolEntry(
        name="workflow.groot.emit_learning_rrd",
        description="Emit native Rerun learning replay with a camera/action/error blueprint.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "emit-rrd",
            "--report-uri",
            "{{config.learning_report_uri}}",
            "--output-uri",
            "{{config.offline_rrd_uri}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workflow.groot.publish_learning": ToolEntry(
        name="workflow.groot.publish_learning",
        description="Validate, hash, and index the complete factual learning output set.",
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "publish",
            "--report-uri",
            "{{config.learning_report_uri}}",
            "--mcap-uri",
            "{{config.mcap_uri}}",
            "--rrd-uri",
            "{{config.rrd_uri}}",
            "--video-uri",
            "{{config.comparison_video_uri}}",
            "--workflow-uri",
            "{{config.workflow_uri}}",
            "--output-uri",
            "{{config.publish_manifest_uri}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workflow.groot.verify_agent_ui": ToolEntry(
        name="workflow.groot.verify_agent_ui",
        description=(
            "Fail unless the deployed agent discovers this exact run, inventories "
            "its report, loads the RRD and MCAP viewers, and serves byte ranges."
        ),
        argv_template=[
            "python3",
            "-m",
            "npa.workflows.groot_learning",
            "verify-agent-ui",
            "--agent-url",
            "{{config.agent_url}}",
            "--report-uri",
            "{{config.learning_report_uri}}",
            "--rrd-uri",
            "{{config.rrd_uri}}",
            "--mcap-uri",
            "{{config.mcap_uri}}",
            "--output-uri",
            "{{config.agent_ui_report_uri}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workbench.sonic.eval": ToolEntry(
        name="workbench.sonic.eval",
        description="Evaluate an exported SONIC ONNX locomotion policy.",
        argv_template=[
            "npa",
            "workbench",
            "sonic",
            "eval",
            "--onnx",
            "{{config.onnx_uri}}",
            "--episodes",
            "{{config.episodes}}",
            "--env",
            "{{config.env}}",
            # `--output` on this command is the RESULT PATH (output_path: str), not a
            # format; `--output-format` is the format. Passing "json" to --output made
            # the tool write the eval result to a relative `json/` directory inside the
            # pod, so the spec's declared eval.json artifact never appeared (found live:
            # runs npa-wf-gpu-sonic-eval-87a704ad / npa-wf-multi-sonic-export-eval-...).
            "--output",
            "{{config.eval_uri}}",
            "--output-format",
            "json",
        ],
    ),
    "workbench.sonic.export": ToolEntry(
        name="workbench.sonic.export",
        description="Export a SONIC locomotion checkpoint to ONNX.",
        argv_template=[
            "npa",
            "workbench",
            "sonic",
            "export",
            "--checkpoint",
            "{{config.checkpoint_uri}}",
            "--output",
            "{{config.onnx_uri}}",
        ],
    ),
    "workbench.cosmos3.generate": ToolEntry(
        name="workbench.cosmos3.generate",
        description=(
            "Generate an image or video with the Cosmos 3 omni model (real "
            "inference in the npa-cosmos3 image; gated weights download at "
            "runtime with the operator's HF token)."
        ),
        argv_template=[
            "npa",
            "workbench",
            "cosmos3",
            "generate",
            "--mode",
            "{{config.cosmos3_mode}}",
            "--prompt",
            "{{config.prompt}}",
            "--output-path",
            "{{config.output_uri}}",
            "--checkpoint",
            "{{config.cosmos3_checkpoint}}",
            "--run-id",
            "{{run.id}}",
        ],
    ),
    "workbench.cosmos3.reason": ToolEntry(
        name="workbench.cosmos3.reason",
        description="Build a Cosmos3 reason-stage manifest over input frames.",
        argv_template=[
            "npa",
            "workbench",
            "cosmos3",
            "reason",
            "--input-uri",
            "{{config.scene_uri}}",
            "--output-uri",
            "{{config.reason_uri}}",
            "--model",
            "{{config.cosmos3_model}}",
            "--run-id",
            "{{run.id}}",
        ],
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
