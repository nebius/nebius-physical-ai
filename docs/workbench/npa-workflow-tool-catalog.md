# NPA workflow tool catalog (v0.0.1)

Workbench tools referenced by `toolRef` in NPA workflow specs. Each tool is
invoked as a container command; artifacts pass via S3 URIs in `config`.

Source of truth: `npa/src/npa/orchestration/npa_workflow/catalog.py`.
This table must list every `TOOL_CATALOG` key (enforced by
`npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py`).

| toolRef | CLI / module | Typical inputs | Typical outputs | Stub? |
| --- | --- | --- | --- | --- |
| `infra.soperator.deploy` | `npa soperator deploy` | `config.soperator_spec` | cluster deploy JSON | no |
| `workbench.vlm_eval.run` | `npa workbench vlm-eval run` | `config.rollouts_uri` | `config.scores_uri` | no |
| `workbench.vlm_eval.benchmark` | `npa workbench vlm-eval benchmark` | `config.benchmark_dataset` | `config.benchmark_output` | no |
| `workbench.token_factory.reason` | `npa workbench token-factory reason` | `config.scene_uri` | `config.plan_uri` | no |
| `workbench.token_factory.caption` | `npa workbench token-factory caption` | `config.images_uri` | `config.captions_uri` | no |
| `workbench.token_factory.generate` | `npa workbench token-factory generate` | `config.prompts_uri` | `config.generations_uri` | no |
| `workbench.cosmos2.transfer` | `npa workbench cosmos2 transfer` | `config.trigger_uri` | `config.augment_uri` | no |
| `workbench.cosmos3.reason` | `npa workbench cosmos3 reason` | `config.scene_uri` | `config.reason_uri` | no |
| `workbench.lerobot.eval` | `npa workbench lerobot eval` | `config.checkpoint_uri`, `config.env` | `config.eval_uri` | no |
| `workbench.retargeting.run` | `npa workbench sonic retargeting run` | `config.motion_uri` | `config.retargeted_uri` | no |
| `workbench.mjlab.eval` | `npa workbench mjlab eval` | `config.motion_uri`, `config.checkpoint_uri` | `config.mjlab_uri` | no |
| `workbench.sonic.train` | `npa workbench sonic train` | `config.checkpoint_uri`, `config.data_uri` | training checkpoint | no |
| `workbench.sonic.export` | `npa workbench sonic export` | `config.checkpoint_uri` | `config.onnx_uri` | no |
| `workbench.sonic.eval` | `npa workbench sonic eval` | `config.onnx_uri` | eval report | no |
| `workbench.sim2real_envgen.raw_shard` | `python -m npa.workflows.sim2real_envgen raw-shard` | `config.raw_envs_uri`, `config.env_count` | raw env manifest on S3 | no |
| `workbench.sim2real.policy_rollouts` | workflow stub (`echo`) | `config.rollouts_uri` | rollout prefix on S3 | yes |
| `workbench.sim2real.heldout_eval` | workflow stub (`echo`) | — | `config.heldout_report_uri` | yes |
| `workbench.sim2real.write_decision` | demo decision writer | `config.decision_uri`, `config.default_decision` | threshold decision JSON | no |
| `workbench.sim2real.finalize` | workflow stub (`echo`) | `config.finalize_report_uri` | final report URI | yes |
| `workbench.byof.repo` | `npa workbench byof run` | `config.repo_url`, `config.repo_ref`, `config.base_profile`, optional `config.build_command` / `config.smoke_command`; registry candidates also set `config.solution_name`, `config.capability_name`, `config.smoke_artifact_name` | BYOF summary, dataset/checkpoint artifacts, solution smoke artifact | no |
| `workbench.isaac_lab.byof_repo` | alias → `workbench.byof.repo` | same as BYOF | same as BYOF | no |
| `workbench.data_transform.rollout_contract` | rollout contract adapter | rollout manifest URI | normalized rollout manifest | no |
| `workbench.data_transform.improvement_summary` | cross-region summary adapter | heldout/report URIs | improvement summary | no |
| `workbench.rl.policy_train` | `npa workbench isaac-lab train` | `config.task_name`, training dataset URI | policy checkpoint | no |
| `workbench.rl.evaluate_policy` | `npa workbench isaac-lab eval` | checkpoint URI, eval episodes | eval report | no |
| `workbench.rl.write_success_decision` | RL decision writer | eval report URI, `config.success_threshold` | training decision JSON | no |
| `workbench.rl.publish_policy` | policy release writer | checkpoint + decision URIs | release manifest | no |
| `workbench.rl.report_failure` | failure report writer | eval + decision URIs | failure report | no |
| `workbench.scenario_gen.generate` | `npa workbench scenario-gen generate` | `config.policy_uri`, `config.base_config_uri` | `config.adversarial_set_uri` (adversarial set manifest) | no |
| `workbench.scenario_gen.rank` | `npa workbench scenario-gen rank` | `config.adversarial_set_uri` | `config.ranked_set_uri` | no |
| `workbench.scenario_gen.write_hardening_decision` | hardening decision writer | `config.failure_rate_threshold`, `config.decision_uri` | hardening decision JSON | no |
| `workbench.dataset.ingest` | `npa workbench dataset ingest` | `config.raw_sensor_uri`, `config.dataset_id` | versioned dataset manifest (`npa.dataset.manifest.v1`) | no |
| `workbench.dataset.validate` | `npa workbench dataset validate` | `config.manifest_uri` | `npa.dataset.validation_report.v1` | no |
| `workbench.dataset.curate` | `npa workbench dataset curate` | `config.manifest_uri`, `config.event_of_interest` | curated dataset version manifest | no |
| `workbench.dataset.query` | `npa workbench dataset query` | `config.curated_manifest_uri` | matching records (LanceDB-backed) | no |
| `workbench.dataset.write_quality_decision` | dataset quality-gate decision writer | `config.quality_gate`, `config.decision_uri` | accept/reject decision JSON | no |
| `workbench.dataset.report_rejection` | dataset rejection report writer | `config.validation_uri`, `config.decision_uri` | rejection report | no |
| `workbench.lancedb.import_bdd100k` | `npa workbench lancedb import-bdd100k --service` | `config.source_uri`, `config.lance_uri` | LanceDB table | no |
| `workbench.lancedb.backfill_cpu_bundle` | five CPU UDF backfills | `config.lance_table`, `config.lance_uri` | enriched table | no |
| `workbench.lancedb.backfill_clip` | CLIP embedding UDF | `config.lance_uri` | `clip_embedding` column | no |
| `workbench.lancedb.create_failure_views` | three materialized views | `config.rider_view`, … | failure-mode views | no |
| `workbench.detection_training.train_*` | `npa workbench detection-training train --service` | view + output URIs | checkpoints | no |
| `workbench.detection_training.eval_*` | `npa workbench detection-training eval --service` | checkpoint + view | metrics JSON | no |
| `workbench.fiftyone.launch_app` | FiftyOne review hook | `config.lance_uri` | review session | yes |

Creative mashup example: `tokenfactory-cosmos-gate.yaml` (reason → augment → VLM gate loop).

OSS onboarding ladder (BYOF → workflow → first-class tool):
`docs/architecture/oss-onboarding-ladder.md`.

Add new entries in `npa/src/npa/orchestration/npa_workflow/catalog.py` when
exposing a tool to workflow specs, then update this table.

## Tokens

| Token | Meaning |
| --- | --- |
| `{{config.*}}` | Value from spec `config` block (after run-id expansion) |
| `{{run.id}}` | Run identifier passed to plan/run commands |
| `{{run.prefix}}` | Default `"{metadata.name}/{run.id}"` or `config.prefix` |
| `{{state.NAME.uri}}` | Primary output URI recorded after state `NAME` executes |

See `docs/workbench/npa-workflow-guide.md` for the full authoring guide.

## Predicates

| Name | True when |
| --- | --- |
| `promote_checkpoint` | Last decision is promote |
| `loop_back` | Last decision is loop-back |
