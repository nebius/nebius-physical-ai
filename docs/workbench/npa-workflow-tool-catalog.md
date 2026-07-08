# NPA workflow tool catalog (v0.0.1)

Workbench tools referenced by `toolRef` in NPA workflow specs. Each tool is
invoked as a container command; artifacts pass via S3 URIs in `config`.

| toolRef | CLI / module | Typical inputs | Typical outputs |
| --- | --- | --- | --- |
| `workbench.vlm_eval.run` | `npa workbench vlm-eval run` | `config.rollouts_uri` | `config.scores_uri` |
| `workbench.token_factory.reason` | `npa workbench token-factory reason` | `config.scene_uri` | `config.plan_uri` |
| `workbench.cosmos2.transfer` | `npa workbench cosmos2 transfer` | `config.trigger_uri` | `config.augment_uri` |
| `workbench.sim2real_envgen.raw_shard` | `python -m npa.workflows.sim2real_envgen raw-shard` | `config.raw_envs_uri`, `config.env_count` | raw env manifest on S3 |
| `workbench.sim2real.policy_rollouts` | workflow stub (`echo`) | `config.rollouts_uri` | rollout prefix on S3 |
| `workbench.sim2real.heldout_eval` | workflow stub (`echo`) | — | `config.heldout_report_uri` |
| `workbench.sim2real.write_decision` | demo decision writer | `config.decision_uri`, `config.default_decision` | threshold decision JSON |
| `workbench.sim2real.finalize` | workflow stub (`echo`) | `config.finalize_report_uri` | final report URI |
| `workbench.byof.repo` | `npa/scripts/run_byof_repo.py` | `config.repo_url`, `config.repo_ref`, `config.base_profile`, optional `config.build_command` / `config.smoke_command` | BYOF summary, dataset/checkpoint artifacts |
| `workbench.data_transform.rollout_contract` | rollout contract adapter | rollout manifest URI | normalized rollout manifest |
| `workbench.data_transform.improvement_summary` | cross-region summary adapter | heldout/report URIs | improvement summary |
| `workbench.rl.policy_train` | `npa workbench isaac-lab train` | `config.task_name`, training dataset URI | policy checkpoint |
| `workbench.rl.evaluate_policy` | `npa workbench isaac-lab eval` | checkpoint URI, eval episodes | eval report |
| `workbench.rl.write_success_decision` | RL decision writer | eval report URI, `config.success_threshold` | training decision JSON |
| `workbench.rl.publish_policy` | policy release writer | checkpoint + decision URIs | release manifest |
| `workbench.rl.report_failure` | failure report writer | eval + decision URIs | failure report |
| `workbench.lancedb.import_bdd100k` | `npa workbench lancedb import-bdd100k --service` | `config.source_uri`, `config.lance_uri` | LanceDB table |
| `workbench.lancedb.backfill_cpu_bundle` | five CPU UDF backfills | `config.lance_table`, `config.lance_uri` | enriched table |
| `workbench.lancedb.backfill_clip` | CLIP embedding UDF | `config.lance_uri` | `clip_embedding` column |
| `workbench.lancedb.create_failure_views` | three materialized views | `config.rider_view`, … | failure-mode views |
| `workbench.detection_training.train_*` | `npa workbench detection-training train --service` | view + output URIs | checkpoints |
| `workbench.detection_training.eval_*` | `npa workbench detection-training eval --service` | checkpoint + view | metrics JSON |
| `workbench.fiftyone.launch_app` | FiftyOne review hook | `config.lance_uri` | review session |

Creative mashup example: `tokenfactory-cosmos-gate.yaml` (reason → augment → VLM gate loop).

Add new entries in `npa/src/npa/orchestration/npa_workflow/catalog.py` when
exposing a tool to workflow specs.

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
