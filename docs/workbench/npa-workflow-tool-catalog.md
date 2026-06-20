# NPA workflow tool catalog (v0.0.1)

Workbench tools referenced by `toolRef` in NPA workflow specs. Each tool is
invoked as a container command; artifacts pass via S3 URIs in `config`.

| toolRef | CLI / module | Typical inputs | Typical outputs |
| --- | --- | --- | --- |
| `workbench.vlm_eval.run` | `npa workbench vlm-eval run` | `config.rollouts_uri` | `config.scores_uri` |
| `workbench.token_factory.reason` | `npa workbench token-factory reason` | `config.scene_uri` | `config.plan_uri` |
| `workbench.cosmos2.transfer` | `npa workbench cosmos2 transfer` | `config.trigger_uri` | `config.augment_uri` |
| `workbench.sim2real_envgen.raw_shard` | `python -m npa.workflows.sim2real_envgen raw-shard` | `config.raw_envs_uri`, `config.env_count` | raw env manifest on S3 |

Add new entries in `npa/src/npa/orchestration/npa_workflow/catalog.py` when
exposing a tool to workflow specs.

## Tokens

| Token | Meaning |
| --- | --- |
| `{{config.*}}` | Value from spec `config` block (after run-id expansion) |
| `{{run.id}}` | Run identifier passed to plan/run commands |
| `{{run.prefix}}` | Default `"{metadata.name}/{run.id}"` or `config.prefix` |

## Predicates

| Name | True when |
| --- | --- |
| `promote_checkpoint` | Last decision is promote |
| `loop_back` | Last decision is loop-back |
