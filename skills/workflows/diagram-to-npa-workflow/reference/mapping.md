# Diagram → NPA Workflow — mapping reference

Companion to `../SKILL.md`. Everything here is data you consult while turning a
diagram + step write-up into a spec. The authoritative `toolRef` list lives in
`npa/src/npa/orchestration/npa_workflow/catalog.py`; keep this table in sync when
the catalog changes.

## Keyword → `toolRef`

Match diagram/step language (left) to a cataloged tool (right).

| If the box/step says… | Use `toolRef` | Typical config keys |
| --- | --- | --- |
| "Cosmos Transfer", "augment", "synthetic variants", "lighting/texture perturbation" | `workbench.cosmos2.transfer` | `trigger_uri` → `augment_uri` |
| "Cosmos reason", "scene reasoning", "Token Factory", "plan" | `workbench.token_factory.reason` | `scene_uri` → `plan_uri` |
| "Isaac Lab envgen", "raw environments", "generate 1K/8K envs", "scenes+physics" | `workbench.sim2real_envgen.raw_shard` | `raw_envs_uri`, `env_count` |
| "LeRobot training image generates actions", "policy rollouts", "action-conditioned envs" | `workbench.sim2real.policy_rollouts` | `rollouts_uri` |
| "VLM evaluation", "success/failure judgment", "critique", "reward signal source" | `workbench.vlm_eval.run` | `rollouts_uri` → `scores_uri`, `vlm_backend` |
| "Isaac Lab eval framework", "held-out eval", "results report", "success rate on test envs" | `workbench.sim2real.heldout_eval` | `heldout_report_uri` |
| "threshold", "good/bad", "promote or train more", "decision" | `workbench.sim2real.write_decision` | `decision_uri`, `default_decision` |
| "checkpoint", "promote to S3", "finalize", "candidate for deployment", "retrigger record" | `workbench.sim2real.finalize` | `finalize_report_uri` |
| "train RL policy" (generic sim RL, not VLM-in-loop) | `workbench.rl.policy_train` | `task_name`, `train_dataset_uri` → `checkpoint_uri` |
| "evaluate policy on held-out episodes" | `workbench.rl.evaluate_policy` | `checkpoint_uri` → `eval_report_uri` |
| "success gate on threshold" (RL) | `workbench.rl.write_success_decision` | `success_threshold`, `decision_uri` |
| "publish/release promoted checkpoint" | `workbench.rl.publish_policy` | `release_uri` |
| "record failure / not promoted" | `workbench.rl.report_failure` | `eval_report_uri`, `decision_uri` |
| "import BDD100K / dataset into LanceDB" | `workbench.lancedb.import_bdd100k` | `source_uri`, `lance_uri`, `lance_table` |
| "backfill CPU columns / dedup / bbox" | `workbench.lancedb.backfill_cpu_bundle` | `lance_table`, `lance_uri` |
| "CLIP embeddings" | `workbench.lancedb.backfill_clip` | `lance_uri` |
| "materialized views / failure-mode slices" | `workbench.lancedb.create_failure_views` | `*_view` |
| "train detector on view" | `workbench.detection_training.train_{rider,nighttime,distant}` | `*_view` → `*_train_uri` |
| "evaluate detector" | `workbench.detection_training.eval_{rider,nighttime,distant}` | `*_train_uri` → `*_eval_uri` |
| "FiftyOne review", "human-in-the-loop viz" | `workbench.fiftyone.launch_app` | `lance_uri` |
| "onboard OSS repo", "BYOF", "build+push+train custom image" | `workbench.byof.repo` | `repo_url`, `workload`, … |
| "deploy Slurm/soperator cluster" | `infra.soperator.deploy` | `soperator_spec` |
| "normalize rollout contract" / "cross-region improvement summary" | `workbench.data_transform.rollout_contract` / `.improvement_summary` | `project_*`, `region_*` |

No match? Add a `ToolEntry` in `catalog.py` (argv template with `{{config.*}}`
tokens), or use `run.shell` for genuinely one-off glue.

## Control-flow patterns

| Diagram cue | Spec fragment |
| --- | --- |
| Straight arrow A→B | `A: { next: B }` |
| Ordered trio under a labeled group | `group: { sequence: [a, b, c] }` |
| "runs N times" / fixed count | `parent: { loop: { max: "{{config.iters}}" }, sequence: [...] }` |
| "iterate until threshold met" | `parent: { loop: { max: "{{config.iters}}", until: promote_checkpoint }, sequence: [...] }` |
| Decision diamond, two exits | see "Decision state" below |
| Nested loops (inner + outer) | `outer` loop whose `sequence` includes `inner`, which has its own `loop.max` |
| Real-world → retrigger Step 1 | terminal `finalize`/`retrigger` state (NOT a back-edge — see below) |

### Decision state

```yaml
decide:
  writesDecision: true
  toolRef: workbench.sim2real.write_decision
  outputs:
    - uri: "{{config.decision_uri}}"
      schema: npa.sim2real.threshold_decision.v1
  transitions:
    - when: promote_checkpoint   # forward
      goto: finalize
    - when: loop_back            # backward into the loop parent
      goto: outer
```

Predicates are a closed set: only `promote_checkpoint` and `loop_back`. Planning a
spec with `transitions` requires `--assume-decision promote_checkpoint|loop_back`.

### Why the loop-of-loops is terminal, not a cycle

`validate-spec` runs `_assert_bounded_control_flow_cycles`: any `next`/`transitions`
path that returns to an already-visited state without a bounding `loop` is
rejected as "unbounded control-flow cycle". Loops (`loop.max` / `loop.until`) are
the *only* sanctioned back-edges, and they are expressed by a parent state wrapping
a `sequence`, not by a `goto` that jumps backward past the loop parent. A
real-world-test → retrigger-Step-1 arrow therefore becomes a terminal state that
writes a retrigger manifest; the next outer-of-outer iteration is a brand-new run.

## Reconciling discrepancies (diagram vs write-up)

The user warned "there will be some discrepancy." Apply these rules and leave a
`# note:` comment on each affected state:

| Discrepancy | Resolution |
| --- | --- |
| Step numbers skip/duplicate (e.g. no "Step 2") | Numbering is cosmetic; order states by arrows, not by step index. |
| Env counts disagree (text "1,000 raw / 800-200" vs boxes "8K/2K") | Keep counts as `config` knobs (`env_count`, `train_fraction`); do not hardcode a specific integer into topology. Note both. |
| Split "80/20" drawn as a diamond | The split is a data operation inside envgen, not a decision diamond — one `envgen` state emits a split manifest; only the **threshold** is a decision state. |
| A vendor/product name (Cortex, Lightwheel, Newrobot) | Map to the *capability* tool, not the brand. "Cortex trainer fork" → `policy_rollouts` + VLM reward; "Lightwheel eval" → `heldout_eval`. |
| Two boxes, same tool (Isaac appears twice) | Two states, same `toolRef` family, different `resources`/URIs. |
| Curate/review human step with no tool | Fold into the trigger artifact URI, or model as a `fiftyone.launch_app` review hook. |

## Sim2real worked example (13 steps → spec)

The provided write-up + diagram map to `examples/sim2real-vlm-rl-from-diagram.yaml`:

| Step (write-up) | Diagram box | State | `toolRef` |
| --- | --- | --- | --- |
| 1 Trigger (LeRobot data in S3) | DB → Curate/Review → LeRobot Data | *(trigger_uri artifact)* | — |
| 3 Augment good data (Cosmos Transfer 2.5) | Cosmos Transfer box | `augment` | `workbench.cosmos2.transfer` |
| 4 Load sim assets into Isaac | *(feeds envgen)* | *(assets_uri input on `envgen`)* | — |
| 5 Raw env generation (Isaac Lab) | Isaac Lab Envgen box | `envgen` | `workbench.sim2real_envgen.raw_shard` |
| 6 80/20 split | split diamond | *(split manifest output of `envgen`; `train_fraction` config)* | — |
| 7 LeRobot generates actions on train envs | 8K TrainEnvs → LeRobot | `rollouts` (in `inner`) | `workbench.sim2real.policy_rollouts` |
| 8 VLM evaluation on training | VLM Eval box (inner loop) | `vlm-score` (in `inner`) | `workbench.vlm_eval.run` |
| 9 VLM critique → RL update | inner back-edge | *(inner loop `loop.max`; reward wired via BYO trainer in engine)* | — |
| 10 Isaac Lab eval framework | Lightwheel Eval → Results Report | `heldout` | `workbench.sim2real.heldout_eval` |
| 11A/11B Good/Bad threshold | Threshold diamond | `decide` | `workbench.sim2real.write_decision` |
| 11A promote → checkpoint to S3 | Checkpoint → S3 | `finalize` (promote path) | `workbench.sim2real.finalize` |
| 11B fail → more RL | outer back-edge | `decide` `loop_back` → `outer` | — |
| 12 Real-world test | Real-world Test box | *(recorded in finalize manifest)* | — |
| 13 Retrigger (real data → Step 1) | retrigger arrow | `finalize` (terminal; new run re-enters) | — |

Loops: **inner** = steps 7–9 (`loop.max: inner_iterations`); **outer** = steps
7–11 (`loop.max: outer_iterations`, `until: promote_checkpoint`); **loop-of-loops**
= steps 12→13→1 (terminal retrigger, new run).

## Validator error → fix

| Error text | Fix |
| --- | --- |
| `unknown toolRef 'x'` | Add `x` to `catalog.py`, or pick the right existing ref. |
| `unbounded control-flow cycle detected: …` | A `next`/`goto` loops back without a `loop` parent. Wrap the repeated states in a `loop` state, or make the back-target terminal. |
| `unknown loop.until 'x'` / `unknown transition.when 'x'` | Only `promote_checkpoint` / `loop_back` are valid predicates. |
| `state X: must set run, toolRef, sequence, transitions, next, or terminal` | Give the state a body or mark it `terminal: true`. |
| `config has no attribute 'k'` (loop max) | Add `k` to `config`, or use an integer literal. |
| `workflow must declare at least one terminal: true state` | Mark each leaf `terminal: true`. |
| `initial state 'x' is not defined` | `initial:` must name a real state key. |
| `transition goto unknown state 'x'` | `goto` must reference a defined state. |
