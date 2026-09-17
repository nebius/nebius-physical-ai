# BEHAVIOR 2026 RLC experiment results — September 17, 2026

This development experiment tested the published RLC π0.5-derived policy, one
execution-horizon change, three released-data fine-tuning runs, and one
independent public checkpoint. The evaluated fine-tune and controller variant
produced no improvement over their observed stock controls. These are private
development results, not a leaderboard score, full 100-task result, or organizer
submission. See [BEHAVIOR challenge evaluation](behavior-challenge.md) for the
protocol and [policy research](behavior-policy-research.md) for the earlier
research review.

## Protocol

Every evaluated cell used the unchanged BEHAVIOR-1K v3.9.2 evaluator, official
R1Pro robot configuration, full-resolution RGB-D observation wrapper, prescribed
development instances 311–320, and the task's default timeout. Each cell ran one
attempt per instance. The policy received only permitted RGB and proprioception;
no simulator pose, global yaw, task state, or evaluator feedback entered
inference.

The frozen three-task panel was:

- task 0, `turning_on_radio`;
- task 1, `picking_up_trash`;
- task 22, `putting_shoes_on_rack`.

Collectors retained the original evaluator JSON and videos, verified artifact
hashes, and decoded every video. All eight requested cells completed 10/10
instances, for 80 development attempts in total. This campaign ran no reporting
instances and produced no submission ZIP.

## Development results

Q is the official goal-completion score and includes partial credit.

| Policy | Task | Cases | Mean Q | Successes |
| --- | --- | ---: | ---: | ---: |
| Published RLC checkpoint 2, stock execution | 0 — radio | 10/10 | 0.300000 | 3 |
| Published RLC checkpoint 2, 16-action uncompressed execution | 0 — radio | 10/10 | 0.000000 | 0 |
| Uniform radio fine-tune, 6,000 updates | 0 — radio | 10/10 | 0.200000 | 2 |
| KMY public step-5000, horizon 16 | 0 — radio | 10/10 | 0.000000 | 0 |
| Published RLC checkpoint 2, stock execution | 1 — trash | 10/10 | 0.433333 | 3 |
| Published RLC checkpoint 2, 16-action uncompressed execution | 1 — trash | 10/10 | 0.066667 | 0 |
| Published RLC checkpoint 2, stock execution | 22 — shoes | 10/10 | 0.550000 | 1 |
| Published RLC checkpoint 2, 16-action uncompressed execution | 22 — shoes | 10/10 | 0.210000 | 0 |

Across the frozen three-task panel, stock execution averaged Q=0.427778 with
7/30 successes. The 16-action profile averaged Q=0.092222 with 0/30 successes.
The shorter uncompressed horizon therefore did not improve this panel.

The uniform radio fine-tune scored Q=0.20 versus Q=0.30 for the campaign's
stock-checkpoint radio cell, so this run provides no evidence of a fine-tuning
gain. Ten cases on one task are too few to estimate a general effect.

An earlier September 16 stock-RLC radio run remains separately recorded at
Q=0.20, alongside the official radio baseline's Q=0.10 on that selection. This
campaign independently evaluated the same RLC checkpoint and prescribed cases.
The current Q=0.30 cell does not replace the earlier result; both remain visible.

Both task-22 payloads are collector-valid and complete, but their outer
workflows reported `FAILED` while each sole evaluation job reported `SUCCEEDED`.
The supervisors failed after six consecutive scheduler queries returned invalid
authentication, and their cancellation attempts also failed. The baseline and
shorter-execution evaluator jobs subsequently completed, approximately 23 seconds
and 24 minutes after the respective outer failures. Both exited successfully and
uploaded all declared outputs. The table uses the durable 10/10 evaluator
artifacts; the outer workflow failures remain part of the evidence.

## Work completed and next measurements

The eight evaluated cells executed 498,108 simulator steps across 80 rollouts.
Those rollouts cover 30 distinct task/instance pairs, with different policies
evaluated on the same prescribed cases. The three training runs completed
18,000 optimizer updates in total. Only one of the three exported fine-tunes
received a development evaluation.

Matching each candidate to the stock policy by task and instance makes the
negative results more concrete:

| Candidate | Paired cases | Higher Q | Lower Q | Equal Q | Mean Q change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-action uncompressed execution, three tasks | 30 | 1 | 17 | 12 | -0.335556 |
| Uniform radio fine-tune | 10 | 0 | 1 | 9 | -0.100000 |
| KMY public step-5000, radio | 10 | 0 | 3 | 7 | -0.300000 |

These are descriptive comparisons from one attempt per policy and case. They
do not estimate significance, simulator variance, or performance on other tasks.

The next experiment should prioritize measuring preserved candidates:

1. Verify renewable authentication and independent deadline cleanup before
   provisioning another campaign. The previous cleanup overrun is unresolved
   as a reusable automation fix.
2. Freeze the balanced three-task checkpoint and published stock control, then
   evaluate both on the same complete panel: six cells and 60 attempts. Retain
   every outcome and record the new comparison separately from this campaign.
3. Verify Meta100 GPU serving, observation/action compatibility, normalization,
   and the submission memory constraint before evaluating it on that panel.
4. Audit the contact annotations against decoded frames before using their
   labels to interpret that candidate. Add held-out loss measurement before
   another training campaign, alongside rollout-based model selection.

This ordering has no new measured performance result. It scores existing
training investments before expanding training or controller experiments.
Further GPU execution requires a new operator time allowance because the
original 12-hour window has ended.

## Completed training artifacts

All three candidates started from published RLC checkpoint 2 and completed
6,000 native JAX updates, with the exported final checkpoint indexed as step
5,999. Training used only released 2026 demonstrations. Development and
reporting instances were excluded.

| Candidate | Released training data | Exported archive SHA-256 | Evaluation status |
| --- | --- | --- | --- |
| Uniform radio action-expert + stage fine-tune | Task 0; 180 train / 20 held out | `cf180473dcb869e53ea863fa3a3b3d1c7544db52e5b067e811a9769b40d2d2a5` | Radio development Q=0.20 |
| Balanced task-panel fine-tune | Tasks 0, 1, 22; 180 train / 20 held out per task | `ec5ef7ba824acb44cd87fdb12e535fb9dc10faa01c4cdcfd763e5fd441d6cdfb` | Not evaluated before the deadline |
| Radio press-interval curriculum | Task 0; same 180 / 20 split, with 50% of draws from released `press` intervals | `33e4758b5a85fb809b5637011834f67a30651dfb7ac04cde62bd0f83277e8d43` | Not evaluated before the deadline |

The dataset revision was
`4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`. All exports preserved the
published normalization bytes, SHA-256
`ccd14a0210fc59b2d2726ba599cc0c4b81347395dd60d2a15b334b28ed15a80b`.
The published parent checkpoint archive was
`9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea`.

Held-out loss is unmeasured because the pinned native trainer has no validation
loop. The held-out episodes remained excluded from training.

The contact curriculum tests annotation-driven press-interval oversampling.
Checks verified annotation bytes, bounds, indexing and deterministic sampling;
semantic alignment of those intervals to RGB/action frames was not independently
verified. The earlier annotation-duration discrepancy remains unresolved. This
candidate supplies no verified contact labels and does not reproduce RLT.

The first contact-curriculum launch failed before a training batch or model was
produced. Its RGB-only data view had retained three
`observation.depth_linear.*` feature declarations, so LeRobot requested depth
videos that were intentionally absent. The corrected view removed only those
three declarations from copied metadata, preserved source data bytes, passed a
real first-batch decode/transform check, and completed the 6,000-update second
run. This was a data-view metadata correction, not a training-data change.

## Prepared work that did not run

The panel and contact checkpoints were fully collected but received no
development evaluation before the hard deadline. Observation-conditioned
`transition_refresh`, `stage_hysteresis_3of4`, bounded stage-4 confirmation, and
proprioception-gated stabilizer-latch profiles were packaged and tested but not
launched. Their tests establish deterministic state/reset behavior only.

An eight-GPU evaluation extension, shared read-only cache, isolated evaluator
roles, paired radio-control harness, and reporting coordinator were prepared and
reviewed but never launched. They provide no rollout evidence.

A newly published [Meta100 step-139999 checkpoint](https://huggingface.co/JackLiu0406/meta-SFT-checkpoints),
commonly described as the 140k checkpoint, completed only a CPU Orbax host restore
of 75 array leaves. The restore did not construct the author model, load
parameters on a GPU, produce an action, start a policy server, or consume an
evaluator case. It has no rollout result in this experiment. The restored task
and task-stage embeddings had
shapes `(100, 2048)` and `(1120, 1024)`. The serving adapter follows the author's
[pinned source](https://github.com/JackLiu0406/behaviour-1k-2026-meta/tree/7146d7b179d2391db963f564868f6ca313185bce)
and actual tokenizer metadata: a 30-action model horizon. This checkpoint uses
different normalization from the RLC parent and requires its own serving check.

The task-1 and task-22 short-horizon cells continued to completion because the
session paused before a planned panel-readiness cutover could stop them. Their
results were retained in full; no cases or cells were selected based on their
outcomes.

## Limits and practical conclusions

- The evaluated evidence covers three of 100 tasks and only the development
  split. It says nothing about the full 1,000-case challenge score.
- Neither the 6,000-update uniform radio fine-tune nor 16-action execution
  improved its matched observed control. The panel and contact fine-tunes remain
  unevaluated.
- Success count and mean Q capture different behavior: task 22 stock execution
  had one full success but Q=0.55 from partial predicate completion across the
  ten cases.
- The current and historical radio cells differ despite the same published
  checkpoint and prescribed cases. Both must remain visible; a larger frozen
  evaluation is needed before attributing a change to a controller or training
  recipe.
- Next measurements should cover the frozen panel fine-tune and the Meta100
  checkpoint, after its GPU serving check. The contact candidate also needs an
  annotation-alignment audit before interpreting it as a press-focused method.

The fixed resource-retirement deadline passed while the operator session was
paused and the run's static cloud token no longer authenticated. No new training
or evaluation was launched after the deadline. After refreshing authentication,
owned fleet retirement finished at 16:16:45 UTC, approximately 86 minutes after
the 14:50:29 UTC hard deadline. The requested 12-hour resource limit was missed.

An independent provider inventory at 16:17:36 UTC confirmed zero instances,
disks, filesystems, clusters, public-IP allocations, AI endpoints, or AI jobs in
the campaign project. All 12 workflow records were terminal, all 12 controller
cleanup receipts verified remote absence, and all eight data volumes were
retired. The campaign observer was stopped. The project, artifact bucket,
storage access, and default network were retained to preserve the results.

## Reproducibility identities

- Independently recomputed eight-cell totals: `f678f72a7a8c3a8f8a2e80f51ca8253b4c59f9993fcd5de5f3320739e57f0409`
- Paired development review and work accounting: `ac54987ab767505639fb44b13628cb5d83ede668f96a9878cddf858be51d4083`
- Six newly collected cells and workflow status receipts: `17302b8f306e738f6ad6a6df854be864ac84120341904a6e278573f9faff4fe8`
- Task-22 supervisor authentication-failure audit: `c36e6eb9ca76e2338f910a72a91a13d1c16c94bfeadd0d8e4b46ac7bb67179b8`
- Final independent cleanup audit: `54ed5907bb9e11107b637607bfcc7639bec55028eae4bf5859cd2900ef3b993f`
- Workbench experiment source: `d430e4157db5dddec6ece2ccdf3478e9bd543524`
- Official evaluator source: `b1979916ec1549b10a4e65e630bc6504a9af1b00`
- Training sources: RLC `ca556f74a455cef7987a2be4537b5ac85cc56dd7`, OpenPI
  `01177e0242a1c7e8fad2547caa0e987def614cda`, BEHAVIOR-1K
  `684a83050ddd398de231e6aa7fc605bc34458d4b`, and LeRobot
  `c43f58116b975ae79af62714e1417b38facd4e37`
- Published RLC checkpoint 2 archive: `9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea`
- KMY public step-5000 archive: `315da34851e63669b4d5a57060a8559531c579e68da02bb8bf78578f065fa97f`
- Released 2026 training dataset revision: `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`

These hashes identify models, source/data inputs, and retained evaluation and
cleanup receipts. Exact operational receipts remain in access-controlled
evidence and are intentionally omitted.
