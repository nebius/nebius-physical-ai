# BEHAVIOR matched training and development evaluation — September 19, 2026

Both matched training arms completed **3,600 optimizer updates**, scored their
six retained checkpoints on the frozen holdout, and exported a selected model.
That is **7,200 new planned optimizer updates**. The replay arm selected update
3599 with equal-task mean held-out action loss **0.160986**. This is an offline
training metric; it is not the challenge's task-completion score. The teacher
control also selected update 3599, with its own conditioning loss **0.160796**.

The replay-selected update 3599 completed all six development cells by
September 20. Its equal-task mean Q is **0.298750**, versus stock's **0.324028**,
with **10 versus 12** full successes across the same sixty cases. The trained
tasks regressed to Q=0.276667 from stock's Q=0.402222. Cooking hot dogs improved
to Q=0.75 from stock's Q=0.50, but that gain did not offset the other losses.
This experiment does not establish an overall improvement. The previous balanced
fine-tune regressed, as documented in the
[earlier comparison](behavior-followup-results-2026-09-18.md).

## Experiment

The two arms use identical released examples, ordering, initialization,
optimizer settings, and random seeds. The control uses demonstration stage
labels. The replay arm uses the frozen parent policy's stage predictions from
the preceding replan point. The hypothesis is that training actions with the
conditioning available during inference will reduce errors caused by stage
prediction mismatch. This experiment does not train a separate recovery policy.

Only 23 declared action-path parameter leaves are trainable. Vision, language,
task selection, and stage-classification parameters remain frozen. Frozen
exponential-moving-average parameters are explicitly preserved byte-for-byte.
Both arms passed two disposable compiled updates through the native metric
reduction and numeric logger before rebuilding production state from the seed.

The data contains 180 training and 20 held-out released episodes for each of
tasks 0, 1, and 22: 540 training episodes and 60 holdout episodes in total.
Canonical single-sample parent inference produced 137,090 training and 14,629
holdout replan records. Development and reporting rollouts do not enter these
traces or optimizer inputs. The native dataset reader validated all 600
episodes and 3,028,707 frames.

Each arm ran batch size 16, four flow samples, and 3,600 updates, retaining
checkpoints 600, 1200, 1800, 2400, 3000, and 3599. Holdout evaluation used all
14,629 fixed records per checkpoint, eight flow draws, and seed 1701. The
predeclared selection rule minimizes the equal-task mean action loss, breaking
exact ties in favor of the earlier checkpoint. The replay arm was the
predeclared rollout candidate; losses under the two different conditioning
regimes do not establish a cross-arm winner. The completed rollout comparison
below retains stock as the overall baseline.

Selection receipt hashes are `c08fd339dd85750930c7be7198204d4f441fad2bda5c1c7a1ce6337d29009072`
(replay) and `b820c464a2f2ec883b47bc0181eebfde32a6be6688386fffe1babea57cc980fe`
(teacher). Both export receipts and the underlying holdout rows are retained
with the private run evidence.

See the [training guide](behavior-matched-training.md) for executable commands
and the exact training contract.

## Completed replay-selected development cells

The replay-selected model used native selected execution and the unchanged
official evaluator on development instances 311–320, once each. Original
metric and video identities were retained, every video decoded fully, and the
comparison used the same stock cases listed below.

| Task | Policy | Mean Q | Full successes | Simulator steps |
| --- | --- | ---: | ---: | ---: |
| Turning on radio | Stock | 0.400000 | 4/10 | 28,422 |
| Turning on radio | Replay-selected update 3599 | 0.100000 | 1/10 | 29,962 |
| Picking up trash | Stock | 0.366667 | 2/10 | 72,280 |
| Picking up trash | Replay-selected update 3599 | 0.300000 | 2/10 | 73,119 |
| Putting shoes on rack | Stock | 0.440000 | 1/10 | 115,199 |
| Putting shoes on rack | Replay-selected update 3599 | 0.430000 | 0/10 | 115,900 |
| Rearranging kitchen furniture | Stock | 0.000000 | 0/10 | 134,150 |
| Rearranging kitchen furniture | Replay-selected update 3599 | 0.000000 | 0/10 | 134,150 |
| Setting the fire | Stock | 0.237500 | 0/10 | 136,780 |
| Setting the fire | Replay-selected update 3599 | 0.212500 | 0/10 | 136,780 |
| Cooking hot dogs | Stock | 0.500000 | 5/10 | 111,803 |
| Cooking hot dogs | Replay-selected update 3599 | 0.750000 | 7/10 | 108,274 |

On radio, the selected model was better on zero paired cases, tied on seven,
and worse on three. It scored zero on instances 311–319 and completed instance
320 with Q=1.0 in 937 steps. On trash, it was better on two cases, tied on four,
and worse on four. On shoes, it was better on four cases, tied on one, and worse
on five. Across the three trained tasks, six cases improved, twelve tied, and
twelve worsened. Selected execution used 218,981 simulator steps versus stock's
215,901. The furniture transfer task tied on all ten cases, including simulator
steps. Setting the fire improved three cases, tied two, and worsened five.

Across all six tasks, selected execution used 598,185 simulator steps versus
stock's 598,634. Mean Q fell by 0.025278, with ten versus twelve full successes.
Thirteen paired cases improved, twenty-nine tied, and eighteen worsened.
The three transfer tasks averaged Q=0.320833 versus stock's Q=0.245833; that
subgroup improvement did not offset the trained-task regression. These are
reused development cases, with one rollout per case. They do not establish an
unseen-test result or the full challenge score.

The candidate-cell, comparison, task-summary, and paired-case evidence have
SHA-256 values `11bd401fed491e1caf388391963cba20d3952c6970d3392095e1fbfc4bc2e6eb`,
`73b6d34569af3853df17ca30587cd89b863533aa27dacc645861ba1ac4872b0b`,
`1d496f0a7216b4de186e9f2e57f72526108ebc9aa7e21c9c74270225e7a78be0`,
and `52058598612125d908860c667476c7e2d90e31737a7be67b748f6d6a6361a7b2`,
respectively, for the radio comparison. The trash candidate-cell SHA-256 is
`6fba61fe05ea091536f88af8049517a5ed529ef86c1cf044a0d7ba8f2040dc0d`.
The shoes candidate-cell SHA-256 is
`7b49d4a1c43e6011b0d297046008b104da6e6996c86f5bcbc6449c042dacf0c1`.
The complete three-task comparison has SHA-256
`9eeae6914477f1fc3d8cd51f580e227c11296d5175c897aa4151ec371bab8b26`.
The furniture candidate-cell SHA-256 is
`69c9d994910cee3ef96702074873f6f852289fcefa68dc14e8877a70cd27ada6`.
The setting-the-fire candidate-cell SHA-256 is
`2e04a2cae4f346491cab2909325b36ca90347fcee557145627eccc956c5049e3`.
The cooking-hot-dogs candidate-cell SHA-256 is
`253434c7bbb1b4f5da32a4a0f0515bf2178325496f750d093af4ce25331038cf`.
The [complete selected per-case CSV](behavior-matched-candidate-cases-2026-09-20.csv)
contains all sixty cases and their original metric identities; its SHA-256 is
`713b7ae0690259f9c8125b959b8bb1665a974bc5cfc718d8865128f139163aeb`.
The complete paired comparison has SHA-256
`5455d7fec577d6da8a63640272bfd8fdcc05101734b470788cb690e77919cf12`.
An operator restart interrupted the last native cell's parent submitter. Its
original ten-case evaluator artifacts are complete, and a fresh controller
query verified scheduler success. The durable workflow completion record
remained unset. The report counts the verified evaluator cases and retains
that orchestration limitation.
The optional transition-refresh public module was not used for these cells and
remains without a public GPU validation or rollout score.

## Completed stock-policy development cells

Each complete cell uses the unchanged official evaluator and all ten prescribed
development instances 311–320, once each. Q is the evaluator's goal-completion
score, including partial credit. Original metric and video hashes are verified,
and every video is fully decoded before a cell is counted complete.

| Task | Mean Q | Full successes | Simulator steps |
| --- | ---: | ---: | ---: |
| Turning on radio | 0.400000 | 4/10 | 28,422 |
| Picking up trash | 0.366667 | 2/10 | 72,280 |
| Putting shoes on rack | 0.440000 | 1/10 | 115,199 |
| Rearranging kitchen furniture | 0.000000 | 0/10 | 134,150 |
| Setting the fire | 0.237500 | 0/10 | 136,780 |
| Cooking hot dogs | 0.500000 | 5/10 | 111,803 |

These six completed cells total 60 rollouts and 598,634 simulator steps.
Their equal-task mean is Q=0.324028, with twelve full successes. These are
released development cases, including cases reused from earlier campaigns;
they do not constitute an unseen test or the full 100-task challenge score. The
[per-case CSV](behavior-matched-stock-cases-2026-09-19.csv) includes the original
metric and collection hashes; its SHA-256 is
`6bad7d083d72f9938e476f85546297765ca11b2a37010809ebeb5ac6f5db9a64`.

Two other cells stopped at the scheduled shutdown boundary:

| Task | Completed prescribed cases | Full successes | Simulator steps |
| --- | ---: | ---: | ---: |
| Slicing vegetables | 5/10 (311–315) | 0 | 111,340 |
| Moving boxes to storage | 9/10 (311–319) | 3 | 175,572 |

Their fourteen completed episodes passed original metric/video hash checks and
full video decoding. They are excluded from the six-cell aggregate. The
[partial-case CSV](behavior-matched-partial-cases-2026-09-19.csv) preserves their
observed results; it cannot supply missing cases or a complete-task score.
Its SHA-256 is `8ff7a71b0ca1943cbc859c4c00ef33ab4c24b8a2a0a80b19d5ee45618b18046e`.
Across complete and incomplete cells, this campaign preserved **74 completed
stock-policy rollouts, 885,546 simulator steps, and 15 full successes**.

## GPU execution and recovered artifacts

Two B200 GPUs ran the matched training arms. Six additional B200 workers were
started to score replay checkpoints in parallel; the original inline holdout
evaluations completed first. RTX PRO 6000 workers ran the robot simulator and
policy inference. The provisioned fleet had eight B200 and sixteen RTX GPUs;
that capacity count does not mean all GPUs were continuously occupied.

The native replay workflow completed at 02:09:42 UTC and the teacher workflow
at 02:12:33 UTC. Both reported success after holdout selection and model export.
Managed-job shutdown then removed their pods, interrupting auxiliary checkpoint
backup children. All six replay inference archives had already been published.
Both selected models and all six inference checkpoints per arm are retained.
The teacher's final inference archive was verified independently after its
archival worker was interrupted: all fifteen files exactly match the native
selected export.

The full replay update-3599 archive also reached storage before cancellation.
A CPU-only recovery read all 15,705,114,945 compressed bytes, verified gzip
integrity and safe archive structure, matched fifteen serving files to the
previously published checkpoint, and matched native checkpoint metadata. It
recorded hashes for thirteen optimizer-state files. There is no earlier full-tree
hash receipt for those files, so their identity to the original volume and
native optimizer restoration remain unverified. The teacher's full final
optimizer-state archive was not preserved.

Independent recovery receipt SHA-256:
`9d87e3adb6b0ae53c3eec7765275e0e523fb336eb23796ce07a2f1a5b6a9658f`.
Replay archive SHA-256:
`14f120f70fae74cad0bb777418dbb7655845362e99e19fd0f840b3bbf13f9362`.
The interrupted workers remain recorded as interrupted; the recovery has its
own receipt. An earlier recovery attempt also failed after I paused an archival
child during diagnostic coordination, causing its managed parent to terminate.
The completed training runs and selected exports were unaffected.

Earlier scorer attempts failed because their ephemeral-storage request exceeded
node capacity. Corrected requests retained the same model, examples, and loss
calculation. A separate 100-task serving smoke failed before model loading on a
reused run-directory assertion. It provides no evidence of 100-task serving or
submission eligibility.

## Serving consistency and the parameter-filter correction

An independent restore comparison at update 600 matched every learned parameter
path, shape, dtype, and value between the native EMA state and the exported
inference checkpoint. The subsequent first-batch loss comparison failed. That
is a serving consistency failure, not evidence of different learned weights.

The native initializer uses the freeze filter both to identify frozen weights
and to cast them to BF16. The previous filter also matched non-parameter
`nnx.Intermediate` state. A real Flax 0.10.2/JAX 0.5.3 test reproduced conversion
of the correlation statistics to BF16; intersecting the filter with `nnx.Param`
keeps those statistics in FP32 and leaves the action-only trainable set intact.
The public implementation now has that correction and a native regression test.
This change applies to future runs. It does not modify the completed runs,
recompute their holdout selection, or make their exports serving-eligible.

The B200 diagnostic completed at update 600. Among all 75 typed model-state leaves,
exactly one differed: the 960×960 `action_correlation_cholesky` intermediate was
BF16 in the native state and FP32 in direct serving. Scalar configuration
matched. On the first 16 fixed holdout records (seed 1701, eight flow draws),
action loss differed by up to 0.00023824 and per-dimension loss by 0.00581741;
stage logits and cross-entropy were already byte-identical.

Replacing only the direct model's correlation array with the native array made
all four metrics byte-identical. Reloading only the native model's correlation
array from canonical normalization likewise matched the original direct model.
This isolates the correlation precision difference as the cause of this
first-batch failure. The [machine-readable diagnostic summary](behavior-serving-parity-2026-09-19.json)
records the exact differences and the original evidence hash.

The selected update-3599 checkpoint subsequently passed the same state and
serving boundary on a B200. Before adaptation, the same correlation intermediate
was the only difference among 75 typed leaves. Installing its validated native
BF16 bytes before the existing policy constructs its JIT sampler made all typed
state equal. Four fixed-batch metric parts matched byte-for-byte between the
native model and selected export. Fixed-RNG physical actions matched across the
native model, selected export, and existing serving wrapper. The wrapper also
returned one finite 23-element physical action.

The auxiliary stage-logit tuple component also matched byte-for-byte. Its numeric
maximum and mean deltas are undefined because invalid task stages use identical
negative-infinity mask sentinels: subtracting `-inf` from `-inf` produces no
finite delta. The public JSON records those two deltas as `null`, with this
reason, and retains the raw validation receipt hash. This serving result admits
the checkpoint to development rollouts. It is not a rollout result or evidence
that training improved the policy.

The stock baseline constructs FP32 correlation statistics, while these selected
rollouts preserve the completed training run's native BF16 statistics. The
stock-versus-selected comparison therefore changes both learned weights and
this intermediate. At update 600, changing only correlation precision raised
the fixed-batch mean action loss by 0.03048% with BF16; it left stage predictions
unchanged. No selected-update-3599 FP32-versus-BF16 same-RNG actions were retained.
These diagnostics establish an inference-state difference, but do not establish
its effect on rollout Q or explain the radio regression.

## Selected failure observations

The first chronological failed radio and trash rollouts were inspected at eight
uniformly spaced frame indices, including the first and last. Both original
videos matched their hashes and decoded completely.

In the radio failure, the robot was near the table early, followed by a long
interval with little visible change in the table-relative view. At the final
sample both grippers were visible, with the left near the radio; the images do
not establish a secure grasp or switch actuation. In the trash failure, sampled
frames showed prolonged interaction near an open refrigerator and late movement
into the living area, without visible pickup completion. These observations
motivate investigating stalls and recovery. Two outcome-selected failures and
sparse frames cannot establish frequency or cause.

## Follow-up execution experiments

The follow-up used RTX PRO 6000 GPUs for development rollouts and a B200 for
the selected-checkpoint serving comparison. It performed **zero additional
training updates**; the 7,200 updates above belong to the completed matched
training experiment.

A private stock-weight radio variant was configured to refresh queued actions
when its stage changed. Its evaluator produced all ten prescribed development
cases: diagnostic mean Q=0.10, one full success, and 30,154 simulator steps.
Stock scored Q=0.40 with four successes. All ten recorded videos decoded fully.
However, the run then failed its requirement for one ordered, nonempty telemetry
episode per case. A model-load warning had installed a WARNING-level logger
before the private server configured INFO logging, suppressing the telemetry
records. The actual transition counts cannot be recovered from the retained
logs. The run did not publish a validated variant score. These outcomes
are diagnostic evidence only, are excluded from the selected-model aggregate,
and do not validate the optional public transition-refresh implementation.
Native execution remains the default.

The original failed run-status receipt has SHA-256
`001a0fe166d9fed55d1bc748e03114c48829ac872d9e1fa2ffc631a1b4e2e399`.

A separate private experiment allows the existing backward-stage vote to operate
after the policy reaches its final stage. It preserves the model weights and
action queue. Its corrected worker passed an RTX startup check of all 75 typed
model-state entries against the selected export, plus all fifteen checkpoint
file identities, before evaluation. The startup receipt has SHA-256
`b8f150cf03178945fc50815e78fbcfa1496129f088eee25deec71b3faca769e5`.
Earlier attempts failed before producing episodes; those failures remain in
the evidence. A Python 3.11/3.12 syntax-tree serialization difference was fixed
without relaxing the source check, and telemetry logging was enabled before
model loading. These are private campaign changes, with no public GPU validation
claim for the optional transition-refresh module.

The first completed recovery case, trash instance 311, scored **Q=1.0 with a
full success in 5,945 steps**. Both stock and the native selected policy scored
Q=0 and failed at 7,902 steps on that case. The log records one final-stage
5→4 backtrack at observation 5780, after three predictions of stage 4. The
original video fully decoded to 5,945 frames. This is one case from an incomplete
cell; the event and outcome do not establish general recovery performance or
prove that the backtrack caused success. It is excluded from the native
six-task aggregate. The scheduled shutdown canceled the remaining evaluation.
A post-cancellation storage inventory confirmed that instance 311 was the only
completed case; the interrupted second episode is excluded.

The [partial recovery CSV](behavior-final-stage-recovery-partial-cases-2026-09-20.csv)
has SHA-256 `069def6bc967ac2a35bab4f167a749cf4bfd84cff2af5b7ab4664eb9bea0307e`.
Its original metric SHA-256 is
`a72f191704dfe60fcbc550b73ec15952eca93142bd544ce0ef13d61d7e68162e`;
the video decode receipt is
`b8ca80c6c8f6123587b35c4408df0ec4a56516e9c5d8d3504e4379e320a5247c`.

## Changes needed before the next campaign

1. Keep native-to-serving typed-state, first-batch metric, and fixed-RNG action
   agreement as a gate before rollouts. The selected update-3599 export now
   passes that gate through an explicit, receipt-bound correlation asset.
2. Preserve final checkpoints and evidence as awaited workflow stages. Backup
   child processes can be terminated when their managed parent completes. Archive
   the selected full checkpoint first; scanning every earlier optimizer-state
   checkpoint delayed final-state preservation in this campaign.
3. Start parallel B200 scoring only after its prerequisite checkpoint and parity
   receipts exist. In this campaign five scorer GPUs waited while native serial
   selection had already finished.
4. Keep stock as the overall baseline after the completed matched comparison.
   Test targeted recovery changes and task-specific checkpoint choices on
   separately reserved cases before adopting them. The hot-dog gain motivates
   another test; selecting it after seeing this panel is not independent evidence.
5. Prove coverage of all required tasks and the challenge's serving-concurrency
   contract before presenting the pipeline as a submission.

## Validation

At code commit `b0989a385c0c3c9083a5ad33644b2a555b672d5e`, 38 focused tests,
3,927 guardrail tests, and 114 onboarding smoke tests passed locally. One
optional-dependency focused test skipped. The separate actual Flax 0.10.2/JAX
0.5.3 regression test passed. Ruff lint, formatting, documentation drift,
confidentiality, and secret scans passed.

The local full suite recorded 23,894 passed, 180 failed, 390 skipped, 13
deselected, and one unexpected pass. All 180 failures reproduced on the clean
base. The security subset recorded 764 passed, 25 failed, and ten skipped;
its 25 failures also reproduced on the base. These local runs were not clean
passes. The retained base-comparison audit and follow-up reproduction have
SHA-256 values `4b5cb03f219b445af606268540b193baa846d487f6ecf1b0a29061a00414ef83`
and `0193bbb476fa76afeaa7a84bcede71595f1b37226a91107c3dea65f442698160`.

All applicable GitHub checks passed for that code commit
([code CI run](https://github.com/nebius/nebius-physical-ai/actions/runs/35468721058)).
The subsequent four-cell report commit
`df0da2344dcfa1072069371000bc253301daa9c7` also passed all applicable checks:
twenty succeeded and two were skipped
([report CI run](https://github.com/nebius/nebius-physical-ai/actions/runs/35474121738)).
The measurements added after those commits do not change the public source code.
