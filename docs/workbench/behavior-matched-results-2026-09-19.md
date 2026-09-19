# BEHAVIOR matched training and development evaluation — September 19, 2026

Both matched training arms completed **3,600 optimizer updates**, scored their
six retained checkpoints on the frozen holdout, and exported a selected model.
That is **7,200 new planned optimizer updates**. The replay arm selected update
3599 with equal-task mean held-out action loss **0.160986**. This is an offline
training metric; it is not the challenge's task-completion score. The teacher
control also selected update 3599, with its own conditioning loss **0.160796**.

No matched-candidate rollout was completed. The previous
balanced fine-tune regressed, as documented in the
[earlier comparison](behavior-followup-results-2026-09-18.md). The matched
experiment changes the training approach in response to that result; it does
not erase it or establish an improvement by itself.

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
exact ties in favor of the earlier checkpoint. The replay arm remains the
primary deployment candidate; losses under the two different conditioning
regimes do not establish a cross-arm winner.

Selection receipt hashes are `c08fd339dd85750930c7be7198204d4f441fad2bda5c1c7a1ce6337d29009072`
(replay) and `b820c464a2f2ec883b47bc0181eebfde32a6be6688386fffe1babea57cc980fe`
(teacher). Both export receipts and the underlying holdout rows are retained
with the private run evidence.

See the [training guide](behavior-matched-training.md) for executable commands
and the exact training contract.

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
4. Run the admitted replay candidate against the fixed stock cases on RTX GPUs,
   with the same session/reset behavior. Use task Q and complete-case coverage
   to decide whether it improves; training loss cannot make that decision.
5. Prove coverage of all required tasks and the challenge's serving-concurrency
   contract before presenting the pipeline as a submission.

## Validation

At commit `20088841e6be5b1ee9878c41f3c86f7ed756c53e`, after merging the latest
main branch, 167 focused tests and 3,920 guardrail tests passed locally. Three
optional-dependency tests skipped. The separate actual Flax 0.10.2/JAX 0.5.3
regression test passed. Ruff lint, repository formatting, confidentiality, and
secret scans passed. The earlier onboarding smoke passed all 114 tests.

The latest local macOS security run recorded 764 passed, 25 failed, and 10
skipped. Three failures concern GNU `mv -T` on macOS; the others concern local
SkyPilot process identity. Representative failures also reproduced on the prior
base. This is not a clean security-suite result. The earlier local Linux full
suite also retained 30 environment-related failures.

All applicable GitHub checks passed for code commit
`20088841e6be5b1ee9878c41f3c86f7ed756c53e`, including all five full Python test
shards, coverage, security checks, and browser/compatibility checks
([CI run](https://github.com/nebius/nebius-physical-ai/actions/runs/35419007216)).
This report adds the final measurements and their limits after that code run.
