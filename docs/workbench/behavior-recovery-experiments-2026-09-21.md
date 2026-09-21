# BEHAVIOR recovery experiments — September 21, 2026

The previous complete six-task development comparison regressed from stock
**Q=0.324028** to fine-tuned **Q=0.298750**. Stock remains the reference policy.
See the [completed results](behavior-matched-results-2026-09-19.md) for per-task
scores, partial diagnostics, and the model-state dtype difference.

This document records the frozen experiment choices and their current evidence.
The five-arm simulator comparison and anchored training completed. The anchored
export has not passed GPU qualification, and no new candidate has passed the
separate reporting gate.

## Recent research reviewed

[Behavior-Skill](https://arxiv.org/abs/2608.30536), published August 31, evaluates
constituent skills and identifies persistent contact-manipulation bottlenecks.
This supports examining where execution fails, although our trajectory time bins
do not reproduce its semantic skill annotations.
[Evidence-Gated Regularization](https://arxiv.org/abs/2609.03142), published
September 2, targets dependence between sensors and robustness to corruption.
Our inference is that it deserves a separate camera-sensitivity experiment; the
current regression alone does not establish that sensor dependence is its cause.

## Fixed-weight execution comparison

Five arms run the same ten public development instances of picking up trash:

| Model | Execution rule | Question |
| --- | --- | --- |
| Stock checkpoint 2 | Native | What is the contemporaneous baseline? |
| Stock checkpoint 2 | Final-stage backward vote | Does allowing consistent earlier-stage predictions help? |
| Stock checkpoint 2 | Shorter late-stage action chunks | Does more frequent replanning help near completion? |
| Previous selected fine-tune | Native | Does the retained candidate reproduce its development behavior? |
| Previous selected fine-tune | Final-stage backward vote | Does the promising one-case recovery repeat across the panel? |

The backward-vote rule requires three unanimous predictions for an earlier stage
while the controller is in its final stage. It preserves native action queues.
The chunk variant changes the next chunk at prediction boundaries, using shorter
chunks for the last two predicted stages. It never truncates an active queue.
Both use policy predictions and permitted onboard observations.

The completed comparison ranked selected plus final-stage backtracking first on
mean Q, but it did not isolate backtracking as the cause. That arm backtracked
only in cases 314 and 319; gains occurred in several cases without a backtrack,
case 319 regressed, and case 320 succeeded without a backtrack. The adaptive
stock arm reached mean Q 0.50 with no full successes. Its telemetry recorded 88
accepted stage transitions and no action-queue refresh at those boundaries.

The next one-factor development arm is
`adaptive-short-chunk-transition-refresh`. It retains the stock checkpoint and
the adaptive horizon settings. When native stage voting accepts a new stage, it
discards the remaining actions and inpainting prefix produced for the old stage,
then performs one bounded resample from the new stage. A second transition during
that resample fails closed instead of recursing. The intervention uses the
policy's stage state and permitted observations; it does not read simulator goal
state, reward, instance segmentation, or report identifiers. Its first rollout
is experimental and cannot be described as an official score or an established
improvement.

[AAC](https://arxiv.org/abs/2604.04161) adapts chunk size using action entropy.
[DEHP](https://arxiv.org/abs/2606.11408), revised July 2026, learns a horizon
predictor while keeping the action policy frozen. These motivate testing execution
horizon independently of weights. Our deterministic stage rule is a simpler
hypothesis; it does not reproduce either paper's algorithm.

All five complete panels are required for selection. Rank by mean Q, then full
successes, then fewer steps. Prefer stock native on an exact tie; otherwise use
the predeclared arm order. Promote only a strict mean-Q gain over contemporaneous
stock native. The fine-tuned pair also provides a within-model comparison.

## Training with a stock-policy anchor

The [objective and frozen recipe](behavior-stock-anchored-training.md) document
the training math and its public CPU checks.

Start again from stock checkpoint 2. Train its 23 action-expert parameter leaves
on the original task-balanced training split for radio, trash, and shoes. Keep
the stage classifier frozen and preserve the action-correlation intermediate in
its canonical FP32 dtype. Add a teacher penalty to the demonstration flow loss:

```text
loss = demonstration_loss + mean((student_velocity - frozen_stock_velocity)^2)
```

Teacher and student receive identical observation augmentation, flow time, noise,
and stage conditioning. The penalty weight is 1. Use 2,400 updates, batch size 16,
four flow samples, and a learning rate decreasing from 2e-6 to 2e-7 after a
400-step warmup. This also changes the previous learning rate and schedule, so
any gain cannot be attributed to the anchor alone.

[RLT](https://www.pi.website/research/rlt) motivates preserving a capable base
policy while learning targeted changes. Our supervised flow distillation has
neither RLT's token architecture nor online actor–critic training.
[RECALL](https://arxiv.org/abs/2606.23617) documents the forgetting risk of
recovery-only adaptation. We use existing demonstrations and have not collected
the recovery supervision needed to reproduce that method.

Before training, a GPU check must verify matching initial teacher/student state,
zero initial anchor loss, an unchanged teacher, and updates confined to the 23
allowed parameter leaves and their moving averages. It records actual initialized
teacher dtypes and memory sizes.

## Checkpoint and reporting gates

Evaluate saved updates 400, 800, 1200, 1600, 2000, and 2399 against stock using
fixed held-out traces and random seeds. Each task must satisfy both checks:

- Uniform held-out flow loss is at most 1.02 times stock.
- The non-late action proxy is at most 1.005 times stock.

Among eligible checkpoints, select the greatest equal-task late-proxy improvement,
breaking ties toward the earlier update. If none qualify, retain stock. Time bins
identify trajectory portions; they are not semantic contact labels.

The selected export must reproduce the native checkpoint's typed model state and
fixed-seed actions after reloading. Simulator development evaluation then decides
whether it improves mean Q across all three tasks. Radio and shoes have historical
stock comparisons; trash also has a contemporaneous stock run.

### Completed training and failed export check

The anchored run completed 2,400 updates. Stock and six saved checkpoints were
scored on 102,403 held-out rows. Selected step 2399 reduced equal-task action loss
from 0.333278408726 to 0.223172082376, a 33.04% reduction; all three trained tasks
improved this offline metric. This does not establish a simulator Q improvement.

The subsequent B200 qualification verified the frozen runtime, parent checkpoint,
dataset, traces, and archived selected-model bytes. In a fresh process, the
reloaded policy's actions differed from the historical export hash, failing with
`fixed-seed action bytes differ`. The earlier exporter retained hashes but omitted
the exact transformed observation and raw action arrays. Consequently, this
check cannot distinguish reconstructed-input differences from numerical
differences across compilations. The failure log and input identities were
preserved; this export remains unqualified for rollout serving.

The next diagnostic saves the transformed observation and raw outputs before
asserting equality, then compares independent loads using those same observation
bytes. It retains the historical mismatch explicitly. Passing that repeatability
check alone would not recover the missing historical native-output evidence.

The first attempt at that diagnostic stopped before recording observations:
NumPy cannot directly serialize a typed JAX random key. Its four original
diagnostic files were recovered and hash-verified. This is a harness failure;
it supplies no new action-repeatability or export-parity result.

The corrected diagnostic completed both fresh-process inference phases and
preserved all 16 original files. The saved and independently reconstructed
observation archives are byte-identical. Same-load repeats passed, but action
bytes differed between independent loads, so final qualification failed.
Raw action arrays, typed random-key records, model-state inventories, and logs
are retained for numerical analysis. This narrows the failure beyond input
reconstruction; it does not resolve its cause or qualify the export.

Development indices 10–19 are reused and are not unseen data. Reporting indices
0–9 remain separate until the choice is fixed. Keep interrupted panels and failed
runs visible; do not select lucky reruns. The official evaluator supplies Q under
the [challenge rules](https://behavior.stanford.edu/challenge/evaluation.html).
Neither this small panel nor lower held-out loss establishes a competitive score
across all 100 challenge tasks.

## Fresh-process trash comparison

The new campaign starts a separate policy process for every episode. Its frozen
short-horizon RLC baseline completed all ten development instances, 311–320:

| Policy | Mean Q | Full successes | Mean simulator steps |
| --- | ---: | ---: | ---: |
| Stock checkpoint with shorter late-stage chunks | 0.333333 | 1/10 | 7,608.7 |

The aggregation stage downloaded and hashed every original metrics file and
video, and fully decoded all ten videos. The Workbench runner is pinned to
`8921f8f252d24485c4dfdde72c87bcc38eb4db70`, and the unchanged upstream evaluator
to `b1979916ec1549b10a4e65e630bc6504a9af1b00`. The panel is
`183e6e38981444fe9523a80edf5daff4c724c6e3cd8a20982f01b2108ebfb9cf`.

This baseline includes an execution modification; it is not a measurement of
native RLC. Its score is separate from the historical 0.50 result above, whose
policy process lifecycle differed. The transition-refresh candidate is running.
Native RLC and both released Comet checkpoints still need the same evaluation
before the baseline tournament and independent reporting gate can be completed.

### Released Comet checkpoint qualification

The shared Comet12/Comet50 adapter at Workbench commit
`13f07569cd95207147be6c5ed5f166236c88b1d6` passed a B200 synthetic-action check.
Each checkpoint produced identical finite 23-action vectors across two fresh
processes on the same frozen observation. Comet12 also reproduced its earlier
reference action bytes. All 33 original output files were downloaded and
hash-verified against the published manifest.

This establishes loader and adapter repeatability. It does not run the
simulator, measure task Q, or prove the official 24GB policy-GPU requirement.
The next comparison uses the complete task panels described above.

### Sampled video observations

Eight evenly spaced frames from each of two hash-verified original videos show
different execution paths. In instance 311 (Q=0), early frames show the robot
near an open refrigerator; later frames show a can still held outside the bin.
Instance 314 (Q=1) reaches the living room earlier and completes the task.
These sampled views suggest inspecting navigation, initial-state disambiguation,
and depositing objects. They do not establish a causal failure classification.
Training should retain full demonstration trajectories; neither these development
videos nor their cases enter the training dataset.
