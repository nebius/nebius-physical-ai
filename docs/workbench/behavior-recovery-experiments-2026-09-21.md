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
state, reward, instance segmentation, or report identifiers. Its completed
fresh-process comparison is reported below; it is not an official submission.

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
short-horizon RLC baseline and transition-refresh candidate each completed all
ten development instances, 311–320:

| Policy | Mean Q | Full successes | Mean simulator steps |
| --- | ---: | ---: | ---: |
| Stock checkpoint with shorter late-stage chunks | 0.333333 | 1/10 | 7,608.7 |
| Same checkpoint and chunks, refresh on accepted stage transitions | 0.500000 | 1/10 | 7,673.6 |

The aggregation stage downloaded and hashed every original metrics file and
video, and fully decoded all ten videos. The Workbench runner is pinned to
`8921f8f252d24485c4dfdde72c87bcc38eb4db70`, and the unchanged upstream evaluator
to `b1979916ec1549b10a4e65e630bc6504a9af1b00`. The panel is
`183e6e38981444fe9523a80edf5daff4c724c6e3cd8a20982f01b2108ebfb9cf`;
the candidate panel is
`feb9f776bf28a3c4755c63f5273b7c8f49b698549b4bb779ea8f6b02519d3a26`.

The complete paired comparison improves mean Q by 0.166667: six cases improve,
three tie, and one regresses. Full successes do not increase; the baseline
succeeds on instance 314 and the candidate on 315. Every original from both
panels was hash-verified, and all twenty videos were fully decoded before
comparison. The reusable comparison receipt digest is
`bc2aedc8083144631eada819d14ef4eaf22ca171319a47f8ec10ad55890755e5`.

This baseline includes an execution modification; it is not a measurement of
native RLC. Its score is separate from the historical 0.50 result above, whose
policy process lifecycle differed. These ten development cases are reused;
this result does not demonstrate generalization or a reliable full-success gain.
The released-baseline development comparison is now complete below. The
separate reporting gate remains outstanding; its cases have some historical
exposure.

### Native RLC development result

Native RLC completed the same ten development cases with mean **Q=0.466667**,
**2/10 full successes**, and **7,206.1 mean simulator steps**. Instances 314 and
320 succeeded; 312, 313, 316, and 319 scored Q=2/3; the other four scored zero.
The policy starts a fresh process for each episode and uses the unchanged
published execution settings and checkpoint 2.

All ten episodes completed before the managed CPU aggregation stage encountered
an image-pull failure. The owned CPU operator recovered the aggregate using the
same pinned Workbench implementation and PyAV 17.1.0. It downloaded and hashed
all original metrics and videos, fully decoded every video, and published the
immutable aggregate. No episodes were rerun. The verified aggregate artifact
SHA-256 is
`c4de9b9a13b15382800b77445f1043779c00cc20034840c4629ee3b8ef47ec02`.

The transition-refresh candidate has higher development Q (0.500000), but fewer
full successes (1/10). It therefore has not met the local task target requiring
both Q and full successes to match or exceed the strongest qualified baseline.
The separate reporting gate remains outstanding.

### Native stage-transition refresh development result

The native-timing variant keeps the released RLC checkpoint and native execution
settings. It discards the current action queue and resamples once only when the
unchanged native stage-voting method accepts a stage change. Correction-only
stage resets leave the native queue untouched.

On the same ten development cases, this variant reached mean **Q=0.466667**,
**2/10 full successes**, and **7,252.9 mean simulator steps**. This matches
native RLC on both selection metrics. Five cases improved and five worsened, so
the panel provides no development advantage over native execution. It does not
support a causal claim about stage-transition refresh.

Every original metrics file and video was downloaded and hashed, and all ten
videos were fully decoded. The managed workers and aggregate stage completed
without rerunning an episode despite an operator transport disconnect.
The durable workflow ledger subsequently confirmed `SUCCEEDED`; the exact
owned controller was then cleaned up and its API stopped. The verified
aggregate artifact is 12,313 bytes with SHA-256
`14cee110b276a87e16e7e500bd050ab5c6286b4df0f708410efff3a17c22dc68`;
its canonical aggregate digest is
`cb5c59f5ee5dcf0492d6619c34dfcc39624f5561c0daafc3b66010937b57964b`.
Reporting remains held pending selection confirmation. This reused development
panel is not an official score or evidence of challenge-wide competitiveness.

### Native Comet50 development result

The released Comet50 checkpoint completed the same ten development cases with
its native 32-action chunk, five-step replanning, and three-chunk ensemble.
Mean Q was **0.266667**, with **0/10 full successes**; all cases reached 7,902
simulator steps. Instances 314, 317, 318, and 320 scored Q=2/3; the other six
scored zero. Every original metrics file and video was hash-verified, and all
ten videos were fully decoded.

The checkpoint revision is `61739ffbced89dd5ba1b87c30d93d6084b79b0af`,
Comet source is `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`, and Workbench
source is `13f07569cd95207147be6c5ed5f166236c88b1d6`. The verified aggregate
artifact SHA-256 is
`ceb168bd42c84894b12bc63a02b2a403985634a854b6fb2ebabe04770da6ba48`.
This result ranks below the transition-refresh candidate on development Q and
full successes. Native RLC remains the stronger baseline. This result does not
establish challenge-wide competitiveness or the official policy-GPU limit.

### Native Comet12 development result

The released Comet12 checkpoint completed the same ten development cases with
mean **Q=0.400000**, **2/10 full successes**, and **7,504.7 mean simulator steps**.
Instances 318 and 320 succeeded; 312 and 317 scored Q=2/3; 313 and 319 scored
Q=1/3; the other four scored zero. It uses the native 32-action chunk, five-step
replanning, three-chunk ensemble, and a fresh policy process for every episode.

A separate legacy submission changed the shared staged-source setting while
this panel was running. The controller's strict configuration-identity check
then stopped supervision; its attempted cancellation failed on the same check.
All episode workers continued and completed. The owned CPU operator recovered
the complete aggregate through the frozen Workbench implementation. Every
original metric file and video was downloaded and hashed, every video fully
decoded, and the published aggregate read back and verified. No episodes were
rerun. The managed workflow itself is not recorded as successful.

The checkpoint revision is `a3d85eb978b58501c99f6c927a18d52ec6c1532c`,
and Comet source is `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`.
The verified aggregate artifact SHA-256 is
`8dcaca605801685df8157609e7874fc089d7b66870ecc05f88156569b0459546`.
Native RLC remains the strongest completed baseline on both selection metrics.
Comet12 is the stronger of the two released Comet training parents; this result
alone does not establish a fine-tuning gain or pass the separate reporting gate.

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

### Comet training-data qualification

The CPU qualification at Workbench commit
`a716c5c41370be9a6b9d18db5b1384b6c3668860` succeeded with the pinned Comet
source and verified 217-distribution runtime. It imported the native training
configuration, loader, and training module, then decoded a real 256-example
batch from the 180 training episodes. The batch contains three 224×224 RGB
views, 32×32 actions, and 200-token prompts. Source inventories, normalization,
the held-out split, and terminal action padding passed their checks.

The qualification receipt has SHA-256
`9b82b9e26131b02ce3b9327a0d0dcccd84f41c5a9e17be93168222b04616b698`.
All three original preparation, qualification, and log artifacts were downloaded
and verified before controller cleanup. This check loaded no model, created no
optimizer, and performed no training. A native B200 optimizer update and the
future eight-worker data-loader throughput remain separate qualification gates.

The subsequent B200 qualification stopped before loading the model or creating
an optimizer: its recreated batch did not match the CPU qualification receipt.
The failure diagnostic and all four original artifacts were downloaded and
hash-verified before controller cleanup. Diagnostic SHA-256:
`fc5f46002bd9b8f03d31cd3238b6173201ddc379f5f45366e86b592de484442c`.
No optimizer update or training checkpoint was produced. The mismatch requires
array-level and backend diagnostics before another qualification attempt.

The second attempt supplied those diagnostics. Actions, state, image masks,
and tokenized prompts matched the CPU reference exactly; all three RGB leaves
differed. The process used the B200 backend, while the reference used CPU.
This narrows the mismatch to RGB data; its numerical cause remains unproven.
The five original artifacts and failure manifest were
downloaded and verified before controller cleanup. The array diagnostic has
SHA-256 `687c01a8f98f10109a301f27c9ac0ac9b9f0f07b45e300737686ead62b659806`.

A third qualification explicitly runs data transforms on CPU, retains the
exact reference-batch gate, and places the accepted batch on the B200 before
the unchanged native optimizer step. It also compares CPU and GPU transforms
of the same decoded sample. Its 35 local checks and independent package review
passed, but the live diagnostic then attempted to slice an optional scalar
observation field as a batch array and stopped before optimizer creation.
All four failure originals were downloaded and hash-verified. The diagnostic
needs to preserve native optional fields. The future eight-worker loader still
requires separate throughput and parity evidence.

A fourth qualification uses the corrected optional-field handling. Its native
observation regression passed in the pinned training interpreter before the
batch check. The three RGB leaves still differed from the earlier CPU receipt,
so it stopped before optimizer creation. The saved diagnostic shows identical
outputs for its first-sample comparison; this does not prove that both paths
used the intended device at every operation. The loader's explicit device
placement is under investigation. No training checkpoint was produced.

### Subsequent native training and scorer qualification

Later probes resolved the batch-placement failures above. The real loader
verified 3,072 sample identities across twelve batches of 256 with eight spawned
workers. A B200 probe completed a native full-model optimizer update, with all
3.35 billion parameters trainable and 49 of 51 parameter leaves changed. That
updated state was discarded. These checks establish execution, not policy quality
or sustained training throughput.

A fresh-process resume probe restored the saved state and next batch exactly.
Its continuation matched native parameters and reported metrics but differed in
four AdamW moment arrays, so resume admission stayed closed. A subsequent probe
excluded nondeterministic compiler operations, completed two native updates, and
saved its checkpoint. A separate eager gradient diagnostic then requested another
59.35 GiB and ran out of memory before the uninterrupted third update or fresh
restore comparison. All 17 original failure artifacts were independently verified;
that checkpoint remains unqualified.

The next probe removed the duplicate eager gradient computation. After two native
updates, it saved the complete FP32 training state, then compared uninterrupted
update three with a fresh-process restore and update three. The next batch,
folded random key, native loss and norm metrics, model parameters, and complete
AdamW state matched exactly. The outer wrapper nevertheless failed because it
looked for GPU visibility in the boundary receipt instead of the cursor receipt.
The original run remains failed; its 23 original artifacts preserve both the
successful numerical comparison and the wrapper failure.

A separate CPU recovery workflow verified that evidence and published the retained
checkpoint without rerunning training. Independent provider readback hashed all
35 checkpoint files, totaling 31,561,938,241 bytes, and reverified all original
evidence. This admits the recovered native-resume result under its recorded
compiler configuration. It does not fabricate the missing wrapper receipt or
qualify serving exports, later checkpoints, sustained throughput, or policy Q.

Two full parent holdout passes exceeded the frozen repeatability tolerance.
A later diagnostic checked 60 examples across the same three tasks in two fresh
processes. All transformed inputs, folded random keys, and cross-process loss
bytes matched; all 120 within-process repeated loss calls also matched, with
unchanged BF16 model state. Its 14 original artifacts were independently verified.
The subsequent instrumented full check reproduced all 3,840 transformed inputs,
folded random keys, and loss bytes across two fresh processes, with unchanged
BF16 state and zero task-level loss differences. All 14 original artifacts were
independently verified. Its per-example hashing and host synchronization change
execution timing, so that result qualifies only the instrumented diagnostic.

The direct production-scorer check then failed before producing scores: its
full-data path imported an omitted inventory module. All 12 original failure
artifacts were preserved and independently verified. The repair supplies that
dependency and exercises the full-data import path. The corrected direct check
completed both fresh processes over all 3,840 examples. Independent readback
verified its 18 original artifacts and reproduced the frozen selection gate:
each task and the combined anchor had zero aggregate loss difference. The
production worker, selection rule, and tolerance remained unchanged. The output
manifest has SHA-256
`2324978271323beb719aaf17b8cc6ade60af747f6766e94092bcd847c481bf22`.
This qualifies the direct production scorer for the recorded parent, inputs,
runtime, and hardware. Serving parity remains a separate gate. No trained
candidate has been scored or admitted to rollouts.

The pretraining serving-parity check reconstructed a real fixed observation,
verified the recovered FP32 checkpoint, and prepared a separate BF16 export.
It then failed before policy inference: the fresh worker passed the prepared
runtime contract to a validator expecting the original preparation template.
All 12 published failure artifacts were independently verified. The retained
checkpoint and export remain available for recovery; this failure supplies no
action-parity result and does not admit training or rollouts.

A separate CPU recovery verified the retained preparation and preserved that
failure. The corrected serving check then loaded the recovered native parameters
through the production BF16 serving path and loaded the BF16 export. Both cold
GPU processes produced byte-identical native-derived and exported actions for
the same real observation and fixed noise; the float32 output shape was
`[32, 32]`. Actions also matched across the two processes. Independent provider
readback and local verification covered all 23 original artifacts, including the
raw action arrays. The output manifest has SHA-256
`7a5b573ffc1076bbbfcd1d373713f279d7b4b09434ae2502253d2a1572d9bf00`.
This qualifies the recorded pretraining serving path. It does not qualify a
later training milestone, the single-24GB deployment requirement, or policy Q.

After complete provider verification, a separate CPU workflow retired the two
obsolete local resume-probe checkpoints and recovered 63,123,878,368 bytes of
training-volume space. Independent readback verified its ten original artifacts.
The checkpoint admitted through recovery remains preserved in object storage.

### Specialist recovery and Workbench task status

The released SFT specialist ultimately completed all ten original development
cases without rerunning an episode. Verification covered 130 original files,
including full decoding of every video. It scored mean Q=0.166667, with 0/10
full successes and 7,902 mean simulator steps. Native RLC scored mean Q=0.466667
with 2/10 full successes on the same development cohort, so the specialist was
rejected before reporting. No specialist report candidate was frozen, and the
positive report materializer did not run. This is a development comparison,
not an official score or evidence that the fine-tuning method caused the gap.
The complete-panel verification evidence has SHA-256
`acf5494ace7e6094d273820e67ef77f495ba831d68d4c169ff72ac31d0c12076`.

Earlier, two workers rejected a loaded correlation-state mismatch before
evaluator startup, leaving six cases unstarted while two other workers completed
four cases. This infrastructure failure exercised Workbench's task-activity
reporting. A failed
parallel workflow can retain running sibling tasks. Status now reports active
and unresolved stage keys separately from workflow outcome and requires unique,
complete scheduler observations before declaring all tasks terminal. Extra or
duplicate rows cannot establish completion. Tests cover the mixed active case;
a read-only Linux live regression passed after the original siblings finished.
It verified all four terminal task states while retaining `UNKNOWN` for the
conflicting workflow outcomes. The exact controller cleanup was then verified.
Live status evidence SHA-256:
`3777ec17916b8884a8d87fdceb45086e79e6e92a386b5ae46cbcf3c58fb07323`.

The recovery retained the first four originals and evaluated only the six
previously unstarted cases. All 52 files from those four cases remained
byte-identical within the final 130-file panel. The final verification
reconstructed the complete panel from both sets of original artifacts while
preserving the strict loaded state check.

### Report admission and singleton task status

The native RLC, Comet12, and Comet50 baseline report workflows completed, and
their exact controller cleanup was verified. Their scores remain sealed and
unread. No candidate has been frozen, no baseline output has been unsealed, and
no positive report materializer has run. The protocol first freezes a candidate
from development evidence, then unseals the three same-cohort baselines to
select the reporting reference before any candidate report run.

The completed Comet12 workflow exposed a task-name attribution gap: SkyPilot
named its serial aggregate task after the managed job. Workbench now accepts
that name only for a singleton and only when it matches the recorded job ID
and name. A read-only Linux regression verified all five tasks terminal with
no unresolved observations. Evidence SHA-256:
`89baa614586d81f9843ea572c0545c100eb235e9088d33900dd81cac8e8a8a2b`.

Specialist report admission now validates receipt bytes and structure before
loading checkpoints. Valid receipt envelopes still require the complete
development-selection, sealed-baseline, unseal, target-selection, and runtime
evidence chain. A committed Linux live test passed against real S3 declarations:
both missing and malformed authorization stopped before case state, policy, or
evaluator startup, and the fresh output prefix remained empty. This tests the
negative entry gates; positive report admission awaits real campaign evidence.
Evidence SHA-256:
`d5d865377703d77823b45c6808b5ef7378f092be026a00467ded13b0d4552ed7`.

### Sampled video observations

Eight evenly spaced frames from each of two hash-verified original videos show
different execution paths. In instance 311 (Q=0), early frames show the robot
near an open refrigerator; later frames show a can still held outside the bin.
Instance 314 (Q=1) reaches the living room earlier and completes the task.
These sampled views suggest inspecting navigation, initial-state disambiguation,
and depositing objects. They do not establish a causal failure classification.
Training should retain full demonstration trajectories; neither these development
videos nor their cases enter the training dataset.
