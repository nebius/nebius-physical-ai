# Improving the BEHAVIOR policy with recent research

[Challenge workflow](behavior-challenge.md)

The current RLC controller already uses a π0.5-derived model. The
[September 17 experiment](behavior-experiment-results-2026-09-17.md) completed
80 development rollouts and three 6,000-update fine-tunes. The evaluated
fine-tune and shorter action execution showed no improvement over the campaign's
stock controls. The [complete 60-rollout follow-up](behavior-followup-results-2026-09-18.md)
also found a regression for the balanced three-task fine-tune: Q=0.227778 versus
stock Q=0.404444. The radio press-phase checkpoint remains unevaluated, and
Meta100 startup validation is in progress. No competitive full-challenge score is
established.

The earlier September 16 radio comparison remains recorded at RLC Q=0.20 versus
the official baseline's Q=0.10. The EGR objective described below has no trained
and evaluated checkpoint.

## Research reviewed

Reviewed September 16, 2026 using author papers, repositories and model releases.

| Research | Relevant finding | Decision for this integration |
| --- | --- | --- |
| [EMERGE-Policy, September 2026 revision](https://arxiv.org/html/2608.29896v2) | Combines policy execution, outcome verification, memory and local recovery. Its LIBERO setup adds five cameras and increases episode step limits. | Study observation-based recovery, but retain BEHAVIOR's permitted sensors and default timeout. Its reported gains do not establish gains under this challenge protocol. |
| [Evidence-Gated Regularization, September 2026](https://arxiv.org/abs/2609.03142) | A π0.5 training objective addresses distraction by irrelevant cameras and dependence on cameras that become occluded. | Implement camera evidence and importance-weighted consistency losses; measure the resulting policy on the unchanged evaluator. |
| [LeRobot π0.5 memory implementation, August 2026](https://github.com/huggingface/lerobot/commit/8ae2354592ec4480c38134fb75350b8e7992e0e9) | Releases optional recent visual and proprioceptive history. | Compare memory-enabled training with its matched single-frame control. This is a separate PyTorch candidate; the flags do not enable memory in the current JAX RLC controller. |
| [SERF, June 2026](https://arxiv.org/abs/2606.12956), [released code](https://github.com/ExistentialRobotics/SERF-VLA) | Spatial and temporal feature maps provide scene memory. Its formulation includes robot base pose and camera poses. | A challenge candidate needs pose estimated from permitted sensors. Do not connect simulator global pose to the released mapper. |
| [ForesightFlow, June 2026](https://arxiv.org/abs/2606.04968) | Jointly generated action and success-potential coordinates support learning from mixed-quality trajectories and selecting action chunks. | Requires additional training-instance rollouts containing both successes and failures. The existing development rollouts are evaluation evidence, not training data. |
| [π0.7, April 2026](https://www.pi.website/blog/pi07) | Diverse conditioning supports compositional behavior and strategy control. | The checked [OpenPI release](https://github.com/Physical-Intelligence/openpi) supplies π0.5 models. This integration does not claim to run π0.7. |
| [Multi-scale Embodied Memory, March 2026](https://www.pi.website/research/memory) | Combines recent visual history with longer-term language memory. | Separate short visual memory, which needs no semantic annotation join, from long-term subtask memory. Elapsed time is insufficient evidence that a subtask finished. |
| [OpenPI Comet](https://arxiv.org/abs/2512.10071) | Studies π0.5 training and data choices on the 2025 BEHAVIOR tasks. | Keep matched controls and separate general training from task adaptation. Its 2025 score is not a 2026 comparison. |

These papers use different tasks, data, resets and evaluation protocols. Their
reported percentages cannot be combined into an expected challenge score.

## First candidate: improve camera use during training

The [camera evidence module](../../npa/src/npa/workflows/behavior_challenge/camera_evidence.py)
implements the visibility gates, hinge weights and importance multipliers needed
for EGR. A frame with no visible task object receives no camera regularization.
A visible table alone cannot make a camera informative when the manipulated
object is absent.

Training compares a clean flow-velocity prediction with two corrupted views:
one erases part of a low-evidence camera; the other preserves a high-evidence
camera while corrupting the others. Both predictions must use identical flow
noise, time and ordinary augmentation. The consistency loss excludes padded
action coordinates and retains gradients through both predictions. Camera
selection is weighted by evidence, with the corresponding importance multiplier.

```python
from npa.workflows.behavior_challenge.camera_evidence import (
    camera_weights,
    consistency_loss,
    sample_camera,
)

# Areas come from frame-aligned training replay, ordered head/left wrist/right wrist.
invariance, sufficiency = camera_weights(focal_area, interaction_area)
camera, multiplier = sample_camera(invariance, random_generator)
# The trainer constructs the selected corruption and shares noise/time with clean.
loss = consistency_loss(
    clean_velocity, corrupted_velocity, multiplier, action_dimensions=23,
)
```

The defaults (`low=0.2`, `high=0.8`, `focal=0.0001`,
`global_visibility=0.001`, `interaction_weight=0.5`) are experimental settings,
not a claim to reproduce the authors' complete training recipe. Configure them
with `EvidenceThresholds`; freeze settings before development evaluation.
The helper does not itself train, serve or improve a checkpoint.

The optional JAX [training objective](../../npa/src/npa/workflows/behavior_challenge/camera_consistency.py)
runs the clean prediction and two camera corruptions. Pass the trainer's
differentiable velocity callback and an explicit camera order:

```python
from npa.workflows.behavior_challenge.camera_consistency import regularization_losses

# weights has shape [batch, 2, cameras], ordered invariance then sufficiency.
invariance_loss, sufficiency_loss = regularization_losses(
    predict_velocity, images, weights, training_key,
    camera_names=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    action_dimensions=23,
)
loss = imitation_loss + 0.1 * invariance_loss + 0.1 * sufficiency_loss
```

The callback receives the same prediction key for all three calls and must use
it for flow noise, time and ordinary augmentation. Erased rectangles span
20–60% of each image dimension and use zero in normalized pixel space. These
sizes and loss coefficients are experimental choices. The objective requires
JAX in the policy environment. Tests compile the real objective, differentiate
it, verify padding exclusion and preserve an informative wrist camera even
when input dictionary ordering changes. Full-model training remains a separate
validation step.

An abstract backward-pass check also reaches both losses through the pinned
RLC model: 3,405,453,735 total parameters and 477,001,399 trainable parameters
at batch size 16. That check uses real demonstration RGB/actions with synthetic
evidence weights to validate graph compatibility. It allocates no trained
checkpoint and does not establish finite numerical gradients in the full model.

Privileged visibility labels are training-only supervision. Serving keeps the
existing RGB/proprioception input filter and requires no segmentation or
success-state input. The [challenge rules](https://behavior.stanford.edu/challenge/evaluation.html)
permit privileged information during training.

## Data findings that change the experiment

The earlier motion-adaptation experiment stopped at step 855 with a
`FrameTimestampError`; it produced no evaluated trained policy. An audit of all
**1,289,784 RGB queries** across the 200 radio demonstrations found 34 queries
outside the previous 0.1 ms tolerance. The largest error was **0.1220703125 ms**.
The new dataset adapter uses **0.5 ms**, matching the pinned official OpenPI
baseline setting and remaining far below one 33.3 ms frame interval. The audit
checks MP4 presentation timestamps with the decoder's float32 conversion; it
does not establish image-content alignment by itself. No samples are dropped.

All 200 downloaded raw recordings have actions identical to their corresponding
released LeRobot episodes. The first audit found that only **9/200** annotation
`task_duration` values equaled stored episode lengths. At that point the fields'
meaning and a direct annotation-to-frame join were unverified, so the training
recipe did not use those annotations.

A later training-only audit resolved that warning for the frozen 180-episode
radio training split. All 180 annotation files matched their expected hashes,
and all 180 raw action sequences were byte-identical to the released LeRobot
episodes. In every training episode, `task_duration` equals the annotated
`valid_duration` length; stored episodes may also contain leading or trailing
frames. The earlier 9/200 count compared those different quantities and was not
itself evidence of a timebase error. This later audit did not recheck all 200
episodes.

Across all 180 training episodes, skill sequences are ordered, nonoverlapping,
and inside `valid_duration`; every `press` interval is inside that same interval
on the released frame/action timebase. A decoded review checked 54 frames and 27 action
rows from episodes 0, 5, 10, and 164 in two released RGB views. Its maximum video
timestamp error was 2.73e-11 frames, and all four intervals covered radio
manipulation in the expected task phase. The intervals are broad: they contain
86–1,075 frames, with a median of 267 at 30 Hz, and reviewed examples include
positioning or withdrawal. They support the description **semantic press-phase
oversampling**, not physical-contact labels or an exact press instant. No
holdout, development, or reporting episode was inspected in this audit.

Direct robot-camera segmentation replay encountered the upstream failure in
[BEHAVIOR issue #2312](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2312).
The training-only viewer-camera workaround also crashed in the simulator's
instance-segmentation renderer after a camera-argument type correction. Neither
path produced verified labels. EGR training is blocked on obtaining camera
evidence that passes source-action, observation and image alignment checks;
no EGR checkpoint has been trained or evaluated. Failed-run logs are retained,
and the failed experiment's compute is retired through Workbench cleanup.

## Later audit of the completed fine-tuning recipe

The three September 17 fine-tunes used the pinned native RLC trainer. The
balanced task-panel run applied 6,000 updates at batch size 16: 96,000 samples,
with exactly 32,000 samples from each of tasks 0, 1, and 22. These starting-frame
counts equal 8.29%, 3.35%, and 2.29% of the available training positions,
respectively. Each sample also contains a 30-action target window, so these
ratios do not measure distinct action-target coverage. Its learning rate warms from
approximately 1e-8 to 1e-5 over 1,000 updates and then cosine-decays to 1e-6.

The native filter trains 477,001,399 of 3,405,453,735 parameters: the action
expert and non-frozen action, KV, task, and stage modules. It freezes the vision
backbone, base language expert, and FAST modules. Training and serving use the
same 30-action horizon, state/action transforms, per-timestep normalization,
and published normalization bytes. The audit found no concrete freeze, adapter,
or normalization defect.

It did find a train/serve mismatch in stage conditioning. Training divides each
stored episode into equal-duration bins and supplies the resulting stage to the
action path. Serving supplies the wrapper's stateful predicted stage. These bins
are elapsed-time supervision, not semantic skill annotations. The model's
attention mask excludes the supplied stage labels from the classifier features.
Training used one flow draw per example, compared with 15 in the native base
recipe, and retained only the final checkpoint without measuring the reserved
holdout loss. These are
plausible generalization and variance risks; they do not establish why the
uniform radio fine-tune failed to improve its matched control.

A prospective fine-tune should retain intermediate checkpoints and select them
with a deterministic loss pass over the reserved training-distribution holdout.
The following audits make two proposed ablations concrete. Neither has been
trained, and neither establishes a performance gain.

### Separate action adaptation from task and stage adaptation

The completed balanced run used `use_knowledge_insulation=False`. Its action
loss could therefore propagate through the prefix cache into task and
stage-fusion modules, while stage cross-entropy trained the stage head. A CPU
partial restore of the actual parent and adapted checkpoints confirmed changes
in the stage head and task-embedding rows 0, 1, and 22. These are real parameter
changes; their effect on the rollout failures has not been isolated.

A narrower prospective partition contains 23 parameter leaves and 430,108,328
scalars: the action expert, action input/output projections, time MLPs, and
`kv_transform`. Enabling knowledge insulation stops action gradients at the
prefix cache, while gradients still reach `kv_transform`. Freezing the task and
stage modules enforces their invariance and removes their optimizer state.
The partition and gradient-routing checks used CPU abstract graphs and small
optimizer sentinels. They did not perform full-model GPU training.

The first matched pair would use that partition, knowledge insulation, and zero
stage/FAST auxiliary loss weights. Arm T supplies native equal-time stage bins.
Arm P replays frozen parent predictions chronologically through the native
three-prediction history and transition filter. It supplies the stage used for
the current action chunk, before that observation's prediction updates the
filter. Raw classifier argmax is not equivalent to this serving behavior.
Images, proprioception, actions, frame order, optimizer, schedule, flow draws,
seed, and evaluation cases remain matched. This isolates the training condition
supplied to the action model. Replaying frozen parent predictions is still an
offline approximation to the observations a new policy will encounter.

### Test semantic phase sampling with a matched control

A training-only audit also checked 180 trash and 180 shoe annotations. It found
540 `place in` intervals for trash and 720 for shoes, all ordered,
nonoverlapping, and within the released valid-duration bounds. Their median
durations were 214.5 and 391 frames, respectively, at 30 Hz. A decoded review
used four training episodes, 48 images from the left wrist and head-mounted ZED
cameras, and the corresponding action records. Together with the radio audit,
these support broad semantic phase labels. They do not identify exact physical
contact, insertion completion, or release frames.

A prepared three-task sampler would draw half of each task's starting positions
uniformly from stored training frames and half from `press` (radio) or `place in`
(trash and shoes) intervals. Its matched control uses two independent uniform
streams. Both use the same repeating task/stream order, share the corresponding
uniform draws, and retain 32,000 starts per task over 6,000 updates. All downstream
RGB, state, action, normalization, and stage-label transforms remain identical.
The prototypes passed sampler and source-identity checks; neither arm has been
trained. The earlier radio-only press-weighted checkpoint is a different
experiment and cannot substitute for this control.

## Next candidate: give π0.5 recent observation history

The official [LeRobot π0.5 memory documentation](https://huggingface.co/docs/lerobot/main/en/pi05#short-horizon-observation-memory-mem)
now provides a concrete implementation to test. It implements short-horizon
visual and proprioceptive memory; long-horizon language summaries remain
separate work. The documented public checkpoints lack memory pretraining, so
fine-tuning them does not reproduce the full MEM model's headline results.

Use six observations, including the current frame, with a stride of 30 frames
for the released 30 Hz data. This covers five seconds of history. Compare
single-frame, visual-memory, and visual-plus-proprioception variants using the
same training split, action representation, initialization and evaluation
protocol. Enable the selected memory inputs during training, and reset history
at every episode boundary. These experiments do not depend on the unverified
semantic annotation join.

This requires a pinned LeRobot PyTorch training and serving integration. It is
not implemented by the JAX EGR helpers, and this PR does not change the shared
Workbench image to an unvalidated upstream revision. Measure inference latency
and the complete policy's 24 GB memory use before combining memory and recovery.

## Evidence required before promoting a candidate

1. Verify replay inputs and frame alignment, including the observation/action
   timing convention. Exclude reporting and hidden instances from training.
2. Verify real π0.5 forward/backward updates, finite losses and checkpoint
   export/reload. A helper test or a decreasing training loss is insufficient.
3. Run the unchanged official evaluator on all ten frozen development instances,
   once each, at its default timeout. Preserve every failure and original video.
4. Compare complete policies against the same campaign's frozen RLC control.
   Never combine the best individual outcomes across policies.
5. Establish gains across a frozen set of additional tasks, expand training to
   all 100 tasks, and validate the submission's 24 GB serving constraint before
   making a full-challenge or winning-performance claim.

Memory-enabled π0.5 training, ForesightFlow-style recovery and semantic history
are subsequent experiments, not implemented capabilities of the current
candidate. The EGR helpers and JAX objective have 32 numerical and gradient
tests; complete GPU training, full-task coverage and organizer submission remain
separate evidence gates.
