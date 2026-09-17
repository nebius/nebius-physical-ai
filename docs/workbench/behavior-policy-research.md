# Improving the BEHAVIOR policy with recent research

[Challenge workflow](behavior-challenge.md)

The current RLC controller already uses a π0.5-derived model. Its verified result
is **Q = 0.20 on ten radio development instances**, against the official radio
baseline's Q = 0.10. Neither result establishes performance across the 100-task
2026 challenge. The research candidate described here has no measured rollout
improvement yet.

## Research reviewed

Reviewed September 16, 2026 using author papers, repositories and model releases.

| Research | Relevant finding | Decision for this integration |
| --- | --- | --- |
| [Evidence-Gated Regularization, September 2026](https://arxiv.org/abs/2609.03142) | A π0.5 training objective addresses distraction by irrelevant cameras and dependence on cameras that become occluded. | Implement camera evidence and importance-weighted consistency losses; measure the resulting policy on the unchanged evaluator. |
| [SERF, June 2026](https://arxiv.org/abs/2606.12956), [released code](https://github.com/ExistentialRobotics/SERF-VLA) | Spatial and temporal feature maps provide scene memory. Its formulation includes robot base pose and camera poses. | A challenge candidate needs pose estimated from permitted sensors. Do not connect simulator global pose to the released mapper. |
| [ForesightFlow, June 2026](https://arxiv.org/abs/2606.04968) | Jointly generated action and success-potential coordinates support learning from mixed-quality trajectories and selecting action chunks. | Requires additional training-instance rollouts containing both successes and failures. The existing development rollouts are evaluation evidence, not training data. |
| [π0.7, April 2026](https://www.pi.website/blog/pi07) | Diverse conditioning supports compositional behavior and strategy control. | The checked [OpenPI release](https://github.com/Physical-Intelligence/openpi) supplies π0.5 models. This integration does not claim to run π0.7. |
| [Multi-scale Embodied Memory, March 2026](https://www.pi.website/research/memory) | Combines recent visual history with longer-term language memory. | Pursue observation-derived subtask memory after validating annotation alignment; elapsed time is insufficient evidence that a subtask finished. |
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
released LeRobot episodes. Only **9/200** skill annotation durations equal their
episode lengths. This mismatch does not prove the annotations are incorrect, but
it means a direct frame-index join is unverified. Do not stretch timestamps,
silently truncate labels, or train a semantic transition predictor on that join.
Replay-derived camera evidence must instead pass frame count, source-action and
observation alignment checks.

Direct robot-camera segmentation replay encountered the upstream failure in
[BEHAVIOR issue #2312](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2312).
A training-only viewer-camera replay is being checked against the recorded
camera poses, intrinsics and released observations. Its labels must pass those
checks before use; no EGR checkpoint has been trained or evaluated.

## Evidence required before promoting a candidate

1. Verify replay inputs and frame alignment, including the observation/action
   timing convention. Exclude reporting and hidden instances from training.
2. Verify real π0.5 forward/backward updates, finite losses and checkpoint
   export/reload. A helper test or a decreasing training loss is insufficient.
3. Run the unchanged official evaluator on all ten frozen development instances,
   once each, at its default timeout. Preserve every failure and original video.
4. Compare complete policies against the Q = 0.20 RLC control. Never combine the
   best individual outcomes across policies.
5. Establish gains across a frozen set of additional tasks, expand training to
   all 100 tasks, and validate the submission's 24 GB serving constraint before
   making a full-challenge or winning-performance claim.

ForesightFlow-style recovery and MEM-style semantic history are subsequent
experiments, not implemented capabilities of the current candidate. The EGR
helpers and JAX objective have 32 numerical and gradient tests; complete GPU training, full-task coverage
and organizer submission remain separate evidence gates.
