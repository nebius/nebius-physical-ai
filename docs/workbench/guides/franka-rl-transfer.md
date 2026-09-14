# Franka RL: train in simulation, measure the transfer gap

Run [`franka-rl-transfer.yaml`](../../../workflows/testing/franka-rl-transfer.yaml)
on a reserved RTX PRO 6000 with the `rtx-rendering` cluster profile. It trains a
real Franka Panda in Isaac Lab using native RSL-RL PPO, evaluates the selected
weights under physical perturbations, and exports genuine rendered rollouts as
LeRobotDataset v3 and Rerun evidence.

This complements the [LeRobot PushT benchmark](lerobot-transfer.md). PushT
measures imitation-learning robustness in a planar task. This workflow provides
the articulated robot, gripper, contact dynamics, and reinforcement-learning
experiment needed to study embodied transfer. Neither experiment proves
physical robot transfer without a subsequent hardware trial.

## Relationship to the reference Sim2Real pipeline

The reference is [`workflows/main/sim2real.yaml`](../../../workflows/main/sim2real.yaml).
This compact workflow retains its relevant experimental contracts:

| Reference responsibility | Franka RL implementation |
| --- | --- |
| Task and embodiment contract | Pin `Isaac-Lift-Cube-Franka-v0`, its Franka joint control, geometric success criteria, and reset streams before training |
| Stage 9: genuine PPO | Use the upstream Franka PPO configuration, randomized object mass/friction, native optimizer updates, initial weights, and periodic checkpoints |
| Validation-only selection | Rank checkpoint success on the same validation resets; break ties by closest goal distance, then earlier iteration |
| Stage 10: exact held-out policy | Evaluate the initial and selected weights on paired, untouched test resets; verify initial physical-state hashes |
| Stage 11: quality decision | Report the measured success threshold independently of workflow completion |
| Stage 14: factual visualization | Convert actual Isaac RTX frames and synchronized state/actions with the existing Isaac-to-LeRobot adapter; record named Franka telemetry and embedded videos in Rerun |

The experiment uses simulator rewards and privileged object state. It does not
run the reference pipeline's Cosmos, environment-generation, or hosted VLM
stages. Its task is **lift and hold at a commanded goal**, not object release
onto a support surface. A hardware student still needs validated perception,
calibrated action conversion, and a separate physical evaluation.

## Reproduce on Nebius

Configure an operator-owned project, writable S3 storage, and an exact
Kubernetes context. Provision a render-capable RTX node with
`npa cluster up --gpu-workload-profile rtx-rendering`, selecting its reserved
capacity group. The profile verifies CUDA and the graphics-driver interfaces.
See [GPU provisioning](../../../skills/tools/gpu-cluster-provisioning/SKILL.md).

The workflow pins the existing payload-clean Isaac Lab 3.0 beta 2 patch 1 image
and LeRobot image by digest. Isaac Lab remains a **beta** runtime. Its NVIDIA
runtime is fetched under the existing [Isaac acceptance and explicit opt-out
policy](../../../skills/atomic/third-party-eula-preflight/SKILL.md); optional
telemetry remains disabled. No new model weights or gated dataset is required.
See [Isaac Lab 3 packaging](../isaac-lab-3.md).

Set `NPA_PROJECT`, `NPA_KUBE_CONTEXT`, `NPA_OUTPUT_BUCKET`, and a fresh
`NPA_RUN_ID`. Use the project's private configuration and a new isolated
SkyPilot directory for this run. The spec enables `source_overlay: "1"` so both
images execute this checkout's modules. Stage the source before starting its API:

```bash
npa/.venv/bin/npa workbench health preflight --project "$NPA_PROJECT" --checks nebius,s3
npa/.venv/bin/npa workbench workflow stage-src --project "$NPA_PROJECT" --bucket "$NPA_OUTPUT_BUCKET"
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/franka-rl-transfer.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/franka-rl-transfer.yaml --run-id "$NPA_RUN_ID" --waves --json
npa/.venv/bin/npa workbench workflow preflight-images workflows/testing/franka-rl-transfer.yaml --json
npa/.venv/bin/npa workbench workflow submit workflows/testing/franka-rl-transfer.yaml \
  --project "$NPA_PROJECT" --infra "k8s/$NPA_KUBE_CONTEXT" \
  --isolated-config-dir "$NPA_SKYPILOT_ISOLATED_CONFIG_DIR" \
  --run-id "$NPA_RUN_ID" --runtime --stage-src --max-wait-seconds 0 \
  --var "bucket=$NPA_OUTPUT_BUCKET" \
  --var "prefix=franka-rl-transfer/$NPA_RUN_ID" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The standard runtime owns all four stages. Stages exchange checksum-verified
S3 directories; native Isaac processes run inside their assigned GPU task.
Evaluation starts separate simulator processes for validation, held-out tests,
and camera capture, giving each reset stream its own Replicator lifecycle.
The stage publishes only after all three completion records are verified.
Keep each experiment's prefix and isolated API configuration unchanged while
it owns jobs. Use a fresh run identity after changing the experimental recipe.
An explicit `--resume-run "$NPA_RUN_ID"` reuses successful stages after checking
their durable outputs. After correcting a terminal implementation failure,
stage a new immutable source snapshot, use a fresh isolated API directory, and
explicitly enable the next payload retry with `--retries`. Keep the earlier
source snapshots and attempt history; completed training need not run again.

## Protocol and configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Training initialization; cannot overlap reserved evaluation streams |
| `iterations` | 1,500 | Real PPO iterations, with 24 transitions per environment per iteration |
| `num_envs` | 4,096 | Parallel training environments; default total is 147,456,000 transitions |
| `eval_episodes` | 128 | Environments per validation checkpoint and per test condition/arm |
| `minimum_success` | 0.7 | Required success rate in every test condition |
| `bucket`, `prefix` | Project binding and run-specific prefix | Durable stage artifacts |

Training randomizes object mass to 0.7–1.3 times the stock mass and object
friction to 0.5–1.5. Applied values are read back from Isaac's rigid-body view.
The resolved PPO architecture and optimizer settings remain explicit in
`training.json`; the digest-pinned task supplies its stock reward terms. The run
records the actual policy parameter change, framework
versions, accelerator, optimizer settings, elapsed training time, and native
checkpoint hashes.

Native periodic checkpoint filenames use the runner's zero-based iteration
counter. The workflow also writes a final checkpoint after all requested
updates; validation may select an earlier checkpoint with equivalent or better
measured behavior.

Validation uses the nominal condition at reset stream 100,000. Checkpoint
selection finishes and writes `selection.json` before test stream 200,000 is
opened. Every test condition compares the initial and trained actor on 128
paired environments. Initial physical-state hashes must match across the arms
and conditions. Environment indices identify independent resets within each
seeded vector environment; they are not invented per-episode RNG seeds.

| Test condition | Object mass | Object friction | Action delay |
| --- | --- | --- | --- |
| Nominal | 1× | 1.0 | None |
| Heavy | 2× | 1.0 | None |
| Slippery | 1× | 0.25 | None |
| Delay | 1× | 1.0 | One control step |

Success requires the object to remain above 10 cm, within 5 cm of its commanded
goal, and below 3 cm/s for **20 consecutive control steps**. Automatic-reset
observations cannot complete or restart that streak. Lift achievement and
closest distance remain diagnostic metrics. Qualification requires every
condition to clear the configured threshold; completing the workflow alone
cannot qualify a policy. The paired bootstrap interval is clustered by vector
environment index across conditions and measures reset variability for one
training seed.

A separate matched PPO ablation is needed to attribute gains specifically to
mass/friction randomization. The ranges are experimental perturbations; a
physical deployment needs measured robot and object calibration.

## Review the artifacts

The output prefix contains `prepared/recipe.json`, `training/training.json`,
native initial/periodic checkpoints and logs, `evaluation/selection.json`,
per-checkpoint validation, paired test episodes, and raw capture trajectories.
The final `reports/` directory contains:

- `report.json`, `trials.csv`, and `success.png`: measured simulation outcomes.
- `lerobot/`: Franka state, applied joint-control actions, and real RTX video.
- `capture.json`: exact checkpoint, capture stream, timing, and per-episode
  success labels. Recorded failures remain failures; these are policy rollouts,
  not automatically accepted expert demonstrations.
- `franka.rrd`: the actual rendered embodiment with its synchronized telemetry.
- `recording-validation.json`: decoded row counts for every camera, joint, and
  action channel, with the workflow run ID used as the recording ID.

Capture uses a third stream, 300,000, independent of checkpoint selection and
test scoring. Nine Franka joint positions are recorded with eight control
actions: seven scaled joint-position targets and one binary gripper command.
These normalized simulator actions require an explicit controller mapping
before use on hardware.

The teacher also consumes privileged joint velocities, object position, and a
commanded goal. The review export contains RGB, joint positions, and actions;
student training needs an explicit observable-goal contract and additional
input features as appropriate.

Download and inspect the recording with `rerun <recording.rrd>` and decode the
LeRobot MP4s before treating an artifact reference as visual evidence. When the
run is complete, cancel its jobs before removing its dedicated cluster and
workflow identity; retain the artifact storage for reproduction.

## Measured RTX PRO 6000 result

The native run completed **1,500 PPO updates across 4,096 Franka environments**:
147,456,000 transitions in 22.1 minutes of learning and final checkpoint export,
excluding worker installation and renderer startup. The actor's parameter
change had L2 norm 32.6653. Validation evaluated five saved checkpoints on
640 episodes and selected checkpoint `model_1499.pt` before opening the test
stream. The final native checkpoint contains the same learned actor after the
requested updates; checkpoint counters follow the upstream runner convention.

All 1,024 test trials passed the exact reset-pairing and checkpoint-lineage
checks. Two evaluation attempts using the same saved weights and recipe in
fresh workers produced byte-identical evaluation JSON, including all validation
and held-out trial records. The initial actor never lifted the cube and had
zero strict successes in every condition. The trained actor learned to lift reliably, but rarely
satisfied the stable-hold criterion:

| Held-out condition | Trained cube lifts | Trained strict successes |
| --- | ---: | ---: |
| Nominal | 128/128 | 10/128 (7.8%) |
| Heavy | 128/128 | 3/128 (2.3%) |
| Slippery | 128/128 | 4/128 (3.1%) |
| One-step delay | 119/128 | 0/128 (0.0%) |

Mean strict success improved from 0% to 3.3%; the paired 95% bootstrap interval
for the improvement is **+1.4 to +5.7 percentage points**. Every condition
failed the preregistered 70% threshold. **Simulation qualification is false;
physical transfer is untested.** Stock lift/tracking reward is not sufficient
evidence of a stable hold. A subsequent experiment should address settling and
actuation robustness, select changes on validation, and seal a new held-out
stream before testing.

The pinned beta simulator reported its TGS warning about potentially noisy
velocities. This run retains the sealed velocity threshold and measured results;
it does not establish whether that warning contributed to the low hold scores.

Four independent capture episodes produced 1,000 synchronized 640×480 RGB
frames at 50 Hz, nine joint positions, and eight applied actions. All four
lifted the cube; none satisfied the strict hold criterion. One reached 19 of
the required 20 stable steps and remains labeled as a failure. Every MP4 was
fully decoded; all 72 Rerun camera/telemetry entities matched their frame and
time timelines and the sealed run identity.

![Actual Franka capture, episode 1 at 2 seconds](../evidence/franka-rl-rtx-preview.png)

This is an unedited decoded frame from the recorded policy rollout. See the
[measured evidence and hashes](../evidence/franka-rl-rtx.json),
[complete held-out trial table](../evidence/franka-rl-rtx-trials.csv),
[native learning curve](../evidence/franka-rl-rtx-training.png), and
[success comparison](../evidence/franka-rl-rtx-success.png).
