# Sim2Real VLM-to-RL Runbook

**Docs:** [data contracts](../../../docs/workbench/guides/sim2real-data-contracts.md) ·
[operator guide](../../../docs/workbench/guides/sim2real-workflow.md) ·
[customer assets](../../../docs/workbench/guides/sim2real-customer-assets.md) ·
[architecture](../../../docs/workbench/guides/sim2real-architecture.md)

This workflow runs the full Sim2Real chain as one inspectable pipeline:

`LeRobot dataset trigger -> augment -> env generation -> train/held-out split -> action rollouts -> VLM critique -> RL signal -> trainer update -> held-out eval -> promote or loop back -> external validation stub -> retrigger`.

Steps 12 and 13 are documented external seams. Stage 2 materializes stock or BYO
scene and robot specs. Every step writes local artifacts and, when
`--upload-artifacts` is set, uploads the run tree to S3.

Canonical operator routing after CLI namespace cleanup: use
`npa workbench workflow submit npa/workflows/workbench/sim2real/runbook.yaml`
for cluster execution (auto-routes to the direct K8s staged Job when SkyPilot is
unavailable), `python -m npa.workflows.sim2real status <run-id> --watch` for live
progress, module CLI staged subcommands (`preamble`, `outer-iteration`,
`finalize`) for manual progression, and `npa workbench health sim2real` for
preflight checks. The SDK (`npa.sdk.workbench.sim2real`) mirrors run/status.

Canonical operator routing after CLI namespace cleanup: use
`npa workbench workflow submit` for cluster execution, module CLI staged
subcommands (`preamble`, `outer-iteration`, `finalize`) for manual progression,
and `npa workbench health sim2real` for preflight checks.

Canonical operator routing after CLI namespace cleanup: use
`npa workbench workflow submit` for cluster execution, module CLI staged
subcommands (`preamble`, `outer-iteration`, `finalize`) for manual progression,
and `npa workbench health sim2real` for preflight checks.

## Easy-Parameters Quickstart

Use this when you want the canonical `lerobot/pusht` demo shape with the fewest
knobs. The trigger path is the input path: dropping a LeRobot dataset there is
what starts a run. Keep the trigger path, simulation asset source path, and
output prefix separate.

```bash
# 1. Easy parameters.
export NPA_SIM2REAL_RUN_ID=pusht-demo-$(date -u +%Y%m%dT%H%M%SZ)
export NPA_SIM2REAL_BUCKET=<default-platform-bucket>
export NPA_SIM2REAL_PREFIX=""
export NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht
export NPA_SIM2REAL_TRIGGER_DATASET_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-triggers/${NPA_SIM2REAL_RUN_ID}/lerobot-pusht/"
export ASSETS_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-assets/pusht/"
export SCENE_SPEC_URI="s3://${NPA_SIM2REAL_BUCKET}/sim2real-assets/pusht/scene-spec.json"

# 2. Credentials and endpoints.
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-key>
export AWS_ENDPOINT_URL=<default-platform-s3-endpoint>

# Non-default S3-compatible endpoints supported but untested.

# 3. Single LeRobot trainer image override. Leave unset to use the reference image.
export TRAINER_IMAGE=<registry>/npa-lerobot-vlm-rl:0.1.0

# 4. Reference image defaults. Override only if you have a newer pushed image.
export AUGMENT_IMAGE=<registry>/npa-cosmos2-transfer:2.5.0
export POLICY_IMAGE=<registry>/npa-sim2real-reference-policy:0.1.1
export VLM_IMAGE=<registry>/npa-cosmos3-reason:3.0.1-genuine-sm120
export EVAL_IMAGE=<registry>/npa-sim2real-eval:0.1.1-genuine-sm120

# 5. Demo scale. Increase these for larger production runs.
export INNER_ITERATIONS=2
export OUTER_ITERATIONS=1
export LOOP_OF_LOOPS_ITERATIONS=1
export ROLLOUT_COUNT=3
export STEPS_PER_ROLLOUT=4
export HELDOUT_ENV_COUNT=8

# 6. Self-hosted dual VLM models (accept on Hugging Face — see operator guide).
export VLM_REASON2_MODEL=nvidia/Cosmos-Reason2-8B
export VLM_REASON3_MODEL=nvidia/Cosmos-Reason2-2B
export NPA_SIM2REAL_VLM_DUAL_REASON=1
# Mirror HF_TOKEN into cluster secret hf-ngc-tokens before GPU sibling Jobs run.

npa workbench workflow submit \
  npa/workflows/workbench/sim2real/runbook.yaml \
  --run-id "${NPA_SIM2REAL_RUN_ID}" \
  --var NPA_SIM2REAL_RUN_ID="${NPA_SIM2REAL_RUN_ID}" \
  --var NPA_SIM2REAL_BUCKET="${NPA_SIM2REAL_BUCKET}" \
  --var NPA_SIM2REAL_PREFIX="${NPA_SIM2REAL_PREFIX}" \
  --var AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_ID="${NPA_SIM2REAL_TRIGGER_DATASET_ID}" \
  --var ASSETS_URI="${ASSETS_URI}" \
  --var SCENE_SPEC_URI="${SCENE_SPEC_URI}" \
  --var INNER_ITERATIONS="${INNER_ITERATIONS}" \
  --var OUTER_ITERATIONS="${OUTER_ITERATIONS}" \
  --var LOOP_OF_LOOPS_ITERATIONS="${LOOP_OF_LOOPS_ITERATIONS}" \
  --var ROLLOUT_COUNT="${ROLLOUT_COUNT}" \
  --var STEPS_PER_ROLLOUT="${STEPS_PER_ROLLOUT}" \
  --var HELDOUT_ENV_COUNT="${HELDOUT_ENV_COUNT}" \
  --var VLM_REASON2_MODEL="${VLM_REASON2_MODEL}" \
  --var VLM_REASON3_MODEL="${VLM_REASON3_MODEL}" \
  --var NPA_SIM2REAL_VLM_DUAL_REASON="${NPA_SIM2REAL_VLM_DUAL_REASON:-1}"
```

Canonical S3 layout for the quickstart:

```text
s3://<bucket>/sim2real-triggers/<run-id>/lerobot-pusht/    # Step 1 trigger path
s3://<bucket>/sim2real-assets/pusht/                       # Step 2 sim assets and SceneSpec stub input
s3://<bucket>/<run-id>/                                     # Per-run output tree
```

The output tree includes:

```text
augment/
envs/raw/
envs/train/
envs/heldout/
actions/
vlm_eval/
training_signal/
inner_loop/
checkpoints/
eval/heldout/
outer_loop/decision.json
stage_13_retrigger/retrigger.json
reports/sim2real-report.json
reports/sim2real.rrd
```

## Prerequisites

- Python 3.11 or newer and this package installed in `npa/.venv`.
- A Kubernetes GPU cluster with schedulable RTX PRO 6000 class `sm_120` GPUs.
- Pushed reference images:
  - `npa-cosmos2-transfer:2.5.0`
  - `npa-sim2real-envgen:0.1.1`
  - `npa-sim2real-reference-policy:0.1.1`
  - `npa-cosmos3-reason:3.0.1-genuine-sm120`
  - `npa-lerobot-vlm-rl:0.1.0`
  - `npa-sim2real-eval:0.1.1-genuine-sm120`
- Gated model repository access accepted where required by the VLM image.
  Self-hosted dual VLM defaults: `nvidia/Cosmos-Reason2-8B` (Reason2) and
  `nvidia/Cosmos-Reason2-2B` (Reason3 sibling). Accept both on Hugging Face before launch.
  `nvidia/Cosmos3-Super-Reasoner` is Token Factory–hosted only — not an HF repo.
  See [sim2real-workflow.md](../../../docs/workbench/guides/sim2real-workflow.md#hugging-face-model-access-self-hosted-workbench).
- `HF_TOKEN` and `NGC_API_KEY` supplied through environment variables or a
  Kubernetes secret such as `hf-ngc-tokens`.
- S3-compatible storage credentials and endpoint configured through environment
  variables, project config, or Kubernetes secrets.

## Preflight First

Before launching anything, run the preflight. It validates the config and
surfaces the recurring blockers (S3 reachability, image pull / `agent-sa` path,
HF/NGC tokens, kube context, schedulable-GPU count, and three-tier coherence) as
PASS/WARN/FAIL/SKIP so you hit them up front instead of mid-pipeline:

```bash
npa workbench health sim2real \
  --s3-bucket <bucket> \
  --s3-endpoint <your-s3-compatible-endpoint> \
  --trigger-dataset-uri s3://<bucket>/sim2real-triggers/<run-id>/lerobot-pusht/ \
  --assets-uri s3://<bucket>/sim2real-assets/pusht/ \
  --policy-image <registry>/npa-sim2real-reference-policy:0.1.1
```

Use `--checks config,coherence` for an infra-free static check, `--json` for
machine-readable output, and `--warn-only` to report without failing the exit
code.

## One BYO Seam, One Value

Each seam is one value you set the same way across all three tiers: the CLI
flag, the SDK keyword argument, and the raw-YAML env are all wired to the same
config field. Set what you need; the rest fall back to reference defaults.

| BYO seam | CLI flag | SDK keyword | Raw-YAML env |
| --- | --- | --- | --- |
| S3 endpoint | `--s3-endpoint` | `s3_endpoint=` | `AWS_ENDPOINT_URL` |
| S3 bucket (required) | `--s3-bucket` | `s3_bucket=` | `NPA_SIM2REAL_BUCKET` |
| Run prefix | `--s3-prefix` | `s3_prefix=` | `NPA_SIM2REAL_PREFIX` |
| Trigger path | `--trigger-dataset-uri` | `trigger_dataset_uri=` | `NPA_SIM2REAL_TRIGGER_DATASET_URI` |
| Source dataset id | `--trigger-dataset-id` | `trigger_dataset_id=` | `NPA_SIM2REAL_TRIGGER_DATASET_ID` |
| Sim-asset source path | `--assets-uri` | `assets_uri=` | `ASSETS_URI` |
| SceneSpec path | `--scene-spec-uri` | `scene_spec_uri=` | `SCENE_SPEC_URI` |
| Augment image | `--augment-image` | `augment_image=` | `AUGMENT_IMAGE` |
| Policy image | `--policy-image` | `policy_image=` | `POLICY_IMAGE` |
| Trainer image | `--trainer-image` | `trainer_image=` | `TRAINER_IMAGE` |
| VLM image | `--vlm-image` | `vlm_image=` | `VLM_IMAGE` |
| Eval image | `--eval-image` | `eval_image=` | `EVAL_IMAGE` |
| Success threshold | `--threshold` | `threshold=` | `SUCCESS_THRESHOLD` |
| Inner-loop cap | `--inner-iterations` | `inner_iterations=` | `INNER_ITERATIONS` |
| Outer-loop cap | `--outer-iterations` | `outer_iterations=` | `OUTER_ITERATIONS` |
| Loop-of-loops cap | `--loop-of-loops-iterations` | `loop_of_loops_iterations=` | `LOOP_OF_LOOPS_ITERATIONS` |
| Rollout count | `--rollout-count` | `rollout_count=` | `ROLLOUT_COUNT` |
| Steps per rollout | `--steps-per-rollout` | `steps_per_rollout=` | `STEPS_PER_ROLLOUT` |
| Held-out env count | `--heldout-env-count` | `heldout_env_count=` | `HELDOUT_ENV_COUNT` |
| VLM command swap | `--byo-vlm-command` | `byo_vlm_command=` | `BYO_VLM_COMMAND` |
| Signal-converter swap | `--byo-signal-converter` | `byo_signal_converter=` | `BYO_SIGNAL_CONVERTER` |
| Trainer command swap | `--byo-trainer-command` | `byo_trainer_command=` | `BYO_TRAINER_COMMAND` |
| Held-out eval swap | `--byo-eval-command` | `byo_eval_command=` | `BYO_EVAL_COMMAND` |
| Rerun viz toggle | `--rerun` / `--no-rerun` | `rerun_enabled=` | `NPA_SIM2REAL_RERUN` |
| Rerun command swap | `--byo-rerun-command` | `byo_rerun_command=` | `BYO_RERUN_COMMAND` |

### Command-Swap I/O Contracts

Each `byo_*_command` is a shell command the loop runs at the matching seam. They
all share the same convention: inputs arrive as JSON file paths in environment
variables, and the command writes its result JSON to `NPA_SIM2REAL_OUTPUT_JSON`.
A command that exits non-zero, writes nothing, or emits a non-conforming /empty
document fails the run loudly — the loop never silently falls back to the
reference implementation. Each run records which path executed
(`trainer_source` / `signal_converter_source` = `byo_command` | `reference`) in
the inner-loop evidence so a run can prove the customer hook actually ran.

| Seam | Reads | Writes (`NPA_SIM2REAL_OUTPUT_JSON`) |
| --- | --- | --- |
| `byo_vlm_command` | `NPA_SIM2REAL_ROLLOUT_DIR`, `NPA_SIM2REAL_ROLLOUT_MANIFEST` | `npa.sim2real.vlm_eval.v1` (score + per_step critiques) |
| `byo_signal_converter` | `NPA_SIM2REAL_EVALUATION_JSON` | `npa.sim2real.rl_signal.v1` (non-empty `per_step` of `{step, reward, advantage?, target?, error_tags?}`) |
| `byo_trainer_command` | `NPA_SIM2REAL_SIGNAL_JSON` (+ `NPA_SIM2REAL_INITIAL_REWARD_HEAD`, `NPA_SIM2REAL_INITIAL_ACTION_BIAS`, `NPA_SIM2REAL_LEARNING_RATE`, `NPA_SIM2REAL_SIGNAL_LOSS_WEIGHT`) | trainer update with at least `reward_head_after`, `policy_output_after` (list), `policy_delta_l2` (optional `loss_before`/`loss_after`) |
| `byo_eval_command` | `NPA_SIM2REAL_HELDOUT_ENVS_DIR`, `NPA_SIM2REAL_INNER_EVIDENCE_JSON` | `npa.sim2real.heldout_eval.v1` (non-empty `per_env`) |
| `byo_rerun_command` | `NPA_SIM2REAL_RUN_DIR`, `NPA_SIM2REAL_REPORT_JSON` | non-empty `.rrd` at `NPA_SIM2REAL_OUTPUT_RRD` |

When a `byo_trainer_command` is set, the per-iteration no-signal **control** still
runs the in-process reference trainer. This keeps the policy-delta attribution
honest: the BYO trainer produces the signal-driven update, and the reference
control provides the shared-initial-state, no-signal baseline that the delta is
measured against.

## Run All Three Tiers

Each tier is independently usable. The raw YAML runs without npa in the loop;
the SDK and CLI wrap the same workflow without gating it.

Raw SkyPilot — `runbook.yaml` is materialized with literal defaults because
SkyPilot 0.12.2 does **not** interpolate `${VAR}` inside the YAML `envs` block or
in `image_id`. Override the literals at submit time with `--env` / `--secret`,
and edit `image_id` to your own registry:

```bash
cat > /tmp/sim2real-skypilot-k8s.yaml <<'YAML'
kubernetes:
  pod_config:
    spec:
      serviceAccountName: agent-sa
      envFrom:
        - secretRef:
            name: hf-ngc-tokens
YAML

# Reaching GPUs: raw `sky jobs launch` against this YAML is currently blocked by
# the SkyPilot 0.12.2 pre-setup getcwd() bug. Until that is fixed upstream, reach
# GPUs through the materialized-runbook / direct-Kubernetes route: render the
# run-block command (literal endpoint, bucket, and images already in place) into
# a Kubernetes Job that uses the agent-sa pull path and the hf-ngc-tokens secret.
sky jobs launch \
  --config /tmp/sim2real-skypilot-k8s.yaml \
  --infra k8s/<cluster-name> \
  --env NPA_SIM2REAL_BUCKET=<bucket> \
  --env AWS_ENDPOINT_URL=<your-s3-compatible-endpoint> \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/sim2real/runbook.yaml
```

SDK:

```python
from npa.sdk.workbench import sim2real

report = sim2real.run(
    run_id="pusht-sdk-demo",
    s3_bucket="<bucket>",
    s3_prefix="pusht-sdk-demo",
    trigger_dataset_uri="s3://<bucket>/sim2real-triggers/pusht-sdk-demo/lerobot-pusht/",
    trigger_dataset_id="lerobot/pusht",
    assets_uri="s3://<bucket>/sim2real-assets/pusht/",
    scene_spec_uri="s3://<bucket>/sim2real-assets/pusht/scene-spec.json",
    threshold=0.75,
    inner_iterations=2,
    outer_iterations=1,
    upload_artifacts=True,
)
print(report["outer_loop"]["latest_decision"])
```

SDK staged helpers:

```python
from npa.sdk.workbench import sim2real

state = sim2real.preamble(run_id="pusht-sdk-staged", output_dir="/tmp/s2r-staged")
iteration = sim2real.outer_iteration(
    run_id="pusht-sdk-staged",
    output_dir="/tmp/s2r-staged",
    outer_iteration=1,
    initial_quality=float(state["current_quality"]),
)
report = sim2real.finalize(
    run_id="pusht-sdk-staged",
    output_dir="/tmp/s2r-staged",
    stage_records=state["stage_records"],
    components=state["components"],
    outer_history=[iteration["history_entry"]],
    final_inner=iteration["inner"],
    final_eval=iteration["heldout_report"],
    final_decision=iteration["decision"],
)
print(report["outer_loop"]["latest_decision"])
```

Workflow submit:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/sim2real/runbook.yaml \
  --run-id pusht-cli-demo \
  --var NPA_SIM2REAL_RUN_ID=pusht-cli-demo \
  --var NPA_SIM2REAL_BUCKET=<bucket> \
  --var NPA_SIM2REAL_PREFIX=pusht-cli-demo \
  --var NPA_SIM2REAL_TRIGGER_DATASET_URI=s3://<bucket>/sim2real-triggers/pusht-cli-demo/lerobot-pusht/ \
  --var NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht \
  --var ASSETS_URI=s3://<bucket>/sim2real-assets/pusht/ \
  --var SCENE_SPEC_URI=s3://<bucket>/sim2real-assets/pusht/scene-spec.json \
  --var INNER_ITERATIONS=2 \
  --var OUTER_ITERATIONS=1
```

Inner loop only (local SDK / module — no top-level workbench CLI):

```bash
npa/.venv/bin/python -m npa.workflows.sim2real_loop inner-loop \
  --run-id sim2real-inner-example \
  --output-dir /tmp/sim2real-inner-example \
  --inner-iterations 2
```

## Stages And Artifacts

1. Trigger: consumes `--trigger-dataset-uri` and writes
   `stage_01_trigger/trigger.json`.
2. Sim assets: writes `stage_02_assets/consumed_scene_spec.json` and
   `consumed_robot_spec.json` (stock Franka + tabletop by default; BYO via
   `ASSETS_URI` / `SCENE_SPEC_URI` / `ROBOT_SPEC_URI`).
3. Augmentation: writes `augment/manifest.json`.
4. Environment generation: writes `envs/raw/manifest.json`.
5. Train and held-out split: writes `envs/train/manifest.json` and
   `envs/heldout/manifest.json` using an 80/20 split.
6. Token manifest: writes `tokens/manifest.json`.
7. Action-conditioned rollouts: writes `actions/train/.../rollout-*/`.
8. VLM eval: writes structured critique JSON under `vlm_eval/train/`.
9. RL signal and trainer update: writes `training_signal/train/` and
   `inner_loop/.../evidence.json`.
10. Held-out eval: writes `eval/heldout/report.json`.
11. Threshold gate: writes `outer_loop/decision.json`; when the threshold is
    met it writes `checkpoints/candidate/candidate.json`, otherwise
    `outer_loop/loopback.json` points back to Stage 7.
12. Real-robot validation: documented external stub at
    `stage_12_external_validation/external_stub.json`.
13. Retrigger: writes `stage_13_retrigger/retrigger.json`, targeting Stage 1
    when a new real-world LeRobot dataset lands in the trigger path.
14. Rerun visualization: writes `reports/sim2real.rrd` (a single Rerun recording)
    from the completed run's artifacts. Default on (`NPA_SIM2REAL_RERUN=1` /
    `--rerun`); set `NPA_SIM2REAL_RERUN=0` / `--no-rerun` to skip. Degrades to a
    WARN (not a hard failure) when `rerun-sdk` is not installed locally, but
    always produces the `.rrd` when it is available.

### Rerun Visualization

The `.rrd` reuses the repo's existing Rerun capability (the same `rerun-sdk`
recording API the LeRobot/GR00T adapters build on) and logs, on a shared
`frame_time` timeline:

- `rollouts/iter_NN/<rollout_id>/camera` — rollout camera frames as image streams.
- `rollouts/iter_NN/<rollout_id>/critique` — per-step VLM critique text + error
  tags as a text document overlay.
- `rollouts/iter_NN/<rollout_id>/score` — the VLM success score.
- `signal/reward` and `signal/advantage` — the per-step VLM->RL signal as scalar
  timeseries, plus `signal/reward_trend` across iterations.
- `heldout/scores` and `heldout/per_env/<env_id>` — held-out per-env scores as a
  scalar/bar view.

Open it with `rerun reports/sim2real.rrd`, or load it headlessly with
`rerun.recording.load_recording(...)` to inspect entity counts. Set
`NPA_SIM2REAL_RERUN_MP4=1` to also emit best-effort per-rollout `rollout.mp4`
files (skipped when `ffmpeg` is not on PATH). When `--upload-artifacts` is set,
the `.rrd` uploads with the rest of the run tree.

## Loops

Inner loop, Stages 7 to 9:

```text
Reference action generation -> VLM eval -> critique-to-reward signal -> trainer update
```

Outer loop, Stages 10 to 11:

```text
held-out eval -> threshold gate -> promote checkpoint or loop back to Stage 7
```

Loop-of-loops, Stages 12 to 13 to 1:

```text
real-robot validation stub -> retrigger manifest -> next LeRobot dataset batch in trigger path
```

The VLM eval schema is:

```json
{
  "schema": "npa.sim2real.vlm_eval.v1",
  "rollout_id": "rollout-0000",
  "success": false,
  "per_step": [
    {"step": 0, "critique_text": "...", "error_tags": ["missed_target"]}
  ],
  "summary": "..."
}
```

The RL signal schema is:

```json
{
  "schema": "npa.sim2real.rl_signal.v1",
  "rollout_id": "rollout-0000",
  "per_step": [
    {
      "step": 0,
      "reward": -0.35,
      "advantage": -0.1,
      "target": {
        "nl_correction": "Move the end effector toward the object center before closing.",
        "action_delta": [0.12, 0.02, 0.0]
      }
    }
  ]
}
```

The reference trainer integration point is after the LeRobot policy forward pass
and before `optimizer.step()`:

```text
loss = imitation_loss
     + signal_loss_weight * corrective_mse
     - advantage * policy_logit_proxy
```

## BYO Seams

Every seam is available in raw SkyPilot envs, SDK keyword arguments, and CLI
options:

- `s3_endpoint`, `s3_bucket`, `s3_prefix`
- `trigger_dataset_uri`, `trigger_dataset_id`
- `assets_uri`, `scene_spec_uri`
- `augment_image`
- `action_rollouts_uri`, `train_envs_uri`, `heldout_envs_uri`
- `policy_image`
- `vlm_image`, `vlm_model`, `byo_vlm_command`
- `byo_signal_converter`
- `trainer_image`, `byo_trainer_command`
- `eval_image`, `byo_eval_command`
- `rerun_enabled`, `byo_rerun_command`
- `threshold`
- `inner_iterations`, `outer_iterations`, `loop_of_loops_iterations`
- `rollout_count`, `steps_per_rollout`, `heldout_env_count`
- `signal_loss_weight`, `learning_rate`
- `no_guardrails`

## Scale Knobs

The demo scale intentionally exercises every stage with small numbers. To scale
toward large environment generation runs, increase:

- `HELDOUT_ENV_COUNT` for generated environment count and held-out eval breadth.
- `ROLLOUT_COUNT` and `STEPS_PER_ROLLOUT` for action and VLM-eval volume.
- `INNER_ITERATIONS` for repeated critique-to-reward trainer updates.
- `OUTER_ITERATIONS` for held-out failures to loop back through Stage 7.
- `LOOP_OF_LOOPS_ITERATIONS` when real-world validation should start a next
  dataset-triggered run.

For augmentation-heavy runs, shard the trigger dataset by prefix and submit
multiple SkyPilot jobs with distinct `NPA_SIM2REAL_RUN_ID` values and output
prefixes. Keep each job pointed at the same pushed reference images unless a
new image has already been built and pushed.
