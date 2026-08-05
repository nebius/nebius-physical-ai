# Canonical Sim2Real workflow

The production Sim2Real entrypoint is
[`npa/workflows/sim2real.yaml`](../../../npa/workflows/sim2real.yaml). It is the
only executable 14-stage Sim2Real YAML and sits beside
[`npa/workflows/physical-ai-data-factory.yaml`](../../../npa/workflows/physical-ai-data-factory.yaml).
For the neighboring data-factory workflow, see the
[Physical AI Data Factory guide](physical-ai-data-factory.md).

`npa workbench workflow submit` recognizes the canonical file and routes it
through `detect.py` to the direct-Kubernetes Sim2Real submitter. The orchestrator
then dispatches real `s2r-*` sibling Jobs. It does not use the generic workflow
graph runtime or demo toolRefs.

## What the 14 stages do

| Stage | Work | Required evidence |
| --- | --- | --- |
| 1 | consume the LeRobot trigger | `stage_01_trigger/trigger.json` |
| 2 | resolve scene, robot, and camera assets | `stage_02_assets/consumed_*_spec.json` |
| 3 | real Cosmos Transfer augmentation | `augment/manifest.json`, generated frames/video |
| 4 | generate raw environments | `envs/raw/` |
| 5 | disjoint train split | `envs/train/envs.jsonl` |
| 6 | tokenize/index environments | `tokens/manifest.json` |
| 7 | real Isaac policy rollouts | `actions/train/` and camera provenance |
| 8 | real Cosmos Reason critique | `vlm_eval/train/**/*.json` |
| 9 | critique-to-reward plus real RSL-RL PPO | `training_signal/train/`, `model_*.pt` |
| 10 | candidate-loaded held-out Isaac inference | `eval/heldout/report.json` and RGB-D captures |
| 11 | aggregate threshold decision | `outer_loop/decision.json` |
| 12 | external real-robot validation record | intentional `SEAM` stub; no GPU compute |
| 13 | retrigger record | `stage_13_retrigger/retrigger.json` |
| 14 | operator visualization | `reports/sim2real.rrd` and `.mcap` |

Stage 12 is deliberately an external-validation record. It records the candidate
checkpoint handoff and its promotion status; it does not claim that a robot was
operated or that another GPU workload ran.

## Preflight

Use the repository environment for every command:

```bash
npa/.venv/bin/npa --version
npa/.venv/bin/npa workbench health sim2real --checks all --json
```

Configure these non-secret values in `~/.npa/config.yaml`:

- `storage.bucket`: usable S3-compatible artifact bucket.
- `storage.endpoint_url`: its endpoint.
- `storage.registry`: qualified Nebius registry, for example
  `cr.eu-north1.nebius.cloud/<registry-id>`.
- `storage.sim2real_stock_trigger_uri`: a populated LeRobot trigger prefix.
- Kubernetes context/profile for the target cluster.

Keep S3 HMAC credentials, `HF_TOKEN`, `NGC_API_KEY`, registry credentials, and
Token Factory credentials in `~/.npa/credentials.yaml`; never put them in YAML or
shell history. The `default` namespace needs `npa-storage-credentials` and
`hf-ngc-tokens`, and the service account needs a current registry pull secret.

Accept access for all three runtime weight repositories with the same Hugging
Face account as `HF_TOKEN`:

- `nvidia/Cosmos-Reason2-8B`
- `nvidia/Cosmos-Reason2-2B`
- `nvidia/Cosmos-Transfer2.5-2B`

Check the cluster before launch:

```bash
kubectl get nodes -L nvidia.com/gpu.product
kubectl -n default get secret npa-storage-credentials hf-ngc-tokens
kubectl -n default get serviceaccount agent-sa -o yaml
```

Isaac rendering is restricted to RT-core products: RTX PRO 6000 or L40S label
variants. It is never routed to H100, H200, B200, or B300. When Kubernetes gives
concrete insufficient-capacity evidence, the submitter tries every compatible
product label in `NPA_SIM2REAL_K8S_GPU_CANDIDATES` in order. Image, credential,
and runtime failures are not misclassified as capacity and do not trigger a GPU
fallback.

The direct-Kubernetes driver is control-plane only and does not reserve a GPU.
Stages that require compute dispatch registry-qualified `s2r-*` sibling Jobs
with explicit GPU requests and product selectors. Consequently
`NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS=16` can use all 16 GPUs on a 16-GPU cluster
instead of losing one device to an idle orchestrator.

## One-command real-tier launch

This example uses the canonical loop counts, no early exit, uncapped held-out
evaluation, real BYO Isaac PPO, all three cameras at 640×480, and required Rerun
and MCAP output. Replace the three angle-bracket values. The image tags shown are
the repository-pinned real component tags; the registry prefix makes every image
pull explicit.

```bash
RUN=sim2real-production-$(date -u +%Y%m%dt%H%M%Sz)
BUCKET=<artifact-bucket>
ENDPOINT=https://storage.us-central1.nebius.cloud
REGISTRY=cr.eu-north1.nebius.cloud/<registry-id>
TRIGGER=s3://${BUCKET}/sim2real-triggers/<batch>/lerobot-pusht/

npa/.venv/bin/npa workbench workflow submit npa/workflows/sim2real.yaml \
  --run-id "${RUN}" \
  --var NPA_SIM2REAL_BUCKET="${BUCKET}" \
  --var AWS_ENDPOINT_URL="${ENDPOINT}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_URI="${TRIGGER}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht \
  --var AUGMENT_IMAGE="${REGISTRY}/npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z" \
  --var ENVGEN_IMAGE="${REGISTRY}/npa-envgen:cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
  --var POLICY_IMAGE="${REGISTRY}/npa-lerobot-vlm-rl:cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
  --var TRAINER_IMAGE="${REGISTRY}/npa-lerobot-vlm-rl:cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
  --var VLM_IMAGE="${REGISTRY}/npa-cosmos3-reason:cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
  --var EVAL_IMAGE="${REGISTRY}/npa-loop-eval:cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z" \
  --var ISAAC_IMAGE="${REGISTRY}/npa-isaac-lab:2.3.2.post1" \
  --var NPA_SIM2REAL_SIM_BACKEND=isaac \
  --var NPA_SIM2REAL_ISAAC_TASK=Isaac-Lift-Cube-Franka-v0 \
  --var BYO_POLICY_COMMAND='python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout' \
  --var BYO_TRAINER_COMMAND='python3 -m npa.workflows.sim2real.byo_isaac_trainer' \
  --var BYO_EVAL_COMMAND='python3 -m npa.workflows.sim2real.byo_isaac_eval' \
  --var INNER_ITERATIONS=3 \
  --var OUTER_ITERATIONS=2 \
  --var LOOP_OF_LOOPS_ITERATIONS=1 \
  --var NPA_SIM2REAL_EARLY_EXIT=0 \
  --var NPA_ENV_COUNT=10000 \
  --var ROLLOUT_COUNT=8 \
  --var STEPS_PER_ROLLOUT=6 \
  --var HELDOUT_ENV_COUNT=8 \
  --var NPA_SIM2REAL_HELDOUT_EVAL_LIMIT=0 \
  --var NPA_BYO_ISAAC_NUM_ENVS=1024 \
  --var NPA_BYO_ISAAC_ITERATIONS=150 \
  --var NPA_BYO_ISAAC_STEPS_PER_ENV=24 \
  --var NPA_SIM2REAL_CAMERA_VIEWS=primary,side,overhead \
  --var NPA_SIM2REAL_CAPTURE_WIDTH=640 \
  --var NPA_SIM2REAL_CAPTURE_HEIGHT=480 \
  --var NPA_SIM2REAL_ROLLOUT_CAPTURE_STRIDE=1 \
  --var NPA_SIM2REAL_HELDOUT_CAPTURE_STRIDE=20 \
  --var NPA_SIM2REAL_PNG_COMPRESS_LEVEL=3 \
  --var NPA_SIM2REAL_CAPTURE_FPS=10 \
  --var NPA_SIM2REAL_RERUN=1 \
  --var NPA_SIM2REAL_MCAP=1 \
  --var NPA_SIM2REAL_REQUIRE_VISUALIZATION=1 \
  --var NPA_SIM2REAL_K8S_JOB_TIMEOUT_S=0 \
  --var NPA_BYO_ISAAC_JOB_TIMEOUT_S=0 \
  --var NPA_SIM2REAL_K8S_GPU_CANDIDATES='RTX PRO 6000,L40S' \
  --var OMNI_KIT_ACCEPT_EULA=YES \
  --var ISAACSIM_ACCEPT_EULA=YES
```

Use `--plan-only` on the same command to materialize and validate without
applying the Job. Every `--var KEY=VALUE` is passed into the materialized
orchestrator environment; it is not limited to a small allowlist.

## Parameter contract

### Loop, evaluation, and thresholds

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `INNER_ITERATIONS` | `3` | integer ≥1 | rollout→critique→PPO passes per outer pass |
| `OUTER_ITERATIONS` | `2` | integer ≥1 | held-out evaluation and threshold decisions |
| `LOOP_OF_LOOPS_ITERATIONS` | `1` | integer ≥1 | Stage 13 external retrigger-cycle ceiling; one run records one cycle and whether another should be triggered |
| `NPA_SIM2REAL_EARLY_EXIT` | `0` | boolean | if true, stop outer passes after promotion; false runs fixed counts |
| `SUCCESS_THRESHOLD` | `0.50` | 0..1 fraction | aggregate held-out success-rate gate at Stage 11 |
| `NPA_BYO_ISAAC_SUCCESS_DIST_M` | `0.05` | 0.001..10 m | per-episode cube-to-goal success distance in Isaac |
| `ROLLOUT_COUNT` | `8` | integer ≥1 | policy rollouts per inner pass |
| `STEPS_PER_ROLLOUT` | `6` | integer ≥1 | policy steps per rollout |
| `NPA_ENV_COUNT` | `10000` | integer ≥1 | generated environments before split |
| `NPA_TRAIN_FRACTION` | `0.8` | 0..1 fraction | train share; remainder is held out |
| `HELDOUT_ENV_COUNT` | `8` | integer ≥1 | held-out episodes requested from Stage 10 |
| `NPA_SIM2REAL_HELDOUT_EVAL_LIMIT` | `0` | integer ≥0 | cap on held-out input rows; `0` means uncapped |

The two thresholds are intentionally different: a single Isaac episode succeeds
when final distance is below `NPA_BYO_ISAAC_SUCCESS_DIST_M`; Stage 11 promotes
only when the fraction of successful held-out episodes reaches
`SUCCESS_THRESHOLD`.

Candidate packaging is distinct from promotion. When PPO produced a real
checkpoint, `checkpoints/candidate/candidate.json` remains deployable and names
those exact weights even if Stage 11 records `loop_back_to_inner_loop`. In that
case `threshold_met` is false and `candidate_status` is
`below_threshold_deployable_candidate`; operators must not present it as a
promoted policy.

Fixed-count evidence run:

```bash
--var INNER_ITERATIONS=3 --var OUTER_ITERATIONS=2 --var NPA_SIM2REAL_EARLY_EXIT=0
```

Promotion-short-circuit run:

```bash
--var INNER_ITERATIONS=3 --var OUTER_ITERATIONS=5 --var NPA_SIM2REAL_EARLY_EXIT=1
```

### Real PPO

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `NPA_BYO_ISAAC_NUM_ENVS` | `1024` | 1..65536 environments | vectorized Isaac PPO environments |
| `NPA_BYO_ISAAC_ITERATIONS` | `150` | 1..1000000 iterations | RSL-RL optimization iterations |
| `NPA_BYO_ISAAC_STEPS_PER_ENV` | `24` | 1..16384 steps | rollout horizon per environment and iteration |
| `LEARNING_RATE` | `0.08` | positive scalar | VLM signal-adapter/no-signal-control step size; does not override the Isaac PPO optimizer |

The default PPO workload is 3,686,400 environment steps per inner pass. Stage 9
records these dimensions, training curves, and the real `model_*.pt` URI. Later
rollout/eval Jobs must load the exact bytes or fail closed.

`0.08` is intentional: it is the longstanding canonical adapter step used by
the runbook; the Python model/CLI default was synchronized to it from `0.05`.
The BYO Isaac trainer uses the value for its adapter-result provenance while
RSL-RL keeps the selected Isaac task's optimizer configuration. Training
evidence, candidate metadata, and the final report record the effective adapter
learning rate and this scope so it cannot be mistaken for a PPO optimizer
override.

### Cameras and recording

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `NPA_SIM2REAL_CAMERA_VIEWS` | `primary,side,overhead` | comma list; aliases `front,left,top` | synchronized front/operator, side, and oblique/top views |
| `NPA_SIM2REAL_CAPTURE_WIDTH` | `640` | 320..4096 px | RGB/depth width |
| `NPA_SIM2REAL_CAPTURE_HEIGHT` | `480` | 240..2160 px | RGB/depth height |
| `NPA_SIM2REAL_ROLLOUT_CAPTURE_STRIDE` | `1` | 1..10000 sim steps | policy-rollout frame sampling |
| `NPA_SIM2REAL_HELDOUT_CAPTURE_STRIDE` | `20` | 1..10000 sim steps | held-out frame sampling |
| `NPA_SIM2REAL_PNG_COMPRESS_LEVEL` | `3` | 0..9 | lossless PNG compression, not visual quality loss |
| `NPA_SIM2REAL_CAPTURE_FPS` | `10` | 0.1..240 fps | artifact/viewer timestamps |
| `NPA_SIM2REAL_RERUN` | `1` | boolean | emit `reports/sim2real.rrd` |
| `NPA_SIM2REAL_MCAP` | `1` | boolean | emit aligned `reports/sim2real.mcap` |
| `NPA_SIM2REAL_REQUIRE_VISUALIZATION` | `1` | boolean | fail Stage 14 if an enabled recording is missing |
| `NPA_SIM2REAL_RERUN_SERVE` | `1` | boolean | publish the completed `.rrd` through the shared authenticated viewer |
| `NPA_SIM2REAL_K8S_JOB_TIMEOUT_S` | `0` | integer ≥0 seconds | orchestrator/sibling deadline; `0` means no deadline |
| `NPA_BYO_ISAAC_JOB_TIMEOUT_S` | `0` | integer ≥0 seconds | Isaac rollout/train/eval wait; `0` means no deadline |

The zero timeout is intentionally uncapped: Kubernetes
`activeDeadlineSeconds` is omitted and the operator keeps polling. It is not an
ignore-failures mode—failed Job counters, deleted Jobs, kubectl errors, image or
runtime failures, and non-zero component exits still terminate the run. To opt
into a four-hour deadline, pass
`--var NPA_SIM2REAL_K8S_JOB_TIMEOUT_S=14400` (and set
`NPA_BYO_ISAAC_JOB_TIMEOUT_S` when the nested Isaac wait should have the same
positive deadline).

Every frame records view name, Isaac world pose and intrinsics, environment and
episode, pixel resolution, frame index, simulation step/timestamp, and candidate
checkpoint URI/SHA. The Rerun scene combines measured Isaac RGB-D point clouds
with clearly labeled nominal table/cube/goal/Franka task context.

### GPU routing

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `NPA_SIM2REAL_K8S_GPU_PRODUCT` | `NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition` | exact cluster `nvidia.com/gpu.product` label for RTX PRO 6000 or L40S | preferred RT-core product |
| `NPA_SIM2REAL_K8S_GPU_CANDIDATES` | `RTX PRO 6000,L40S` | ordered comma list of compatible aliases or exact labels | capacity fallback order, normalized against live node labels |
| `NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS` | `16` | integer ≥1 | maximum concurrently dispatched sibling GPU tasks |
| `NPA_ENVGEN_SHARD_COUNT` | `16` | integer ≥1 | indexed Stage 4 environment-generation shards |

Isaac jobs reject non-RT products before apply. A fallback is attempted only
after concrete Kubernetes capacity evidence; image, credential, and runtime
failures stay attached to the selected product and fail closed.

Direct Kubernetes submit validates `VLM_REASON2_IMAGE`, `VLM_REASON3_IMAGE`,
and `NPA_RERUN_VIEWER_IMAGE` as registry-qualified real-path images before any
Job is dispatched. A `VLM_IMAGE` override supplies both reason-image aliases
unless they are set explicitly. The viewer image is optional when either
`NPA_SIM2REAL_RERUN` recording or `NPA_SIM2REAL_RERUN_SERVE` auto-serve is
disabled, and the registry pull-secret set follows the same effective guard.

Architecture markers in an image tag are also fail-closed evidence. RTX PRO
6000 (CUDA major 12) requires explicit `sm120` SASS or `compute120` PTX when a
tag advertises architecture markers; `sm100`/`sm103` SASS is not portable to
it. L40S (sm_89) accepts the repository-proven same-major `sm80`/`sm89` SASS or
`compute80`/`compute89` PTX, but never `sm90` SASS. Version-only Isaac tags stay
backward compatible because they make no architecture claim, while Isaac's
separate RT-core filter still rejects H100, H200, B200, and B300.

An untolerated taint, a cordoned node, or a NotReady node does **not** trigger a
GPU-product fallback by itself. These conditions describe cluster placement or
health, not proof that the selected GPU product lacks capacity. Switching
products could hide a broken node pool or bypass an operator policy, so the
workflow fails closed unless the same scheduler evidence also contains the
narrow recognized signal: insufficient `nvidia.com/gpu` or a concrete
product-selector/node-affinity mismatch. Inspect and remediate with:

```bash
kubectl get nodes -o wide
kubectl describe node <node>
kubectl -n default describe job <job>
kubectl -n default get events --sort-by=.lastTimestamp
```

Check readiness, `spec.unschedulable`, taints/tolerations, GPU Operator health,
and the exact `nvidia.com/gpu.product` label, then resubmit the same runbook.

## Monitor and diagnose

```bash
npa/.venv/bin/npa workbench workflow status "${RUN}" --watch
kubectl -n default get jobs -l sim2real.local/run-id="${RUN}" -o wide
kubectl -n default get pods -l sim2real.local/run-id="${RUN}" -o wide
kubectl -n default logs job/"${RUN}" --follow
```

The orchestrator Job name may be DNS-truncated; use the `job_name` printed by
submit when that happens. Stage status is also persisted at:

```text
s3://<bucket>/sim2real-b/<run-id>/state/workflow_state.json
```

On `ImagePullBackOff`/401, refresh the registry pull secret and delete only the
failed run-scoped Job so the same documented submit can be repeated. On
`Insufficient nvidia.com/gpu`, inspect Job events; the submitter will try every
compatible RTX PRO 6000/L40S label before returning capacity exhaustion.
Taint-, cordon-, or NotReady-only scheduling events stop on the selected
product; fix node health/schedulability or the required toleration rather than
expecting the workload to move to another GPU family.
If stale ambient `AWS_ACCESS_KEY_ID` values are denied during source staging,
the submitter retries the HMAC pair in `~/.npa/credentials.yaml`; it never logs
either credential value.

## Artifact and policy access

The run root is `s3://<bucket>/sim2real-b/<run-id>/`. Key objects are:

| Object | Purpose |
| --- | --- |
| `reports/sim2real-report.json` | all-stage report, tiers, Jobs, digests, metrics, runtime parameters |
| `reports/sim2real.rrd` | Rerun 3D scene, synchronized cameras, charts, timeline, provenance |
| `reports/sim2real.mcap` | aligned Foxglove/Lichtblick cameras, point cloud, signals, provenance |
| `eval/heldout/report.json` | per-env result and exact loaded checkpoint SHA/size |
| `outer_loop/decision.json` | aggregate threshold and promotion decision |
| `checkpoints/candidate/candidate.json` | deployable candidate metadata, promotion status, and authenticated access instructions |
| `byo-trainer/**/model_latest.pt` | real learned policy weights |

Download and inspect without presigned URLs:

```bash
PREFIX=s3://${BUCKET}/sim2real-b/${RUN}
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/reports/sim2real-report.json" - | jq .
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/eval/heldout/report.json" - | \
  jq '{success_rate, policy_inference_provenance, capture, camera_metadata}'
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/checkpoints/candidate/candidate.json" - | \
  jq '{deployable_policy, candidate_status, threshold_met, promotion_decision, policy_checkpoint_uri, policy_checkpoint_sha256, policy_checkpoint_size_bytes}'
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/reports/sim2real.rrd" /tmp/sim2real.rrd
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/reports/sim2real.mcap" /tmp/sim2real.mcap
```

Open the recordings:

```bash
npa/.venv/bin/rerun /tmp/sim2real.rrd
npa/.venv/bin/npa workbench lichtblick serve --input-path /tmp/sim2real.mcap --execute
```

The visualization is evidence, not an inference engine. Its held-out camera and
point-cloud streams are generated by Isaac after loading the candidate weights;
the recording embeds the same URI/SHA/size as `candidate.json` and the held-out
report so that relationship can be independently checked.

## Cleanup

Cleanup is run-scoped. It does not delete the S3 evidence tree:

```bash
kubectl -n default delete job -l sim2real.local/run-id="${RUN}"
```

Remove the shared hosted viewer only when intended:

```bash
npa/.venv/bin/npa workbench sim2real rerun serve --run-id "${RUN}" --destroy
```

## Troubleshooting guardrails

- A real-component stage without a registry-qualified image or real command is
  rejected before submit. The canonical YAML keeps
  `NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS=1`, so component provenance and candidate
  checkpoint byte hashing also fail closed.
- A supplied candidate checkpoint that cannot be loaded is a hard failure; the
  policy/eval path never substitutes a stock policy.
- Missing or malformed Cosmos, Isaac, trainer, Rerun, or MCAP evidence must be
  fixed and rerun; a process exit code alone is not proof.
- `NPA_SIM2REAL_HELDOUT_EVAL_LIMIT=0` is uncapped, not zero episodes.
- `SUCCESS_THRESHOLD` is a fraction; `NPA_BYO_ISAAC_SUCCESS_DIST_M` is metres.
- H100/H200/B200/B300 are invalid for Isaac rendering even if idle.
- Stage 12 remains an intentional external-validation stub and is the only
  expected non-`WORKS` canonical stage.
