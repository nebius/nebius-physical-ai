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
| 1 | validate the task-aligned Isaac seed dataset | `stage_01_trigger/task-dataset-manifest.json` |
| 2 | normalize task, scene, robot, and camera contracts | `stage_02_assets/task-contract.json`, `consumed_*_spec.json` |
| 3 | real Cosmos Transfer augmentation, coupled to scenario/VLM lineage | `augment/manifest.json`, generated frames/video |
| 4 | generate and curate scenario configs | `envs/manifest/curation-manifest.json`, `envs/raw/` |
| 5 | disjoint stratified train/validation/gold splits | `envs/{train,validation,gold-heldout}/envs.jsonl`, `envs/manifest/split-manifest.json` |
| 6 | record scenario features and their honest state-PPO consumer | `tokens/manifest.json` |
| 7 | real Isaac rollouts across curated configs | `actions/train/`, applied config digests, simulator event telemetry |
| 8 | dual Cosmos Reason critique plus simulator-grounded temporal credit | `vlm_eval/train/**/*.json`, `training_signal/train/**/*.json` |
| 9 | real RSL-RL PPO plus fixed-validation checkpoint selection | `byo-trainer/**/{model_*.pt,model_latest.pt,ppo-telemetry.json}`, `checkpoints/validation-selection/` |
| 10 | exact selected-checkpoint gold Isaac inference | `eval/gold-heldout/**/report.json` and RGB-D captures |
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
- `storage.sim2real_stock_trigger_uri`: a populated Isaac lift-cube seed prefix.
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
kubectl auth can-i patch jobs.batch \
  --as=system:serviceaccount:default:agent-sa -n default
kubectl auth can-i list workloads.kueue.x-k8s.io \
  --as=system:serviceaccount:default:agent-sa -n default
kubectl -n default get pvc npa-sim2real-isaac-cache -o json
```

Both authorization commands must return `yes`. The controller Role needs
`create`, `delete`, `get`, `list`, `watch`, and `patch` on `batch/jobs`.
`patch` is load-bearing: durable reconciliation records structured heartbeats
and adopts exact-identity sibling Jobs through the Kubernetes API. The
Role also needs `list` on `kueue.x-k8s.io/workloads` so the controller can
attest generated Workload admission, assigned flavor, and terminal state. A
Kueue API denial retains its structured status/reason and fails closed; it is
never reported as an absent Workload. The Sim2Real health check fails before
launch when the configured service account lacks either permission. It also
requires `NPA_SIM2REAL_ISAAC_CACHE_PVC` to name a Bound `ReadWriteMany` claim.
Warm that claim once on a CPU node with the exact digest-pinned Isaac image and
the operator's EULA acceptance; Isaac GPU Jobs mount it offline and read-only.
See [the durable-controller contract](./sim2real-durable-controller.md#isaac-runtime-dependency-closure).

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
TRIGGER=s3://${BUCKET}/sim2real-triggers/<batch>/isaac-lift-cube-franka/

npa/.venv/bin/npa workbench workflow submit npa/workflows/sim2real.yaml \
  --run-id "${RUN}" \
  --var NPA_SIM2REAL_BUCKET="${BUCKET}" \
  --var AWS_ENDPOINT_URL="${ENDPOINT}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_URI="${TRIGGER}" \
  --var NPA_SIM2REAL_TRIGGER_DATASET_ID=npa/isaac-lift-cube-franka-seed-v1 \
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
  --var OUTER_ITERATIONS=3 \
  --var LOOP_OF_LOOPS_ITERATIONS=1 \
  --var NPA_SIM2REAL_EARLY_EXIT=0 \
  --var NPA_ENV_COUNT=10000 \
  --var ROLLOUT_COUNT=64 \
  --var STEPS_PER_ROLLOUT=32 \
  --var NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS=300 \
  --var NPA_COSMOS_REASON_MAX_FRAMES=32 \
  --var NPA_COSMOS_REASON_MAX_NEW_TOKENS=8192 \
  --var VALIDATION_ENV_COUNT=64 \
  --var HELDOUT_ENV_COUNT=64 \
  --var NPA_SIM2REAL_HELDOUT_EVAL_LIMIT=0 \
  --var NPA_BYO_ISAAC_NUM_ENVS=1024 \
  --var NPA_BYO_ISAAC_ITERATIONS=500 \
  --var NPA_BYO_ISAAC_VALIDATION_INTERVAL=100 \
  --var NPA_BYO_ISAAC_STEPS_PER_ENV=24 \
  --var NPA_BYO_ISAAC_PPO_LEARNING_RATE=0.001 \
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

The trigger prefix must contain `task-dataset-manifest.json` with schema
`npa.sim2real.task_seed_dataset.v1`, the exact task ID, dataset ID, task-contract
digest, Isaac source run, positive trajectory/action/camera counts, and an S3 URI
to a sample rollout manifest containing actions and camera observations. The
canonical real path fails closed if this contract is absent or if a PushT source
is paired with the Franka lift task. The normalized task contract also fixes the
embodiment, object/scale, goal and physics ranges, cameras, state observation and
action terms, 5 cm stable-placement definition, and train/eval parity ID.

Environment generation rejects schema/digest mismatches, unreachable or
intersecting placements, missing assets/cameras, unusable physics, and duplicate
config digests. It balances difficulty strata and emits count/reason/coverage and
leakage evidence. Training rotates the full curated train distribution across
parallel environments and reset episodes. The trainer transports that large split
by its existing S3 URI plus an exact SHA-256, verifies the bytes inside the Isaac
pod, and refuses an oversized embedded fallback; validation and gold inference
require the exact runtime-applied digest set to match their reported rows.

## Parameter contract

### Loop, evaluation, and thresholds

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `INNER_ITERATIONS` | `3` | integer ≥1 | rollout→critique→PPO passes per outer pass |
| `OUTER_ITERATIONS` | `3` | integer ≥1 | validation decisions followed by final-gold evaluation |
| `LOOP_OF_LOOPS_ITERATIONS` | `1` | integer ≥1 | Stage 13 external retrigger-cycle ceiling; one run records one cycle and whether another should be triggered |
| `NPA_SIM2REAL_EARLY_EXIT` | `0` | boolean | if true, stop outer passes after promotion; false runs fixed counts |
| `SUCCESS_THRESHOLD` | `0.50` | 0..1 fraction | aggregate held-out success-rate gate at Stage 11 |
| `NPA_BYO_ISAAC_SUCCESS_DIST_M` | `0.05` | 0.001..10 m | per-episode cube-to-goal success distance in Isaac |
| `ROLLOUT_COUNT` | `64` | integer ≥1 | distinct curated-scenario policy rollouts per inner pass |
| `STEPS_PER_ROLLOUT` | `32` | integer ≥1 | simulator-grounded decision/event samples per rollout |
| `NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS` | `300` | integer ≥ sampled points | task-length Isaac steps over which the decision/event samples are evenly taken |
| `NPA_COSMOS_REASON_MAX_FRAMES` | `32` | integer ≥ `STEPS_PER_ROLLOUT` | exposes every sampled event frame to each Cosmos model; smaller real-tier windows fail before submit |
| `NPA_COSMOS_REASON_MAX_NEW_TOKENS` | `8192` | integer ≥ 64 × sampled points | output budget for compact indexed per-event JSON; complete event rows survive a token-truncated outer object, while missing/malformed entries are rejected at zero confidence and never filled from the rollout summary |
| `NPA_ENV_COUNT` | `10000` | integer ≥1 | generated environments before split |
| `NPA_TRAIN_FRACTION` | `0.8` | 0..1 fraction | exact stratified train share; remainder is divided between validation and untouched gold-heldout |
| `VALIDATION_ENV_COUNT` | `64` | integer ≥1 | fixed validation episodes used only for checkpoint ranking |
| `HELDOUT_ENV_COUNT` | `64` | integer ≥1 | untouched gold episodes requested from final Stage 10 |
| `NPA_SIM2REAL_HELDOUT_EVAL_LIMIT` | `0` | integer ≥0 | cap on held-out input rows; `0` means uncapped |

Isaac seed `0` is a valid reproducibility value, not an unset sentinel. The
canonical trainer applies it to both the environment and the native RSL-RL
agent; fixed-validation and gold-evaluation Jobs apply it to Isaac, Torch, and
NumPy before environment creation. Runtime logs must therefore contain
`ROBOT_SEED_APPLIED 0` or `EVAL_SEED_APPLIED 0`; `Environment seed: None` is a
failed reproducibility check.

Stage 8 partitions every event label into `vlm_accepted_steps` or
`vlm_rejected_or_downweighted_steps`. Acceptance requires a model-local,
non-broadcast critique with confidence at least 0.5, no dual-model
disagreement, and consistency with simulator truth. The calibration manifest
also reports overlapping reason counts for missing/malformed, low-confidence,
disagreeing, summary-broadcast, and simulator-contradicting labels. Rejected
labels never override simulator-grounded reward; missing/malformed labels have
zero VLM contribution.

The two thresholds are intentionally different: a single Isaac episode succeeds
when its final distance is below `NPA_BYO_ISAAC_SUCCESS_DIST_M` and the placement
is stable; Stage 11 promotes
only when the fraction of successful held-out episodes reaches
`SUCCESS_THRESHOLD`.

Candidate packaging is distinct from promotion. When PPO produced a real
checkpoint, `checkpoints/candidate/candidate.json` keeps downloadable exact
weights even below threshold (`policy_bytes_available=true`), but
`deployable_policy=false` until every promotion gate passes. Validation ranks
checkpoints with strict success dominant; the final gold split is never used for
selection. Diagnostic 10/15/20 cm distances never relax the strict 5 cm gate.

Fixed-count evidence run:

```bash
--var INNER_ITERATIONS=3 --var OUTER_ITERATIONS=3 --var NPA_SIM2REAL_EARLY_EXIT=0
```

Promotion-short-circuit run:

```bash
--var INNER_ITERATIONS=3 --var OUTER_ITERATIONS=5 --var NPA_SIM2REAL_EARLY_EXIT=1
```

### Real PPO

| Variable | Default | Allowed value / unit | Operational effect |
| --- | ---: | --- | --- |
| `NPA_BYO_ISAAC_NUM_ENVS` | `1024` | 1..65536 environments | vectorized Isaac PPO environments |
| `NPA_BYO_ISAAC_ITERATIONS` | `500` | 1..1000000 iterations | RSL-RL optimization iterations per inner pass |
| `NPA_BYO_ISAAC_STEPS_PER_ENV` | `24` | 1..16384 steps | rollout horizon per environment and iteration |
| `NPA_BYO_ISAAC_VALIDATION_INTERVAL` | `100` | positive integer iterations | validation sweep interval over durable `model_*.pt` checkpoints; the final numbered checkpoint is always included |
| `NPA_BYO_ISAAC_PPO_LEARNING_RATE` | `0.001` | positive scalar | native RSL-RL optimizer initial rate before its adaptive schedule |
| `LEARNING_RATE` | `0.08` | positive scalar | VLM signal-adapter/no-signal-control step size; does not override the Isaac PPO optimizer |

The canonical 10,000 scenarios split exactly into 8,000 train, 1,000 validation,
and 1,000 gold-heldout records. Stage 5 downloads and hashes every expected Stage
4 raw shard before curation; both manifests carry the shard names, row counts, and
SHA-256 values.

The default PPO workload is 12,288,000 environment steps per inner pass. Stage 9
records structured per-iteration return, value/surrogate losses, entropy, noise,
task rewards, distances, termination rates, and the real `model_*.pt` URI. Fixed
periodic validation adds authoritative reach/contact/stable-grasp/lift/place rates
to checkpoint comparison and Rerun; those rates are not inferred from reward
proxies. Later rollout/eval Jobs must load the exact bytes or fail closed.

`0.08` is intentional: it is the longstanding canonical adapter step used by
the runbook; the Python model/CLI default was synchronized to it from `0.05`.
The BYO Isaac trainer uses the value for its adapter-result provenance while
RSL-RL separately receives `NPA_BYO_ISAAC_PPO_LEARNING_RATE` and records it with
its entropy/noise/reward settings. The adapter and native optimizer scopes remain
separately named in every result.

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
avoid a transient final-slot race in an exact-size parallel batch, concrete
compatible-GPU capacity failures are retained in provenance and the ordered
products are rechecked without a retry limit. A positive timeout instead makes
one ordered capacity pass and then fails closed. To opt into a four-hour
deadline, pass
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
| `envs/manifest/{curation-manifest.json,split-manifest.json}` | curation coverage, strata, and no-leakage proof |
| `checkpoints/validation-selection/*.json` | fixed-validation checkpoint ranking |
| `eval/gold-heldout/**/report.json` | gold per-env result and exact loaded checkpoint SHA/size |
| `outer_loop/decision.json` | aggregate threshold and promotion decision |
| `checkpoints/candidate/candidate.json` | packaged candidate bytes/provenance, honest deployability/promotion status, and authenticated access instructions |
| `byo-trainer/**/{model_*.pt,model_latest.pt,ppo-telemetry.json}` | enumerated real weights, latest compatibility alias, and PPO curves |

Download and inspect without presigned URLs:

```bash
PREFIX=s3://${BUCKET}/sim2real-b/${RUN}
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/reports/sim2real-report.json" - | jq .
aws --endpoint-url "${ENDPOINT}" s3 cp "${PREFIX}/eval/gold-heldout/outer-03/report.json" - | \
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
