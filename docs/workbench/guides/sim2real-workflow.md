# Compositional Sim2Real workflow

The single canonical operator spec is
[`npa/workflows/workbench/npa-workflows/sim2real.yaml`](../../../npa/workflows/workbench/npa-workflows/sim2real.yaml), beside the Physical AI Data Factory spec. It is a normal
`npa.workflow/v0.0.1` graph. The normal planner renders each state into a
SkyPilot task, the normal runtime persists its S3 ledger, and `--resume`
reconciles completed or in-flight waves. There is no Sim2Real detector bypass,
driver pod, or hidden sibling-Job controller on this path.

## What the graph executes

The visible states preserve the canonical 14-stage contract:

1. validate the task-aligned Isaac trigger and seed manifest;
2. consume the asset, task, robot, camera, and strict-success contract;
3. run real input-conditioned Cosmos Transfer 2.5;
4. generate raw environment shards in parallel GPU tasks;
5. curate and seal disjoint train, validation, and untouched gold sets;
6. publish the explicit token/scenario handoff;
7. execute real Isaac policy rollouts with primary/side/overhead cameras;
8. execute real Cosmos Reason2 and Reason3 lanes in parallel;
9. merge bounded temporal signals, run genuine BYO Isaac RSL-RL PPO, and use
   only validation scenarios for checkpoint selection;
10. load that exact checkpoint for real Isaac evaluation on untouched gold;
11. record the unchanged strict 5 cm stable-placement metric and loop decision;
12. record the external physical-validation boundary as `SEAM`, never `WORKS`;
13. persist the retrigger/loop record; and
14. publish the final report, Rerun recording, and MCAP recording.

Policy quality is reported honestly but is not a workflow-plumbing success
gate. Long 3x3 PPO-500 efficacy studies are post-merge work. A reduced 1x1 run
may lower counts to prove every real boundary and artifact handoff, while the
strict metric and sealed gold contract remain unchanged.

## Required operator inputs

The committed spec is tenant-neutral. At submission, set the S3 bucket/trigger,
six registry-qualified immutable images, and the prewarmed Isaac cache PVC.
The image adapters require `NPA_TASK_IMAGE` to contain `@sha256:` and attest the
image source SHA. Isaac also verifies its read-only content-addressed runtime
cache before simulator startup. The operator must also pass explicit NVIDIA
acceptance through `omni_kit_accept_eula` and `isaacsim_accept_eula`; both
committed defaults are empty. The inline Isaac adapter validates both values
before starting Kit, so the workflow never accepts terms on the operator's
behalf and never falls through to an interactive prompt in an unattended Job.

`controller_image` is the digest-pinned, CPU-only `npa-sim2real-control` image
built from `npa/docker/workbench/sim2real-control/Dockerfile`. It deliberately
contains no Genesis, Isaac, CUDA, or trainer runtime; using a GPU solution image
for CPU bookkeeping is unsupported because its cold pull can exhaust a CPU
node before Stage 1. It does contain the non-root SkyPilot Kubernetes bootstrap
prerequisites, which are part of the schedulable-image contract and are tested
through the same standard runtime path used live. Its exact source is installed
as a site-packages path as well as exported through `PYTHONPATH`, because the
SkyPilot setup login shell is allowed to rebuild its environment. Transfer,
EnvGen, Reason, Isaac, and visualization each retain their own immutable image
at the corresponding workflow boundary.

SkyPilot 0.12.2 still performs its Kubernetes bootstrap through passwordless
`sudo`. For this CPU-only, ephemeral task image the task pod is the security
boundary; the finite exception and prohibited service/public-ingress uses are
enforced in `packaging-contract.yaml`. Separately, the Isaac cache warmer and
the retained standalone BYO Isaac Job builders run as uid/gid 1000 (the warmer
uses an fsGroup-owned PVC); none of the Sim2Real Isaac paths may override
`runAsUser: 0`.

```bash
SPEC=npa/workflows/workbench/npa-workflows/sim2real.yaml

npa/.venv/bin/npa workbench workflow validate-spec "$SPEC"
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" --waves \
  --run-id <run-id>

npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  --runtime \
  --run-id <run-id> \
  --resume \
  --max-wait-seconds 0 \
  --var bucket=<bucket> \
  --var trigger_uri=s3://<bucket>/<task-aligned-trigger>/ \
  --var seed_manifest_uri=s3://<bucket>/<task-aligned-trigger>/dataset-manifest.json \
  --var controller_image=<registry/controller@sha256:...> \
  --var transfer_image=<registry/transfer@sha256:...> \
  --var envgen_image=<registry/envgen@sha256:...> \
  --var reason_image=<registry/reason@sha256:...> \
  --var isaac_image=<registry/isaac@sha256:...> \
  --var viewer_image=<registry/viewer@sha256:...> \
  --var isaac_cache_pvc=<pvc> \
  --var omni_kit_accept_eula=YES \
  --var isaacsim_accept_eula=YES \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

Use `--var outer_iterations=1 --var inner_iterations=1` and reduced scenario/PPO
values for the merge-proof ladder. Omit those overrides for production-size
defaults. `--max-wait-seconds 0` deliberately means no arbitrary per-wave
deadline; managed-job state and the S3 runtime ledger remain observable.

## Durable evidence

The runtime ledger is
`s3://<bucket>/sim2real/<run-id>/npa-workflow/runtime.json`. Stage outputs are
explicit S3 inputs to the next state. Each stage publishes a canonical
`components/stage_XX.json` pointer backed by an immutable
`components/history/stage_XX/<content-sha256>.json` record. Restarting at the
Stage 8/9 barrier or during Stage 14 reuses successful output only after the
runtime validates its declared artifacts.

Final artifacts are:

- `reports/sim2real-report.json`
- `reports/sim2real.rrd`
- `reports/sim2real.mcap`
- the exact selected checkpoint and validation/gold lineage named by the report.

See [the architecture/resume contract](../../architecture/sim2real-compositional-workflow.md)
for execution ownership and compatibility details.
