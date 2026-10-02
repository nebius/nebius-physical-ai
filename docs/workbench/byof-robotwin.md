# RoboTwin customer runtime

The RoboTwin image contains an Ubuntu bootstrap and NPA runtime-fetch code.
After an explicit customer decision, it downloads a locked runtime and runs
upstream `beat_block_hammer` with `demo_clean` on exactly one STRICT-bound
RTX PRO 6000 Blackwell. The collector searches for one successful seed, replays
its planned trajectory in SAPIEN, and exports native HDF5 and MP4. Validation
requires successful planning **and replay**, aligned finite state/action arrays
with articulated motion, three decoded camera streams, and a fully decoded,
nonstatic MP4 with the matching frame count. B200 is not a compatible target.

This is an operator-run BYOF candidate, not a supported agent or CLI capability.
The public `npa workbench byof run` command refuses RoboTwin execution, and the
outer `npa workbench workflow submit` path remains blocked on remote-source and
worker identity proof. The standalone script below invokes the guarded inner
launcher from the operator host. Its results do not enable either public path
or remove the supported-release quarantine.

## Retained operator evidence and readiness

The standalone operator path completed one `beat_block_hammer` episode on an
RTX PRO 6000 using this development image:

`ghcr.io/nebius/nebius-physical-ai/npa-robotwin@sha256:952ab5101953b45e4a2cfa323ac6934ad447af8443858214220cd349e5564a34`

The [image build](https://github.com/nebius/nebius-physical-ai/actions/runs/35818763939)
used source `a2fdc0b583f6201adc89b6aa15a987c32213ed84`. Retained private run
evidence records successful planning and replay on seed 1 after attempts at
seeds 0 and 1, 123 aligned state/action pairs, one native HDF5 episode, 124
distinct decoded video frames, and three PNGs. Readback verified all ten
objects, totaling 18,350,266 bytes. OIDN errors and visible grain remain in the
record; this demonstrates functional collection and replay for one task, not
renderer quality, policy training, or a benchmark success rate.

[`byof-robotwin.readiness.json`](../../workflows/testing/byof-robotwin.readiness.json)
tracks the checked-in normal workflow. Its runtime prerequisites remain
`unverified` because the supported submission path has not completed that
workload. `UNVALIDATED_PUBLICATION_TOOLS` and the release tag ending in
`-unbuilt` continue to exclude RoboTwin from supported publication. Those
release-policy markers do not imply that the development digest above was
never built. New operator runs still require their own authorization, exact
image verification, input probes, placement, storage, and cleanup checks.

## Inputs and redistribution

The image retains its zero-vendor-payload boundary. Its pinned Ubuntu 22.04
base, 84 Ubuntu `main` packages and corresponding-source annex are unchanged.
Public image publication still requires the complete byte, native-content,
security, license, SBOM and provenance gates. No vendor runtime, source, asset,
customer decision, credential, cache or generated output is baked.

The customer runtime lock is
[`runtime-lock.json`](../../npa/docker/workbench/robotwin/runtime-lock.json).
It records 474 exact archive URLs, sizes and SHA-256 values: the Ubuntu runtime
package closure, Python 3.10 distributions, and NVIDIA CUDA 12.8.1 compiler,
runtime headers and CCCL archives. PyTorch `2.7.1+cu128` supports the selected
Blackwell target and requires cuDNN `9.7.1.26`; the customer notice names the
matching cuDNN 9.7.1 agreement. CuRobo is compiled for `sm_120` on the customer
runtime. Runtime packages total approximately 5.15 GB compressed.

| Boundary | Runtime treatment |
| --- | --- |
| Source | Official RoboTwin commit `96c1feab536306b50c26af200044fcdf126e8904` (MIT) and CuRobo v0.7.8 commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (NVIDIA Source Code License for cuRobo). Checkouts must match the locked commit, tree and license hash before execution. |
| Baked runtime | Only the reviewed neutral Ubuntu bootstrap and NPA code. All simulation, CUDA, cuDNN and Python application bytes are fetched after customer authorization. |
| Weights | None. This capability collects demonstrations without a model checkpoint. |
| Assets | Public, ungated `TianxingChen/RoboTwin2.0@785feb15aa4a4f532395ad2b1d2be5f28cb561ad`: `embodiments.zip` and `objects.zip`, approximately 3.96 GB combined. The repository card declares MIT; exact sizes and SHA-256 values are verified before extraction. No Hugging Face token or separate asset-acceptance flag is required. |
| Runtime cache | A unique owner-only directory for one customer/run on the worker. Runtime artifacts are verified before installation; the ready marker is published last. It is node-local and ephemeral, not durable storage or a redistributable image. |
| Outputs | Native HDF5, MP4, PNG frames and JSON evidence go only to the authorized output prefix. No inspected term restricts these generated artifacts independently; underlying CuRobo use remains noncommercial research/evaluation. |

The task retains upstream configuration except the requested one-episode
collection and the authorized output path. It invokes the pinned upstream
asset-path renderer and applies the two compatibility edits from upstream's
installation script. Pointcloud collection is disabled in `demo_clean`, so
this capability does not install the optional PyTorch3D pointcloud dependency.

## Customer decision and launch

An authorized customer representative must review and accept the
[CUDA 12.8.1 EULA](https://docs.nvidia.com/cuda/archive/12.8.1/eula/index.html),
[cuDNN 9.7.1 SDK agreement](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/reference/eula.html),
and [CuRobo v0.7.8 license](https://github.com/NVlabs/curobo/blob/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/LICENSE)
for this run's noncommercial containerization and technical validation activity.
NPA and the infrastructure manager do not accept those terms for the customer.
The existing hosted authorization-boundary interface remains available.

On the authenticated customer operator terminal, first display the notice:

```bash
npa/.venv/bin/python -m npa.orchestration.npa_workflow.robotwin_customer notice
```

Create an owner-only directory outside the checkout, then record the customer
choice using the same run and customer scope as the private runtime context:

```bash
npa/.venv/bin/python -m npa.orchestration.npa_workflow.robotwin_customer \
  customer-decision --run-id "$ROBOTWIN_RUN_ID" \
  --customer-scope-id "$ROBOTWIN_CUSTOMER_SCOPE" \
  --expires-at "$ROBOTWIN_AUTHORIZATION_EXPIRY" \
  --decision accept --receipt "$ROBOTWIN_PRIVATE_DIR/customer-decision.json"
```

The acceptance command requires the customer to type `accept` at a terminal.
Use `--decision decline`, or take no action, to decline. There is no default
acceptance. The private receipt binds the exact runtime manifest, terms,
customer scope, run, intended activity, issuance and expiry. Keep it outside
Git, image layers, workflow YAML, caches, logs and uploaded artifacts. The
retired unsigned entitlement-file variable remains rejected.

Once the exact image, STRICT GPU placement, storage, and private context are
ready, launch the existing inner profile from the operator host:

```bash
npa/.venv/bin/python npa/scripts/run_robotwin_customer.py \
  --context "$ROBOTWIN_PRIVATE_DIR/runtime-context.json" \
  --customer-decision "$ROBOTWIN_PRIVATE_DIR/customer-decision.json"
```

The launcher validates the customer decision before upstream probes or launch,
probes the exact anonymous payload endpoints, verifies the immutable image,
and reuses the existing BYOF submit, wait and cleanup path. Missing, declined,
stale or mismatched decisions refuse before governed side effects. A receipt
is consumed once at submission; a failed consumed attempt requires a new
customer decision for that run and manifest. The separate CPU outer workflow
continues to refuse an unverified remote NPA source population; this local
entrypoint does not install NPA source in a remote controller.

The host launcher removes its temporary credential copies after success and
after any failure before it calls the inner launcher. Once launch has been
attempted, a failure can leave remote resources in an uncertain state, so the
owner-private recovery configuration is retained. Verify terminal state and
owned-resource cleanup before removing that directory. The original context
and customer receipt are never deleted by this cleanup.

The installed runtime retains private compiler/simulator diagnostics under
its owner-only `logs` directory. These logs are excluded from uploaded output.
A successful run uploads `robotwin-smoke.json`, native artifacts and a final
`summary.json`; hashes and decoded frame counts make the result reviewable.

## Validation

```bash
npa/.venv/bin/python -m pytest -q \
  npa/tests/docker/test_robotwin_runtime_bootstrap.py \
  npa/tests/docker/test_robotwin_runtime_delivery.py \
  npa/tests/workflows/test_robotwin_customer_runner.py
```

The existing live path remains registered in
`npa/tests/e2e/test_byof_onboarding_live_e2e.py`. Its success requires the exact
immutable image and real native workload evidence, followed by owned-resource
cleanup. Acceptance and infrastructure evidence remain private run records.
