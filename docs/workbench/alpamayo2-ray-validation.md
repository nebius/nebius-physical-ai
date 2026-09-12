# Alpamayo Ray workflow validation

The scenario sweep and baseline/refinement templates completed 20 real
Alpamayo 2 Super inferences on two separate RTX PRO 6000 GPUs. Each workflow
used one Ray GPU actor; CPU Ray tasks reduced the measured errors. The
Hugging Face model and gated dataset were fetched at runtime after the
operator's access check passed.

## Execution and evidence

Two isolated Kubernetes Jobs ran `npa workbench workflow run-spec --execute
--persist-state --require-inputs` against the new workflow YAMLs. This proves
the workflow executor, Ray application, real model inference, and S3 stage
handoffs on the selected RTX cluster. It does not prove the standard SkyPilot
`submit` control plane, the templates' default B200 allocation, or a shared
multi-node Ray cluster.

- Sweep: samples `0,1` × seeds `42,43` × steps `10,20`: **8 cases**.
- Hard-case baseline: samples `0,1` × seeds `42,43` × steps `10`: **4 cases**.
- Refinement: both samples × the same seeds × steps `20,30`: **8 cases**.
  The live override `minimum_ade=0.0` exercised refinement for both samples.
  Their baseline means were below the checked-in `2.0` default; that default
  would correctly produce an empty refinement on this small dataset.

The accepted image digest was
`sha256:2164450f8baf57d8798f64063ea27bf11611f5b695c467de0c2e319e3134ebd5`.
The staged source archive was independently checked in each worker against
SHA-256 `a04ec3f2ff53020ca90d0ff5873b4e7469921a37ed2ad94b048741228f14ed74`.
The final sweep driver and CLI are byte-identical to that snapshot. Later
changes tightened finite-metric validation, corrected renderer dependency
probe formatting, and updated documentation/tests. The final artifact
validator was run against all 20 downloaded real outputs; the GPU runs were
not repeated for those later changes.

Each of the three report hashes passed the independent live artifact verifier.
All 20 case JSON/PNG artifacts decoded and matched their requested sample,
seed, pinned model/dataset revisions, projection shape, and error measurements.
Eight refinement comparisons and six per-setting ADE summaries were also
independently recalculated from downloaded measurements.

## Measured results

ADE is average displacement error; lower is better. Values below are means
over the two explicit seeds, in meters.

| Manifest sample | 10 steps | 20 steps | 30 steps |
|---|---:|---:|---:|
| 0 | 1.3513 | 1.3974 | 1.4140 |
| 1 | 1.7464 | 1.8885 | 1.9407 |

Extra steps increased mean ADE for both samples. One sample/seed improved
slightly while the other three worsened; the report preserves these paired
differences instead of assuming refinement is an improvement. This is a
two-scenario offline experiment, not evidence of driving safety or general
model quality. Case elapsed time includes model loading, rendering and upload.

## Local checks and remaining validation

| Check | Result |
|---|---|
| Guardrails | 2,826 passed |
| Renderer dependency requirements | 21 passed |
| Final Alpamayo runtime and real local Ray scheduling tests | 36 passed |
| Workflow/catalog/smoke checks | 227 passed |
| Onboarding CLI smoke checks | 114 passed |
| Independent live artifact tests | 2 passed for each of 3 reports |
| Ruff across `npa/` | Passed |
| Generated CLI documentation drift | Passed |

The broad macOS test run was interrupted after 19,678 passes, 221 failures,
36 errors, 203 skips and one XPASS. Two new integration failures found there
were fixed and passed subsequent checks. A clean base checkout reproduced
13 representative failures, including Linux `/proc` assumptions, platform
cache paths and missing SkyPilot CLI prerequisites. The other broad-suite
failures have not all been classified. Full Linux CI remains unverified.

The GPU nodes had constrained root disks. During the initial model download,
completed immutable model blobs were copied to pod memory storage, checked
with SHA-256 and replaced by links to those copies. Incomplete blobs were left
alone. This manual cache intervention preserved real model bytes but means the
run does not establish unattended operation on a disk-constrained worker.
Provide adequate model/data cache space for a repeat run.

Both Jobs completed and were deleted before their owned namespace was removed.
The shared cluster's four nodes remained Ready. Reports, workflow logs and
licensed camera images remain in private operator evidence; no live
infrastructure identifiers or gated dataset images are included here.

See the [workflow usage and artifact-verification commands](alpamayo2-super.md#ray-experiments).
