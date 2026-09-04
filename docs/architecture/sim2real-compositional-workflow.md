# Sim2Real compositional workflow and resume contract

The operator-facing Sim2Real pipeline is
`npa/workflows/workbench/npa-workflows/sim2real.yaml`. It is an
`npa.workflow/v0.0.1` graph executed by `npa workbench workflow ... --runtime`.
It is not detected or submitted through a Sim2Real-specific controller.

## Solution boundaries

Each real solution boundary is a workflow state with its own immutable image,
resource request, inputs, and outputs. CPU contract states surround Cosmos
Transfer, parallel environment-generation shards, Isaac policy rollouts,
a CPU-only hosted Cosmos3 evaluator, BYO Isaac RSL-RL PPO, Isaac gold evaluation, and
Rerun/MCAP finalization. Stage 12 is intentionally an external `SEAM`; it is
recorded as such and is never reported as `WORKS`.

The outer and inner loops are ordinary workflow loops. Named `{{loop.*}}`
tokens scope iteration artifacts, while the standard decision artifact controls
the outer gate. The reduced merge proof sets both bounds to one and disables
early exit; higher iteration counts are a post-merge efficacy choice.

## Artifact and restart contract

Every stage consumes explicit S3 URIs and publishes an immutable result plus a
`ComponentRecord`. The canonical pointer `components/stage_XX.json` is backed by
a content-addressed copy under `components/history/stage_XX/<sha256>.json`.
Records attest the source SHA, workflow-owned image digest, input/output URIs,
and GPU evidence where applicable. Train, validation, and gold paths remain
disjoint; checkpoint selection records the exact checkpoint URI, SHA-256, and
training iteration. Gold reports retain their exact render prefix.

The standard workflow runtime persists its ledger at
`<run-prefix>/npa-workflow/runtime.json`. On `--resume` it adopts an in-flight
managed SkyPilot job, reuses completed waves only after their declared S3
outputs pass validation, and resubmits only incomplete work. Consequently a
restart at the Stage 8/9 barrier or during finalization reconciles from the
same workflow/S3 state instead of reconstructing private controller memory.

## Execution ownership

SkyPilot/Kubernetes owns each state Job. Isaac rollout, PPO, and evaluation run
their existing fail-closed payload in the workflow task; they do not create
hidden sibling Jobs. Kubernetes scheduling, retries, credentials, and image pull
behavior therefore remain visible to the standard runtime. SkyPilot submits each
bounded parallel wave directly to Kubernetes; `gpu_concurrency` keeps each wave
within the cluster's schedulable GPU capacity. The workflow task image must equal
the payload's immutable digest.

Isaac terms remain runtime state rather than image metadata. NPA applies its
single `ACCEPT_EULA=Y` non-interactive default only to resolved Isaac routes;
recognized negative values and `--no-accept-eula` opt out before scheduling,
and invalid values fail locally. No image bakes acceptance, privacy consent, or
telemetry consent.

CPU contract/bookkeeping states use the dedicated digest-pinned
`npa-sim2real-control` image. Its Python-slim base, exact source, and
resolver-closed S3 dependency set are intentionally separate from Genesis,
Isaac, CUDA, and trainer images. The image also bakes the standard non-root
SkyPilot Kubernetes bootstrap closure from a fixed Debian snapshot; no task
depends on installing sshd or rsync from a mutable mirror after scheduling.
The exact source is also registered through a site-packages `.pth` file, so
SkyPilot login/setup shells can import the stage adapter even when they do not
preserve the image's `PYTHONPATH` environment variable.
Importing a focused stage adapter also leaves the archived controller facade
unloaded. This keeps a cold CPU pull small and prevents control-plane success
from depending on GPU-runtime packaging.

`config.require_baked_npa` makes the renderer reject mutable or missing images
and replaces source-tarball/package bootstrap with a baked-source attestation.
The task verifies `NPA_IMAGE_SOURCE_SHA` against the exact workflow SHA before
it runs; no dependency installation happens after admission.

`engine.py` is a lazy, finite compatibility facade for pre-standard-runtime
callers and archived artifacts. Its implementation is split into bounded legacy
orchestration, component/training, held-out-contract, Isaac, and artifact
modules. The canonical workflow imports none of them and never calls
`run_preamble`, `run_inner_loop`, `run_single_outer_iteration`, or
`run_finalize`. The target removal is NPA 0.5.0, no earlier than 2027-02-01;
the direct-controller materializer and submit implementation are already gone.
