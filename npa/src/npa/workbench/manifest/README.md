# Manifest-driven invocation (MVP)

An experiment in collapsing workbench onboarding from ~18 coordinated file
changes to **one descriptor file**.

## The idea

Today, adding a workbench tool means hand-maintaining parallel surfaces:
a Typer CLI, an SDK module, catalog `ToolEntry`s, docs, skills, and the
guardrails that keep them coherent. This MVP inverts that: the tool author
writes a single versioned descriptor (`descriptors/*.yaml`), and a **shared
invocation runtime** drives every surface from it.

```
descriptors/cuda_matmul.v1.yaml   <- the only per-tool file
        |
        v
Runtime.invoke()                  <- the one execution path ("the expediter")
   /    |     \        \
 CLI    SDK   YAML     API        <- thin waiters: syntax translation only
        |
        v
Backend (local docker | nebius mk8s pod)  <- reuses existing machinery
```

## What's here

- `schema.py` — `manifest/v0.1` descriptor: digest-pinned image, typed
  params, argv templates (rendered as a list, never a shell string),
  outputs (file or stdout), resources, success checks, payload files.
- `catalog.py` — project catalog: `name@version -> Descriptor`.
  Registration is the core of onboarding.
- `runtime.py` — `Runtime.invoke()`: validate inputs, render argv,
  dispatch to a backend, collect artifacts, evaluate success checks,
  persist a verification record **separately from the author's descriptor**.
- `backends.py` — local `docker run` translator.
- `nebius_backend.py` — ephemeral mk8s **Job** translator (ConfigMap payload,
  inline base64 markers for small artifacts, S3 upload for binary ones,
  always torn down; standard `npa` labels + TTL).
- `cli.py`, `sdk.py`, `yaml_spec.py`, `api.py` — the four waiters.
- `descriptors/` — `cuda-matmul` (deterministic CUDA matmul),
  `gpu-info` (nvidia-smi probe), and `large-artifact` (tiled matmul, S3 input
  staging, ~1 GiB binary artifact via S3) — each onboarded with zero code
  changes.
- `workloads/matmul.py`, `workloads/large_matmul.py` — the GPU workloads.

## Artifact transport

Small artifacts (`json`/`text`) travel inline as base64 log markers. Large
or opaque artifacts (`format: binary`) travel through the descriptor's
`artifact_store` (S3): the in-pod runner uploads them and reports `s3://`
URIs; the verification record carries the URIs, and success checks still run
against the inline JSON summary. A descriptor declaring binary outputs
without an `artifact_store` fails fast with a clear error instead of a
silent log blowup. `inputs` (`s3_uri` -> container path) are staged in-pod
by the runner, so large inputs never transit the 1 MiB ConfigMap limit.

**Operator note — S3 lifecycle.** Every run writes to
`manifest-mvp/<tool>/<run_id>/` under the store prefix, so repeated runs
accumulate objects indefinitely (~1 GiB per `large-artifact` run). Put a
lifecycle/expiry rule on the bucket (or prefix) before this sees regular
use; the runtime deliberately sets no retention policy itself.

## Design rules

1. **One execution path.** Surfaces translate syntax only, never semantics.
2. **Zero per-tool branches** in the runtime. Watch the count; zero is the goal.
3. **Immutable image refs.** The schema rejects descriptors without a
   `sha256:` digest pin, and `environment.pip` entries must carry exact
   `==` pins — a floating install would void the digest pin.
4. **Verification separate from declaration.** Records live apart from
   descriptors; a private run never implies shared-catalog admission.
5. **Reuse, don't rebuild.** Backends translate into existing launch/storage
   machinery.

## Evidence

Validated on a real NVIDIA RTX PRO 6000 (Nebius mk8s, ephemeral Jobs):
`cuda-matmul` 16384x16384 FP32 through all four surfaces from one descriptor —
cli/sdk/yaml at seed 0 produced bit-identical checksums (determinism),
api at seed 1 produced a different checksum (seed sensitivity). `large-artifact`
proved the transport generalizes: a ~1 GiB FP32 result matrix uploaded to S3
with the URI recorded, plus S3-staged input config and a tiled computation
driven by a non-trivial flag set. See
`npa/tests/workbench/test_manifest_mvp.py` (stub-backend unit tests).

## Status

MVP / experimental. Jobs first; services (readiness, ports, sessions) are
future work. Not wired into the `npa` CLI — intentionally standalone so the
concept can be evaluated without disturbing existing surfaces.
