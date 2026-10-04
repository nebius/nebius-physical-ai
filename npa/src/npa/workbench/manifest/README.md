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
- `nebius_backend.py` — ephemeral mk8s pod translator (ConfigMap payload,
  base64 artifact markers in logs, always torn down).
- `cli.py`, `sdk.py`, `yaml_spec.py`, `api.py` — the four waiters.
- `descriptors/` — `cuda-matmul` (deterministic CUDA matmul) and
  `gpu-info` (nvidia-smi probe), the latter onboarded with zero code changes.
- `workloads/matmul.py` — the GPU workload (seed-controlled, checksum artifact).

## Design rules

1. **One execution path.** Surfaces translate syntax only, never semantics.
2. **Zero per-tool branches** in the runtime. Watch the count; zero is the goal.
3. **Immutable image refs.** The schema rejects descriptors without a
   `sha256:` digest pin.
4. **Verification separate from declaration.** Records live apart from
   descriptors; a private run never implies shared-catalog admission.
5. **Reuse, don't rebuild.** Backends translate into existing launch/storage
   machinery.

## Evidence

Validated on a real NVIDIA RTX PRO 6000 (Nebius mk8s, ephemeral pods):
`cuda-matmul` 16384x16384 FP32 through all four surfaces from one descriptor —
cli/sdk/yaml at seed 0 produced bit-identical checksums (determinism),
api at seed 1 produced a different checksum (seed sensitivity).
See `npa/tests/workbench/test_manifest_mvp.py` (stub-backend unit tests).

## Status

MVP / experimental. Jobs first; services (readiness, ports, sessions) are
future work. Not wired into the `npa` CLI — intentionally standalone so the
concept can be evaluated without disturbing existing surfaces.
