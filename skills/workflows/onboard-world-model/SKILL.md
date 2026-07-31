---
name: onboard-world-model
description: Use when onboarding and containerizing a world model (learned action-conditioned simulator — Dreamer/Genie/Cosmos-style latent video prediction) into NPA as a multi-GPU BYOF registry candidate. Generalizes the Open Dreamer onboarding into a reusable playbook — containerize the repo, stage a real dataset, encode the train to tokenize to dynamics to dream to visualize loop as capability smokes, and validate on real GPUs.
---

# Onboard a World Model (generic playbook)

Reusable procedure for turning an arbitrary world-model research repo into an NPA
**multi-GPU BYOF registry candidate** that runs live on Nebius GPUs and drops a
viewable artifact into the agent's Rerun viewer. Distilled from the Open Dreamer
onboarding; `skills/tools/open-dreamer/SKILL.md` is the worked reference example.

Load together with:

- `skills/workflows/byof-onboard/SKILL.md` — containerize + push + run mechanics
  (base profiles, `run_byof_repo.py` / `workbench.byof.repo` toolRef, workloads).
- `skills/workflows/oss-solution-registry-onboard/SKILL.md` — registry/catalog
  admission contract (read upstream docs, encode each real capability as a
  `solution-smoke` with a named JSON artifact, collect live evidence).
- `skills/atomic/real-components/SKILL.md` — every advertised stage must invoke
  the REAL component, not an echo/manifest stub.
- `skills/tools/nebius-infra/SKILL.md`, `skills/atomic/gpu-selection/SKILL.md`,
  `skills/atomic/agent-visual-feedback/SKILL.md`.

## What Is A World Model (framing)

A world model is a **learned, action-conditioned simulator**: it watches video
(and actions), then, given context frames + a future action sequence, **predicts
("dreams") what happens next** — no game engine, no physics sim. The payoff is
training/planning agents inside cheap, safe, parallel imagination instead of the
slow/expensive/dangerous real world (the sim-to-real bottleneck). It is neural
generation (Tensor-core matmuls/attention in bf16), **not** ray tracing — latents
just make prediction cheap (predict a compact latent per frame, decode once).
Contrast: Isaac Sim/Omniverse *renders* a hand-built 3D world; a world model
*imagines* the world it learned.

Platform punchline: "point NPA at a frontier world-model repo → it containerizes,
stages data to S3, runs the full loop live on Nebius K8s GPUs, and drops a
viewable `.rrd` into the agent's Rerun viewer, self-serve."

## When To Use

- Onboard a Dreamer-family / Genie-like / Cosmos-style latent video world model.
- Package a JAX/Flax **or** PyTorch world model as a `>=2` GPU BYOF candidate.
- Author the capability contract + live GPU test for a world-model smoke.

## Canonical World-Model Loop → Capability Contract

Map the repo's real entrypoints onto this loop and encode each stage as a
solution-specific capability id (use the repo's own names; do not force a shared
taxonomy). Three **hard gates** must pass for admission; the rest headline the
run.

| Stage | Capability (rename per repo) | Real component to invoke |
| --- | --- | --- |
| Multi-GPU mesh (**hard gate**) | `<fw>_two_gpu_data_parallel_mesh` | the repo's device-mesh/parallel builder over `>=2` devices |
| Data loader | `<dataset>_video_dataloader` | the repo's real streaming dataloader (decode + action parse) with device sharding |
| Tokenizer/encoder train (**hard gate**) | `<fw>_tokenizer_train_two_gpu` | the real tokenizer/VAE training entrypoint, sharded, to legibility |
| Offline latent tokenization | `<fw>_latent_tokenization` | encode episodes → latent records + latent stats (mean/std) |
| Dynamics train | `<fw>_dynamics_train_two_gpu` | the action-conditioned latent dynamics training loop (the core world model) |
| Action-conditioned rollout (**hard gate**) | `<fw>_action_conditioned_dream_rollout` | sampler: context frames + future actions → predicted frames; report PSNR vs GT |
| Visualization | `world_model_rerun_visualization` | emit a Rerun `.rrd` with synchronized GT / dream / decoded-ceiling / recon streams |

The driver must `raise SystemExit` if the hard-gate capabilities are missing from
`capabilities_exercised`, so a single-GPU fallback cannot masquerade as a
multi-GPU run **and** a green smoke means the marquee dream actually ran (not just
that training started). Gating the action-conditioned rollout transitively gates
the whole loop, since it depends on every upstream stage. Emit one JSON per
capability + a combined results JSON (the `smoke_artifact_name`) under
`$NPA_SMOKE_OUTPUT_DIR`.

## Containerization Recipe (generic)

Follow `byof-onboard`; the world-model specifics:

- **Base image:** prefer a plain `ubuntu:22.04` (`--base-profile ubuntu`). Modern
  CUDA `jax[cuda12]` / `torch` wheels bundle their own CUDA runtime, so a CUDA
  *devel* base is usually unnecessary. Only pin a CUDA base if the repo builds
  custom CUDA/C++ extensions.
- **Interpreter + deps, world-readable:** the runtime user is **non-root**. Install
  the interpreter and venv to a world-readable dir and `chmod -R a+rX` them
  (uv's default `/root/...` dirs are unreadable by the runtime user):
  ```bash
  export UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_CACHE_DIR=/opt/uv/cache
  uv python install <pyver> && uv venv --python <pyver> /opt/byof/.venv
  (uv sync --frozen || uv sync)          # or: pip install -e . / -r requirements
  chmod -R a+rX /opt/byof/.venv /opt/uv/python
  ```
  For pip-based repos, mirror the pattern with a venv under `/opt/byof/.venv`.
- **Pin the CUDA build** matching the repo (e.g. `jax[cuda12]`, or a specific
  `torch==x.y.z+cuNNN`). Include media deps the loader needs (`decord`/`av`,
  `imageio[ffmpeg]`).
- **Rerun for the artifact:** install `rerun-sdk==<viewer-version>` at build time;
  it **must match the agent viewer** version (currently `0.31.4`) or the `.rrd`
  won't load live.
- **Fail the build early:** end `build_command` with an in-image import check,
  e.g. `python -c "import jax, flax, <repo_pkg>"` (PyTorch: `import torch; assert torch.cuda.is_available` at *run* time, not build).
- **Supply-chain hygiene:** disable optional weight downloads at runtime when
  possible (e.g. `lpips_weight=0` to avoid an HF alexnet fetch); note it as
  intentional. Never bake secrets/tokens/bucket names into the image.
- Run with the venv interpreter directly (`/opt/byof/.venv/bin/python`), **not**
  `uv run` (which tries to re-sync the root-owned venv at run time).

## Data Contract (real data, staged to the run bucket)

- Stage a **real** dataset once (operator VM has egress) and upload it to the
  project run bucket under a stable prefix (e.g. `datasets/<name>/`). The smoke
  derives the bucket from `S3_OUTPUT_PREFIX` and pulls it at run time. Do not
  hardcode buckets/paths in the spec.
- `boto3` typically lives only in the system `python3` (not the uv venv), so
  download the dataset in a **bash prelude** before the venv driver runs.
- Preprocess to the repo's expected record format (e.g. center-crop + resize to
  the tokenizer resolution, fixed-length clips, the repo's action layout).

## NPA Wiring

| Item | Path / pattern |
| --- | --- |
| Workflow spec | `npa/workflows/workbench/npa-workflows/byof-<model>.yaml` (`workload: solution-smoke`) |
| N-GPU resource profile | `npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-2gpu.yaml` (or a new N-GPU profile) |
| toolRef | `workbench.byof.repo` (argv consumes `smoke_command`, `resource_profile_yaml`, `output_root`, `run.id`, …) |
| Catalog + contract | `docs/workbench/oss-solution-catalog.md` + `npa/tests/workflows/test_byof_solution_smokes.py` (`must_exercise`) |

Spec hygiene (learned from Open Dreamer):

- `config.bucket: example-bucket` is a placeholder; real runs pass the bucket via
  `--var bucket=<real>` (submit) or the runner resolves the project run bucket.
- **`output_root` must NOT already contain `{{run.id}}`.** `run_byof_container_verify`
  appends `/<run-id>/` to `--output-root`, so a run-id in `output_root` yields a
  doubled `.../<run-id>/<run-id>/` prefix that `summary_uri` never points at. Use
  a base `output_root` and put `{{run.id}}` in `summary_uri`.
- Keep the inline `resources.gpu.accelerators` aligned with the resource profile
  (the profile is the scheduling source of truth).

## Validate (before any live run)

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-<model>.yaml --json
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py \
  npa/tests/guardrails/test_skills_index.py npa/tests/smoke/test_all_workflow_yamls.py -q
```

Ship a tiny **CPU profile** in the driver (`OD_MODE=cpu`-style env: 1 device, a
few steps, tiny model, a local sample dataset) so the whole chain can be
validated offline before burning GPU-hours.

## Live GPU Test

Model the checked-in live test on `npa/tests/e2e/test_byof_open_dreamer_live_e2e.py`:

- Gate it (`NPA_INTEGRATION_E2E=1` + a per-model `..._LIVE_GPU=1` + a prebuilt
  `..._TEST_IMAGE` to skip rebuilds); a non-GPU render check still asserts the
  spec resolves the `workbench.byof.repo` toolRef (assert on `plan.to_dict()["steps"]`).
- Drive the **spec's own** `smoke_command` + resource profile through the BYOF
  runner (the real GPU path); the `npa workbench workflow submit` path is
  intentionally plan-only for BYOF (its outer K8s pod can't build images).
- Make output **deterministic**: pass an explicit `--output-root s3://<bucket>/<prefix>`
  (test-resolvable bucket, dataset staged there) and verify at that exact prefix
  via S3 — the pod resolves a tenant run bucket at runtime the test can't
  rediscover, especially when the runner returns before upload.
- Assert all capabilities in `capabilities_exercised`, `deferred == []`, the
  device mesh `>= 2`, and a non-trivial `.rrd`.

## View / Share

Upload the `.rrd` to the run's S3 prefix and load it into a live agent:

```bash
curl -sk -u "$AGENT_USER:$AGENT_PASSWORD" -X POST https://<agent-ip>/api/sim-viz/load-artifact \
  -H 'content-type: application/json' \
  -d '{"s3_uri": "s3://<bucket>/<prefix>/<model>_world_model.rrd"}'
```

## Cross-World-Model Gotchas

- **Rollout context <= attention window.** Trim rollout context to
  `min(T, context_length)` or the KV cache overflows at prefill.
- **Action layout is asserted.** Match the repo's action encoding (e.g. N binary
  + M categorical); shift/align actions to the frame the model predicts.
- **Divisibility:** batch `B` must divide by device count (and process count);
  `packing_factor` must divide the tokenizer `n_latents`.
- **Latent stats:** carry mean/std from tokenization into dynamics; guard std
  with `np.maximum(std, 1e-6)`.
- **`sky launch` blocks** for the whole (multi-hour) run; the solution-smoke
  runner caps its own polling — never SIGKILL it mid-run (use `--no-cleanup` and
  monitor via `kubectl`/S3; the job uploads on its own).
- **Grain/multiprocessing dataloaders can intermittently deadlock** (one GPU →
  0% while the other pins at 100%, "leaked shared_memory" warnings). Lower
  `num_workers` and ensure clean shutdown if a longer run hangs.
- **Fidelity scales with budget** (tokenizer/dynamics steps ≈ 200k upstream);
  a legible-but-soft short-horizon dream is expected at smoke budgets.
- FVD/I3D scoring needs I3D weights — defer unless egress is available.

## Onboarding Procedure (checklist)

1. Read upstream docs; identify the loop entrypoints (mesh, dataloader,
   tokenizer, tokenization, dynamics, sampler, eval).
2. Write the driver (env-parameterized, GPU defaults + CPU profile) that runs the
   loop and emits per-capability JSON + the combined results artifact.
3. Author `byof-<model>.yaml` (`workload: solution-smoke`) + an N-GPU resource
   profile; wire the capability contract into `test_byof_solution_smokes.py`,
   the catalog, and this/related skills. Add the index entry.
4. CPU-validate the driver offline; `validate-spec` + run the guardrail gate.
5. Stage the real dataset to the run bucket.
6. Build + push the image (`byof-onboard`); run the live 2-GPU smoke; confirm all
   capabilities, 0 deferred, and a viewable `.rrd`.
7. Add the gated live GPU e2e test; load the `.rrd` into the agent viewer.
8. For registry admission, follow `oss-solution-registry-onboard`.
