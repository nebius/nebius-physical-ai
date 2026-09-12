---
name: isaac-arena
description: Use when packaging, running, validating, or troubleshooting Isaac Lab-Arena policy evaluation in NPA, including the public runtime-fetch image, zero/replay/RSL-RL modes, B200 state-only routing, RTX video routing, artifacts, and upstream alpha limitations.
---

# Isaac Lab-Arena

Operate the pinned upstream `isaac-sim/IsaacLab-Arena` 0.3.0 release at
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd`. This release calls itself alpha,
warns that APIs are unstable and incomplete, and says not to use it in
production. Treat the NPA integration as a hardened, reproducible wrapper for
its supported evaluation entrypoint; never describe upstream Arena itself as
production-ready.

## Capability and terms

The only supported execution path is upstream
`isaaclab_arena/evaluation/policy_runner.py`. It must complete scored episodes
and retain the upstream JSONL plus HTML report. Imports, `--help`, a simulator
launch, or an incomplete fixed-step rollout do not establish evaluation.

- Arena source: Apache-2.0, baked from the checksum-verified release commit.
- Isaac Sim/Lab: NVIDIA-proprietary wheels, absent from the image and fetched
  into the operator cache by `/isaac-sim/python.sh` after the shared
  `ACCEPT_EULA` preflight.
- Replay HDF5 and RSL-RL checkpoints: operator runtime inputs; never bake or
  publish them.
- Arena 0.3.0 is tested against Isaac Lab 3.0 beta 2. NPA uses the compatible
  patched Lab `3.0.0b2.post1` / Isaac Sim `6.0.1.0` baseline and must requalify
  every changed image digest.

Before build, download, provisioning, or submission, also load
`skills/atomic/third-party-eula-preflight/SKILL.md`. Never invoke the Isaac
launcher in a Dockerfile `RUN`; that would bake the restricted runtime.

## Evaluate

Use the CUDA state-only workflow on B200; it must not record cameras or a
viewport because B200 has no RT cores. Use RTX PRO 6000 for the independent
graphics qualification and require a non-empty MP4.

```bash
npa workbench health preflight --checks nebius,storage
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-b200.yaml
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-rtxpro.yaml
```

`npa workbench isaac-arena evaluate` supports only upstream-shipped
`zero_action`, `replay`, and `rsl_rl` policies. Replay requires one HDF5 file.
RSL-RL requires a `model*.pt` checkpoint beside `params/agent.yaml`, matching
upstream's real runner contract. Use `--input-path` with a local path or S3 URI;
NPA materializes the input before starting the simulator.

Successful evaluation requires:

- one or more episode records with boolean `success` and positive
  `episode_length`;
- `upstream/<timestamp>/episode_results_rank*.jsonl`;
- `upstream/<timestamp>/report/index.html`;
- a required MP4 when `--record-video` is selected; and
- `result.json` with the source revision, request, measured GPU identity,
  success rate, byte sizes, and SHA-256 hashes.

The zero-action qualification is a real baseline evaluation and may correctly
report zero success. Do not turn that expected policy result into a synthetic
pass; the capability gate is factual execution and artifact integrity.

## Build and release

Build only from a clean exact commit. Official public development bytes use
`dev-<full-git-sha>`; scan the built filesystem and history, not merely the
Dockerfile.

```bash
bash npa/docker/workbench/isaac-arena/build.sh
npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py \
  --docker-image npa-isaac-arena:<tag>
```

Require a clean Omniverse payload scan, non-root user, exact Arena source
labels/license, empty runtime cache, anonymous digest resolution, and real
completed-episode runs on both B200 (`sm_100`) and RTX PRO 6000 (`sm_120`)
before promotion. Preserve the B200 no-video and RTX required-video distinction
in evidence.

Cancel exact workflow runs before removing any dedicated resources. Do not
destroy shared clusters, buckets, or reserved capacity after a validation run.

## Diagnose

- Exit 78 before download: explicit EULA opt-out; do not bypass it.
- No episode JSONL: use `--num-episodes`, not an incomplete step-only smoke.
- Missing `params/agent.yaml`: stage the complete RSL-RL checkpoint directory.
- Missing MP4 on RTX: inspect camera enablement, Vulkan/RT drivers, and the
  upstream viewport recorder; do not downgrade the artifact requirement.
- B200 render failure: the workload is misrouted. Keep B200 state-only and move
  rendering to RTX PRO 6000.
- Source/runtime incompatibility: retain the exact pins and report the alpha
  upstream boundary; do not patch around failures with fake output.

## Verify changes

```bash
npa/.venv/bin/python /home/ubuntu/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/tools/isaac-arena
npa/.venv/bin/python -m pytest npa/tests/workbench/test_isaac_arena.py npa/tests/docker/test_packaging_contract.py npa/tests/guardrails/test_skills_index.py -q
```
