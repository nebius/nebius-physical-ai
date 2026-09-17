# OpenArm runtime-fetch onboarding contract

Review date: 2026-09-15. This engineering record is not legal advice.

## Decision

| Field | Value |
| --- | --- |
| Solution and capability | Enactic OpenArm v2 MuJoCo control/render plus OpenArm Isaac Lab reach rollout and RSL-RL training |
| Packaging shape | whole-SDK runtime fetch for NVIDIA Isaac Sim/Isaac Lab; OpenArm and MuJoCo OSS are baked |
| Image redistribution | public, limited to the reviewed baked boundary below |
| Intended use | robotics simulation, evaluation, and reinforcement-learning training |
| Service use allowed | yes for OpenArm Apache-2.0 code; NVIDIA runtime use remains governed by the current NVIDIA Software License Agreement and Omniverse product terms |
| Field/output restrictions | no OpenArm-specific output restriction found; operators remain responsible for applicable NVIDIA terms and the rights in their inputs/outputs |

## Shared operator decision

| Field | Value |
| --- | --- |
| Exact operator statement | “The operator explicitly accepts any required model or software access terms that can legally be accepted on their behalf.” |
| Scope record | the bounded OpenArm onboarding goal that introduced this contract |
| Scope lifetime | expires with the named task/run; never global or permanent |
| Inherited by | the Isaac Sim 5.1.0.0 and Isaac Lab 2.3.2.post1 runtime fetch used only for this OpenArm qualification |
| Operator responsibility | operator supplies and controls credentials and is responsible for use under upstream terms |
| Reopen conditions | operator changes scope or exact authoritative terms impose a concrete incompatible requirement |

## Six boundaries

| Boundary | Exact artifact and immutable identity | Official license/terms | Disposition | Evidence |
| --- | --- | --- | --- | --- |
| Source | `enactic/openarm_mujoco@a8c979629f2591ad035d99d338ce114969e6cddc`; `enactic/openarm_isaac_lab@bad82e23716e6941c2de78ccb978f57c78b37734` | Apache-2.0 license files in each repository | baked | `components.json`, retained license files, and source-revision image labels |
| Baked runtime | CUDA 12.8.1 cuDNN development base at `sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85`; snapshot-pinned Ubuntu packages; exact Python closure | NVIDIA Deep Learning Container License; CUDA Toolkit EULA distributable-component conditions; cuDNN runtime-only supplement; Ubuntu copyright records; each Python distribution's retained license | baked | Dockerfile digest, locks, generated SBOM, `/usr/share/doc/npa-openarm/python-license-inventory.json`, and the vendor developer-payload prune record |
| Weights | no external model weights are needed for rollout; training initializes through upstream RSL-RL and produces a new checkpoint | upstream Apache-2.0 code plus applicable NVIDIA runtime terms | excluded as input; generated at runtime | workflow qualification rejects a training run without `model_*.pt` |
| Dataset/assets | MJCF/meshes and USD robot assets from the two exact OpenArm commits | Apache-2.0 license files in each source tree | baked | retained source licenses and full-filesystem inventory |
| Runtime cache | Isaac Sim 5.1.0.0 wheel manifest and Isaac Lab 2.3.2.post1 source commit `37ddf626871758333d6ed89cf64ad702aef127d0` | NVIDIA Software License Agreement and Omniverse product terms; Isaac Lab is BSD-3-Clause | runtime only | common bootstrap manifest, EULA refusal test, ready-marker checksum validation, and restricted-byte image scan |
| Outputs | NPZ traces, MuJoCo-rendered MP4, RSL-RL checkpoint/logs, and qualification JSON | operator-controlled output; no separate OpenArm restriction found | external output | `workbench.openarm.qualify` hashes and fail-closed validates the declared artifacts |

## Shared delivery identity

| Field | Value |
| --- | --- |
| Upstream endpoint/provider | NVIDIA Python package index and GitHub, with no signed URLs recorded |
| Immutable revision/digest | Isaac Sim 5.1.0.0 hashes and Isaac Lab 2.3.2.post1/source commit fixed above |
| Expected files/checksums | `docker/workbench/common/isaac-nvidia-wheels.txt` and the common bootstrap ready manifest |
| Authorization source | explicit runtime `ACCEPT_EULA=Y`; no credential is baked |
| Credential phase | runtime |
| Acceptance mechanism | documented NVIDIA environment-variable mechanism; the bootstrap maps the scoped operator decision to the exact vendor variable when `ACCEPT_EULA` is absent, and refuses an explicit negative, empty, or invalid value before network access |
| Exact access evidence | public package access; acceptance is recorded only in the run-scoped execution environment |
| Cleanup | workload teardown removes run-owned pods; explicitly mounted cache volumes remain operator-owned and are never deleted implicitly |

## Runtime-fetch delivery

| Field | Value |
| --- | --- |
| Cache tier | operator-owned durable PVC for the service; run-owned task storage for workflow jobs |
| Cache owner/access policy | non-root workload user and the run-selected namespace; the PVC remains under operator access control |
| Cache reuse permission | ordinary installed-software reuse remains subject to the NVIDIA agreement; NPA never republishes or promotes cache bytes |
| Temporary-download path | unique directory below the external cache mount, never an image layer |
| Atomic ready marker | versioned common-bootstrap manifest containing exact package/source identities and checksums |
| Offline/restart behavior | reuse only after manifest verification; otherwise fail or re-fetch after the acceptance gate |

Build-your-own delivery is not applicable because this is a runtime-fetch shape.

## Validation ledger

| Gate | Evidence | Required result |
| --- | --- | --- |
| SBOM, vulnerability, license, and secret scans | exact development digest reports from the repository image-security tooling | pass before publication |
| Real capability on target hardware | MuJoCo golden result plus four-stage workflow on an RT-core NVIDIA GPU | pass before release promotion |
| Workflow validate/plan and declared artifacts | repository validator, planner, and artifact-qualification stage | pass |
| Refusal before network | common bootstrap image-build and unit checks with acceptance variables absent | exit 78 and no fetch |
| Empty image/cache and restricted-payload absence | complete filesystem, layer, history, and OCI configuration scan | pass |
| Immutable delivery digest | full-source-SHA development tag, resolved digest, and anonymous pull proof | pass |
| Exact delivery and checksum verification | common wheel/source manifest and ready marker | pass |
| Restart/cache reuse and concurrent population | common bootstrap tests and live second use of the same cache | pass |

The NVIDIA CUDA and cuDNN redistribution permissions are conditional rather
than open-source grants. The workbench supplies material OpenArm simulation and
training functionality, confines those components to that application, retains
notices, removes separately installed SDK headers/static archives from NVIDIA
Python wheels, and requires downstream terms protective of NVIDIA. No patent
license is inferred. The exact installed files remain subject to built-image
SBOM and license review before publication.

Build-your-own validation is not applicable because this is a runtime-fetch shape.

## Claim disposition

- Accepted only after exact-image live evidence: OpenArm v2 MuJoCo rollout/render,
  `Isaac-Reach-OpenArm-v0` rollout, one real upstream RSL-RL training iteration,
  and artifact qualification.
- Deferred: lift, drawer, bimanual, imitation learning, teleoperation, and
  sim-to-real until each exact capability has independent live evidence.
- Rejected: publishing any Isaac Sim, Omniverse Kit, runtime-cache, credential,
  checkpoint, or operator-data bytes in the public image.
- Human/vendor decision required: any use whose field, service model, or
  redistribution scope is not clearly permitted by the current NVIDIA terms.
