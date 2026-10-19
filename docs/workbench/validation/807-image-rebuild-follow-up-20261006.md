# Image rebuild follow-up to PR #807, 2026-10-06

PR #807 withdrew image bytes that exposed SSH host private keys, operator build
metadata, or the wrong runtime user. Keeping those releases quarantined is
intentional. Rebuilding a child from an affected parent retains the affected
layers, even when a later layer removes the files. A source fix or a successful
import does not qualify a public workload.

This follow-up repairs additional build recipes and current shipped image
defaults. It does not promote an image, change `publication_pending`, or replace
archived validation evidence. The
[original planning inventory](public-default-quarantine-impact-20261005.md)
remains a dated record of the default resolver audit.

## Repaired parent builds

[PR #877](https://github.com/nebius/nebius-physical-ai/pull/877) repairs the
canonical Sim2Real image builds. Its exact source
`127f93d3cf6d10cfb540766578834c50400f8a09` passed the
[trusted five-image publication job](https://github.com/nebius/nebius-physical-ai/actions/runs/37409328900)
for the controller, Transfer, EnvGen, Isaac Lab and viewer. Independent anonymous
manifest and config requests also verified each immutable digest, source
revision and current SkyPilot bootstrap label.
Both digest-bound SLSA provenance and SPDX SBOM attestations were independently
verified against the official hosted publication workflow and exact source for
all five images. These checks do not replace native workload qualification.

Two of those built parents are now selected by the derivative recipes:

| Parent | Immutable development digest | Children requiring their own rebuild and qualification |
| --- | --- | --- |
| Isaac Lab | `sha256:ef7f4234839852ed7b48ea4b88fd61f1c45a37ca00294c10a0a23fb335807ef4` | Isaac Arena |
| EnvGen | `sha256:55f1541c1a86e9d865963753cfa38fa1dbf88928c8d28d9ee30e2851a5604b88` | Reference Policy, LeRobot VLM RL, Loop Eval, alternate Explore Policy |

These are build inputs, not public workflow default exemptions. PR #877 must
land before this source-dependent change is merged, so the parent revision is
reproducible from `main`. The child image's own published graph, inherited layers,
config, redistribution gates and native capability must still pass. Canonical
Sim2Real can independently qualify the five-image set through its existing
explicit operator-image contract; the legacy automatic defaults remain
quarantined.

## Additional source repairs

| Family | Source change | Required runtime proof |
| --- | --- | --- |
| Genesis | Correct the exact reviewed scikit-image data-recipe text in the dependency install layer, including bytecode and package RECORD. Stock Python 3.10 uses 0.25.2; the additive Python 3.11 CUDA 13 recipe uses 0.26.0. Build pinned imageio-ffmpeg from its pure-Python source distribution and verify that it uses Ubuntu's FFmpeg with no bundled executable. Unknown source hashes fail before mutation; pip wheel caching is disabled. | A real teacher optimization run, loadable checkpoint, exported actor equivalence and rendered physics evidence on its supported GPU family. |
| LeRobot 0.5.1 and 0.6.0 | Pin the reviewed scikit-image 0.26.0 dependency and correct its inert historical recipe in each version's original install layer, with bytecode/RECORD updated and wheel caches disabled. The fresh default 0.5.1 build exposed the same MEDIUM JWT finding as the optional 0.6.0 closure. | Version-specific real policy training and loadable checkpoints; neither optional Diffusion evidence nor an image scan establishes default ACT or VM deployment capability. |
| Isaac Arena | Replace the withdrawn Isaac parent; keep the exact Arena source, runtime-fetch licensing and replay evidence patches. | Native reset/action replay, completed episodes, simulator metrics and decoded rendering. |
| Reference Policy and Explore Policy | Replace the stale EnvGen parent and copy the complete child SDK source so the declared revision identifies the code that runs; preserve their action-policy variants. | The actual policy action and schema handoff; delegating an environment smoke is insufficient. |
| LeRobot VLM RL | Replace the stale EnvGen parent. | Real VLM reward signals, nonzero optimizer and weight changes, and a loadable trained checkpoint. |
| Loop Eval | Replace the stale EnvGen parent and install the fixed Jammy kernel headers from the October 1 snapshot. | Native Genesis rollouts and joined policy/VLM reports; this legacy tool is separate from canonical Sim2Real. |
| SONIC / Isaac 2 | Update GitPython, Werkzeug and the fixed Jammy header snapshot. Refresh the eight exact Python 3.11.17 Debian artifacts against the signed Deadsnakes index after the old 3.11.15 URLs returned 404. Preserve the Python 3.11 ABI, Isaac Lab 2.3.2.post1 / Isaac Sim 5.1 and runtime-fetched vendor payloads. | Each advertised training, checkpoint, evaluation and export capability, with its exact gated runtime dependencies. |

The fresh stock Genesis build then failed its unchanged license gate on
TetGen 0.8.2's AGPL distribution. The
[public Genesis recipe contract](../../../npa/docker/workbench/genesis/README.md)
now explicitly retains the rigid Panda/Box workload and ACT dependencies while
excluding tetrahedralization, Diffusion/SmolVLA dependencies, W&B and the
incompatible TorchCodec decoder in the original installation layer. It binds
the original Genesis/LeRobot metadata to exact reviewed hashes, refreshes RECORD
and runs `pip check`. The native CPU build gate must decode real converted camera
videos through LeRobot's default PyAV backend and round-trip ACT plus its saved
normalization processors. The first corrected gate decoded both real camera
videos, then exposed eager GR00T imports while importing ACT. The public recipe
now binds ACT-only registration/factory changes to exact upstream hashes while
retaining native ACT logic and saved-processor loading. Other factory families
fail closed. The exact-source Linux build has now passed its native default PyAV
camera dataset and ACT checkpoint/saved-processor round trip, as well as all
ordinary publication gates. The signed SBOM does not contain TetGen. GPU
teacher optimization, export and rendered physics qualification remain pending.
No license exceptions or incompatible Diffusers metadata widening are introduced.

## Trusted child publication evidence

These exact image jobs passed the ordinary publication gates. Independent
anonymous manifest/config checks bound their bytes and source revision, and
both provenance and SPDX attestations were cryptographically verified against
the official hosted workflow. The aggregate first build failed because its
Genesis and SONIC jobs failed; only its four successful image jobs are credited.

| Child | Source revision | Immutable development digest | Successful publication run |
| --- | --- | --- | --- |
| Loop Eval | `3478a0748bbd6f021643cbc03ed37f2dcc20f8c2` | `sha256:84ffb8b34c93281b8efa0dd7c9dfdb8ac88d52cbae88519719faf4796e660582` | [37417561123](https://github.com/nebius/nebius-physical-ai/actions/runs/37417561123) |
| Isaac Arena | `3478a0748bbd6f021643cbc03ed37f2dcc20f8c2` | `sha256:70fcd46d1655431f9f2a2655f32f1b2e4156bb259291ed0b20e70dacd153c5d5` | [37417561123](https://github.com/nebius/nebius-physical-ai/actions/runs/37417561123) |
| Reference Policy | `3478a0748bbd6f021643cbc03ed37f2dcc20f8c2` | `sha256:26f532db2365c4d4bafab8f3ac994e04a754689c9baee1e3b5d2166f828cf40d` | [37417561123](https://github.com/nebius/nebius-physical-ai/actions/runs/37417561123) |
| LeRobot VLM RL | `3478a0748bbd6f021643cbc03ed37f2dcc20f8c2` | `sha256:5dddf9bd1cae85d19c0e570a3d52ddcf7efae8ebab323092e2cfcb461966008d` | [37417561123](https://github.com/nebius/nebius-physical-ai/actions/runs/37417561123) |
| SONIC | `540e07bdbcbdba1fb994946aad9a151df6330ad9` | `sha256:efcd93d00940b6ee718aeb5907e02dcc6a576b7d3f4151a6d40607d2c55545ca` | [37418916800](https://github.com/nebius/nebius-physical-ai/actions/runs/37418916800) |
| Default LeRobot 0.5.1 | `4f77cb3ae430aad96c34705d7219069d7b7c4494` | `sha256:d8cd706592a37b0cff91297ef0aafd543b274b0b795699d015d0d50c04584b17` | [37484670973](https://github.com/nebius/nebius-physical-ai/actions/runs/37484670973) |
| Genesis, public rigid/ACT scope | `002bf337dd7e8ac260ea4be09f784a61cd4e0863` | `sha256:4b2f5eebc399d9d463f197542fe146f5c4dfa3cb6e9c32df3802b3aea9d25459` | [37496185473](https://github.com/nebius/nebius-physical-ai/actions/runs/37496185473) |

These tool defaults remain quarantined. Genesis passed the narrow native CPU
gate described above; its real GPU teacher/export/rendering workload and the
other child workload qualifications remain pending. SONIC is not currently a SkyPilot-bootstrap-attested tool;
its successful publication does not establish that independent runtime contract.
The default LeRobot 0.5.1 source-correction rebuild and both digest-bound
attestation checks succeeded, separate from the optional 0.6.0 candidate.
An independent execution of the exact default LeRobot candidate then failed
`pip check` and native default camera decoding: forced Torch 2.12.1 and
torchvision 0.27.1 exceeded upstream bounds, TorchCodec 0.10 could not load
against that Torch ABI, and required W&B was absent. Its ACT optimizer changed
62 parameter tensors and checkpoint/processors reloaded correctly, but that
narrow result does not accept the overall image. These older incompatibilities
came from PR #173, not PR #807. A separate
[maintained integration repair](https://github.com/nebius/nebius-physical-ai/pull/888)
binds upstream source/license, declares its own version and requires real native
decoding, ACT, Diffusion and server gates; its acceptance is independent of this
Genesis/derivative change.

The stock Genesis CUDA 12.4 / Torch 2.6 recipe does not establish RTX PRO 6000
support. Its CUDA 13 recipe remains an additive operator-base build, and
correcting a child layer cannot sanitize already-published parent layers. These
limitations must be resolved before making a corresponding release claim.

The distinct Cosmos3 Ray Serve candidate also remains blocked: its trusted
build's payload gate reported credential-shaped bytes in its dependency closure.
No filename exclusion or blanket scanner suppression is introduced. Native
generation and native Ray Serve batching are separate qualification scopes.

## Current specifications and explicit operator inputs

The original resolver inventory did not inspect literal image digests as
quarantined tool defaults. Current example specs must not automatically select
withdrawn Isaac or SONIC bytes through that seam. Resource-only consumers use
governed tool image resolution. Consumers that pass an exact runtime image into
provenance or child launch requests require explicit immutable operator inputs
before planning or execution. Archived receipts keep their original digests and
do not become proof of the rebuilt images.

| Current shipped specification | Image input after the repair |
| --- | --- |
| `field-failure-reference-demo.yaml` | Required exact `navigation_image` and `reconstruction_image`; withdrawn Isaac and SONIC defaults removed. |
| `rgbd-scan-to-policy-demo.yaml` | Required exact `assembly_image`, `reconstruction_image` and `isaac_image`; reconstruction can use the explicitly supplied assembly image. |
| `rgbd-scan-to-isaac.yaml` | Required exact `isaac_image`, also consumed by runtime-image provenance. |
| `scan-to-isaac-navigation.yaml` | Required exact `isaac_image`, also consumed by runtime-image provenance. |
| `multicamera-rgbd-warehouse.yaml` | Required exact `isaac_image`, also consumed by runtime-image provenance. |
| `franka-rl-transfer.yaml` | GPU `isaac_image` uses governed `tool://isaac-lab` resolution. Its distinct optional LeRobot 0.6 CPU image is unchanged. |

All six specs live under `workflows/testing/`. The exact original specs and
readiness records are preserved in
[`docs/workbench/evidence/workflow-defaults-807`](../evidence/workflow-defaults-807/).
The current readiness records require fresh workload qualification. The demo
aliases accept the same repeated `--var key=value` inputs as standard workflow
submission, and missing exact inputs fail before submission.

The optional LeRobot 0.6 training family is a separate version and dependency
closure from the quarantined default 0.5 family. Fresh image-byte and real
training qualification are required before selecting a new optional candidate;
neither a version switch nor historical Diffusion evidence proves ACT training
or VM deployment on the current GPU.
