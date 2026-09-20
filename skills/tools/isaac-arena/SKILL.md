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

The capability payload's `upstream_workflows` also lists agentic environment
generation, experiment orchestration, sensitivity analysis, teleoperation,
Mimic data generation, imitation learning, and reinforcement learning. These
are `unsupported`, `input_required`, and `upstream_alpha` in the NPA Arena
integration. Their pinned upstream entrypoints or documentation and required
inputs are listed for discovery. Do not route them to the evaluation runner or
claim that other NPA tools qualify these Arena workflows. Prompt resolution
requires model access; upstream schema/catalog inspection does not. The
upstream OSMO experiment backend is not part of NPA's SkyPilot integration.

- Arena source: Apache-2.0, baked from the checksum-verified release commit.
- Lightwheel SDK 1.0.3: upstream-declared Apache-2.0 client, baked from the
  exact wheel and SHA-256 in Arena's `uv.lock`. Its package description and some
  module headers carry the Apache-2.0 grant and license URL; the wheel has no
  structured license field or standalone license text. The image includes the
  complete Apache-2.0 text at `/opt/isaac-arena/LICENSE.md` and the SDK notices.
- Lightwheel registry assets: provider-controlled runtime inputs, never baked
  or redistributed by NPA. Applicable environments name their exact selector
  or layout requirement in the capability payload. NPA supplies no Lightwheel
  credential or asset-license grant; confirm upstream access before GPU spend.
- Isaac Lab 3.0.0b2.post1: its wheel declares BSD-3-Clause. It is absent from
  the image and fetched into the operator cache with its runtime dependencies.
- Isaac Sim and its proprietary runtime dependencies: absent from the image
  and fetched by `/isaac-sim/python.sh` after the shared `ACCEPT_EULA` preflight.
  The Lab wheel's BSD license does not replace those separate runtime terms.
- NVIDIA viewport graphics userspace: prefer and validate the target's native
  headless EGL/Vulkan, `libnvoptix.so.1`, and readable regular nonempty
  `/usr/share/nvidia/nvoptix.bin`. If a target omits these dependencies, fetch only the Ubuntu
  signed `libnvidia-gl-<branch>-server` package whose version exactly matches
  the loaded driver, validate its identity, SHA-256, and packaged ICD metadata,
  extract it into run-private scratch, and derive a canonical private EGL ICD.
  The image contains only an empty weights directory. Copy missing weights only
  into its verified user-owned private root overlay, refusing symlinks and
  submounts; publish atomically without overwriting and verify the copied hash.
  Retain copied weights only for the worker container lifetime. Never install
  the package on the node, bake its bytes, publish them as artifacts, or
  redistribute them. Library/file/settings readiness is not denoising success.
- Replay HDF5 and RSL-RL checkpoints: operator runtime inputs; never bake or
  publish them. The result records source hashes, not storage locations.
- Inherited open-source dependencies retain public examples and test fixtures,
  including Newton's sample policy and USD assets and ONNX conformance models,
  with their installed licenses. Describe this boundary precisely: those
  fixtures are not Arena policy inputs or qualification evidence, and the
  image does contain model-shaped dependency files.
- Arena 0.3.0 is tested against Isaac Lab 3.0 beta 2. NPA uses the compatible
  patched Lab `3.0.0b2.post1` / Isaac Sim `6.0.1.0` baseline and must requalify
  every changed image digest.

Before build, download, provisioning, or submission, also load
`skills/atomic/third-party-eula-preflight/SKILL.md`. Never invoke the Isaac
launcher in a Dockerfile `RUN`; that would bake the restricted runtime.

## Evaluate

Use the four-seed CUDA state-only workflow on B200; it must not record cameras
or a viewport because B200 has no RT cores. Its four sequential real evaluation
states are the comprehensive daily workflow coverage for this image. Use RTX
PRO 6000 for the independent graphics qualification and require a non-empty
MP4 with successful task and coherent-motion evidence. NPA's context-bound source
patch must keep Kit camera support
enabled while leaving unused embodiment-mounted observation cameras disabled;
the upstream camera-video recorder remains unsupported. The RTX replay workflow
selects CPU physics and replay tensors following upstream's GR1 tutorial, while
the viewport still uses the reserved RTX GPU. Record both devices separately.
The fixture does not record its original physics device; this selection needs
fresh task/video validation and is not proof of successful reproduction.
The RTX renderer must explicitly enable legacy RTX and disable RT2 plus
interactive path tracing through AppLauncher Kit startup arguments, then
reassert legacy RTX Real-Time plus supported TAA,
the DL denoiser, native per-pixel sampling of 32 direct-light, 32 indirect-diffuse,
and 16 reflection samples, disabled frame generation, and at least eight
consecutive ready physics-frozen settling renders after the readiness baseline at the live capture
boundary and verify
exact Carb-setting readback before rendering and again,
without another write, after the final accepted render. Treat a mismatch as a failed run; do not accept a
default renderer or infer stability from the requested configuration alone.
The explicit sample counts improve native rendering quality after a completed
successful replay failed the existing progress-overlap gate. Keep that gate,
physics binding, action order, resolution, and settling requirement unchanged;
require fresh immutable-image qualification before claiming the correction works.
Task-qualified evidence also requires the registered environment/policy pair,
a sanitized policy-to-`env.step` action journal covering the full scored episode,
finite nonzero varied actions, and zero synthetic padding. Replay must match the
exact prepared private action-sequence prefix through the native terminal, while
the complete prepared source sequence independently passes held-tail checks.
Only `gr1_open_microwave` currently has semantic task-progress wiring. Its
adapter must declare the visual interval strategy, bounded context, progress
visual signal, normalized task-object region, and spatial association radius.
Keep its measured task-progress interval separately; include up to 30
leading action steps for approach/contact context, require an accepted coherent
track to overlap native progress, and require connected monotonic structural
change inside the exact progress interval, microwave workspace, and declared
distance of that track. Require duplicate source-frame indices across every
evidence view to retain identical decoded hashes and timestamps. Never let
motion outside progress, disjoint unrelated motion, or post-transform stochastic
grain satisfy the shared gate. Do not admit a
post-terminal reset or invented frames. Capture, actions, and the terminal PNG
must span that same complete native episode.
Treat an adapter's normalized task region as fixed-camera semantics. A camera
change requires a new registration and live qualification; never silently reuse
the prior region.
Enforce the adapter's declared maximum trailing-held-action fraction on both
prepared and executed sequences, using a shared maximum inter-step delta of `1e-6`;
exact repetition or tiny numerical jitter cannot hide a dominant held final action.

```bash
npa workbench health preflight --checks nebius,s3
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-b200.yaml
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-rtxpro.yaml
```

The current release is `0.3.0-isaaclab3-20260917-r4`, exact manifest
`sha256:9c6a417672d6f87499680ba337c90488c2a33d41ac9f7b5452eb5d97d00e097e`,
promoted without rebuilding from development source SHA
`ae5adea6ab895660996f513f14160c89d06f47e5` after fresh exact-digest B200 state
and RTX task/visual qualification. B200 seeds 42–45 each completed 1,050
native steps. RTX executed 43 replay actions through native terminal with
zero padding, success 1.0, door openness 0.200→0.815 and four spatially bound
progress-overlap pairs. Both raw and evidence MP4s were fully decoded;
the separate factual RRD preserves 44 source captures and native metrics.
Both new controllers are SUCCEEDED with zero active workers; shared capacity
and controllers are retained. See the digest-scoped readiness records and
[fresh proof](../../../docs/workbench/isaac-arena.md#fresh-recovery-qualification--2026-09-17).
The historical r2 RTX proof remains
rejected: 80 source actions were followed by 170 held-action steps, the recorded
initial state was not applied, success rate was zero, and render grain passed
the former pixel-change gate. Do not treat that historical digest or its
zero-action predecessor as meaningful visual evidence. On a target whose accelerator
spelling has already passed `npa workbench workflow gpus`, set
`NPA_WORKFLOW_GPU_ACCELERATOR=B200:1` for the state-only spec or
`NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` for the
video spec. This is an exact placement pin, not cross-platform fallback.

`npa workbench isaac-arena evaluate` supports only upstream-shipped
`zero_action`, `replay`, and `rsl_rl` policies. Replay requires one HDF5 file;
NPA selects its first sorted episode, requiring finite multi-step actions and a
finite `initial_state` group. Source success and recorded-state histories are
optional diagnostics, never runtime outcomes. Zero-action recordings in compatible action spaces and missing
or static state histories remain valid ordinary evaluation inputs. Replay visual
qualification additionally requires measured nonzero source actions.
RSL-RL requires a `model*.pt` checkpoint beside `params/agent.yaml`, matching
upstream's real runner contract. Use `--input-path` with a local path or S3 URI;
NPA materializes the input before starting the simulator. The upstream replay
loader eagerly moves every episode field to the execution device, even though the policy uses
only `actions` and `initial_state`; NPA therefore creates a private minimal
execution HDF5 and never publishes either input. Its result binds the source and
prepared execution-input hashes. These identify input bytes and do not prove
that actions ran. Replay supports one episode in one environment. Apply its
recorded initial state with Isaac Lab `reset_to(is_relative=True)` and execute
the exact source prefix through the requested native episode terminal. Record
any unexecuted suffix explicitly and keep it outside the scored/captured episode.
Match the scene to the recording: the microwave tutorial has no optional extra
object, so its shipped workflow uses `config.object: ""`. Upstream's `--object`
adds another physical asset and needs its recorded initial state. Preserve an
explicit matching selector; do not fill missing state from simulator defaults.
Do not invent an early stop, synthetically pad/repeat actions, or append a held
action to manufacture a completed episode. A naturally stable source tail remains
subject to the registered adapter limit. A source recording that cannot complete the task is an
input/qualification limitation, not permission to fabricate success.

Use the declared HDF5 quaternion format: missing/integer 0 is legacy WXYZ;
integer 1 is XYZW. Reject unknown or malformed versions. The pinned native
loader converts only root poses; it leaves embedded Pink action quaternions
untouched. For the resolved `gr1_pink` embodiment, NPA validates the 36-column
layout, nonzero quaternion norms, and single-environment initial robot root
pose. It reorders legacy hand-target slices `3:7` and `10:14` and all initial
root quaternions into XYZW before marking the private execution file version 1.
Preserve physical orientations, non-quaternion values, action order/count, and
original source bytes. Retain source/execution hashes and representation-change
metadata. Never guess the convention from values or select Pink conversion
from action width: `gr1_joint` also has 36 columns. Other action contracts keep
native root-pose handling and require compatible embedded action values.

For Pink recordings with native `obs/datagen_info/target_eef_pose/left` or
`right` matrices, validate each provided target's position and orientation
against its same-step action before creating execution input. Reject malformed
matrices and contradictions in quaternion interpretation, frame, or sample
alignment. Mixed-format sources need explicit source-specific normalization
and retained provenance; never repair them by changing only a format label or
guessing from replay success. The recorded-target check describes source
consistency, not current-run motion or task success.

Successful evaluation requires:

- one or more episode records with boolean `success` and positive
  `episode_length`;
- `upstream/<timestamp>/episode_results_rank*.jsonl`;
- `upstream/<timestamp>/index.html` plus linked pages under `report/`;
- the current-run simulator metric-recorder HDF5, consistent with the JSONL;
- a required MP4, capture sidecar, and actual initial/terminal PNGs when
  `--record-video` is selected; and
- `result.json` with the source revision, request, measured GPU identity,
  success rate, byte sizes, source/prepared execution-input binding, and SHA-256 hashes.

Ordinary evaluation may correctly report zero success for any adapter. Preserve
failed task results and distinguish completed evaluation from successful task
execution. The B200 zero-action qualification is a baseline: its capability gate
is factual execution and artifact integrity, with no visual claim.

Task-qualified video uses a shared fail-closed contract that binds one action
horizon across executed actions, the native scored episode, native task progress,
simulator capture, and noise-resistant video. Replay actions must be nonzero and
must not be synthetically padded or dominated by a numerically held tail.
Environment-specific progress semantics belong in
the explicit task-progress adapter registry; never embed one task's thresholds
as generic runtime assumptions. The sole current registration supports
`gr1_open_microwave` with `replay` or `rsl_rl`. Require matching current-run
JSONL/HDF5 success, numeric
`success_rate > 0`, final door openness above the upstream threshold `0.8`, and
maximum openness at least `0.5` above the initial state. Retain initial/final
openness and the full metric trace. Restrict visual acceptance to first measured
door progress through the first threshold crossing, excluding idle/reset frames.
Record the adapter identity and `npa.isaac-arena.visual-acceptance.v1` result.
Other registered scored environments retain ordinary evaluation support; their
nonzero-policy video qualification remains unsupported until explicitly registered.

Capture actual initial and terminal PNGs before automatic reset. Verify
`simulator-video-evidence.json` against their file and decoded-RGB hashes,
contiguous HDF5 action steps, and the matching decoded terminal MP4 frame within
encoding tolerances. The shared acceptance record must bind the sidecar, PNG,
raw-MP4, and evidence-MP4 hashes to the same native episode and exact action
horizon. Require the version-2 sidecar's initial-plus-every-action
physics checks: native PhysX step-event counts and elapsed event time, the Lab
physics counter, and uncached robot/object state must remain unchanged during
rendering. The native subscription must also observe progress between real
actions; elapsed event time starts at capture setup, not an absolute clock.
Verify asset/texture readiness and the exact legacy RTX Real-Time
`RaytracedLighting`, RT2-disabled, TAA/DL-denoiser settings readback,
disabled frame generation, and at least eight consecutive ready physics-frozen
settling renders after the readiness baseline per captured frame. Do not use
stochastic path-tracing accumulation for acceptance footage; only the declared
TAA history may span those render-only settling calls. The separate temporal-median
and coherent-motion verifier remains mandatory. Rendering must
not add physics steps or video frames. An unfinished recorder buffer belongs only in the separate
unscored diagnostic and cannot create upstream success or completed episodes.
When emitted, `simulator-phases-rank*.jsonl` contains fixed phase/event labels,
monotonic timestamps, rank, action/render counters, and readiness booleans for
policy, Pink IK, environment, and capture operations. It never contains their
arguments, input arrays, or exception text. One unfinished phase alone is not a
task result. The independent parent observer fails closed after eight completed
samples when a phase has no advancement for over 4,096 times its longest measured
duration. It retains scalar liveness diagnostics and terminates only its owned
process group. This is a phase-progress policy, not an episode budget. Cold phases
without a baseline remain unclassified. Keep replay, physics and graphics fidelity
unchanged; do not describe liveness failure as a proven upstream native cause.
Journal writes remain best effort inside the simulator; malformed or truncated
progress evidence fails in the parent observer.
An `unavailable` method-phase event means the native binding could not be
overridden on its instance. Keep that binding untouched and use the enclosing
simulator phase; availability is neither execution evidence nor task progress.
Preserve the raw MP4 and label the denoised half-speed derivative with its
source hash and exact spatiotemporal/low-pass transform. Keep the frame count
unchanged; never use duplicated frames as padding. Validate coherent motion over
the adapter-declared visual interval; a noisy static scene must fail both before
and after the evidence transform.
Zero-action output is always a baseline and never task-qualified. If video is
requested for that baseline, the same capture and coherent-motion checks still
apply; static/noisy output fails and retains diagnostic artifacts. Independently
retrieve and hash every artifact, inspect the playable video for visible
contact/door progress, and verify authenticated Agent playback before claiming
qualification. Never infer success from nonzero actions or an arbitrary joint.

Historical exact-digest evidence comprises independent 1,050-step episodes on
B200 `(10, 0)` and RTX PRO 6000 `(12, 0)`. B200 retained five task artifacts /
86,082 bytes with no MP4. RTX retained six / 1,119,004 bytes, including an
independently decoded 1,024,140-byte H.264 viewport MP4 at 1280×720 for 70.067
seconds. Because that video was not checked for temporal change and used a
zero-action policy, it is not meaningful visual evidence. The supported
comprehensive B200 YAML subsequently completed all four
seed states with 1,050 steps each and 20 independently hash-verified task
artifacts / 344,330 bytes. Consult
`npa/docker/workbench/blackwell-dc-images.json` for the machine-readable,
sanitized record.

## Build and release

Build only from a clean exact commit. Official public development bytes use
`dev-<full-git-sha>`; scan the built filesystem and history, not merely the
Dockerfile.

Publish official public images only through the trusted
`.github/workflows/publish-public-images.yml` path with its security, licensing,
SBOM, provenance, and anonymous-access gates. The local build helper refuses
direct pushes to the official public namespace.

```bash
bash npa/docker/workbench/isaac-arena/build.sh
npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py \
  --docker-image npa-isaac-arena:<tag>
```

Require a clean Omniverse payload scan, non-root user, exact Arena source
labels/license, empty runtime cache, anonymous digest resolution, and real
completed-episode runs on both B200 (`sm_100`) and RTX PRO 6000 (`sm_120`)
before promotion. The RTX run must use a nonzero replay/RSL-RL input, complete a
successful task, and pass denoised coherent-motion checks bound to current-run
simulator state; a decodable static or noise-only video is failure. Preserve the
B200 no-video and RTX required-video distinction in evidence.

Cancel exact workflow runs before removing any dedicated resources. Do not
destroy shared clusters, buckets, or reserved capacity after a validation run.
Publication-failure copies are private mode `0700` but are not automatically
bounded or pruned. Recover or remove the reported local directory after triage;
repeated failures can consume worker disk.

## Diagnose

- Exit 78 before download: explicit EULA opt-out; do not bypass it.
- No episode JSONL: use `--num-episodes`, not an incomplete step-only smoke.
- Replay ends before an episode result: verify that the recorded initial state
  was applied and the input matches the environment/embodiment/action space.
  Inspect the declared quaternion representation and its recorded conversion;
  preserve every physical command and its ordering. A recording that still cannot complete a scored
  episode needs a compatible input; do not extend it with held actions.
- Missing `params/agent.yaml`: stage the complete RSL-RL checkpoint directory.
- Missing `lightwheel_sdk`: reject that image as incomplete; the accepted image
  must contain hash-locked SDK 1.0.3 while retaining an empty asset cache.
- Lightwheel registry denial or changed selector: treat it as an external
  input/access failure. Do not bake the returned USD or substitute another
  object while claiming the requested environment.
- Missing MP4 on RTX: inspect camera enablement, Vulkan/RT drivers, and the
  upstream viewport recorder. A missing native graphics userspace may use the
  exact-driver private extraction path; a version mismatch must fail closed.
  Check both `libnvoptix.so.1` and the documented weights path. The runtime-only
  container-overlay copy requires fresh image/GPU qualification; settings alone
  cannot override actual denoiser loading errors or qualify a noisy video.
  Do not alter the shared node or downgrade the artifact requirement.
- B200 render failure: the workload is misrouted. Keep B200 state-only and move
  rendering to RTX PRO 6000.
- Source/runtime incompatibility: retain the exact pins and report the alpha
  upstream boundary; do not patch around failures with fake output.

## Verify changes

```bash
npa/.venv/bin/python /home/ubuntu/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/tools/isaac-arena
npa/.venv/bin/python -m pytest npa/tests/workbench/test_isaac_arena.py npa/tests/docker/test_packaging_contract.py npa/tests/guardrails/test_skills_index.py -q
```
