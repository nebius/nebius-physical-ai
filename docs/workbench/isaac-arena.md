# Isaac Lab-Arena policy evaluation

NPA supports the official Isaac Lab-Arena 0.3.0 policy evaluator as a pinned,
artifact-producing workbench job. The integration invokes upstream
`policy_runner.py`; it does not substitute a synthetic environment or infer
success from imports.

Upstream describes 0.3.0 as alpha, with unstable and incomplete APIs, and says
not to use it in production. NPA therefore supports only the immutable
evaluation contract documented here. Its packaging and qualification records
do not make the broader upstream project production-ready.

## What the image contains

The public `npa-isaac-arena` image adds the Apache-2.0 Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd` and the upstream-declared
Apache-2.0 `lightwheel-sdk==1.0.3` client to the accepted payload-clean Isaac
Lab image. The source archive and Python wheels are SHA-256 verified. Arena
tests, sample checkpoints, demonstration data, and documentation media are
removed.

The historical published release is
`npa-isaac-arena:0.3.0-isaaclab3-20260912-r2`, manifest
`sha256:5e2099a83ce4fd090bcb9bbac3004f5ddf37ef9e2c763a225b04e5daab46a4c2`.
It was promoted without rebuilding from source revision
`7dd3a2bf3aa228dd3c201ba72ac0aae3d9559ab1` after complete payload/security,
SBOM, provenance, bootstrap, anonymous-pull, and two-platform workload gates.
Its RTX visual qualification was subsequently rejected: stochastic rendering
noise passed the former pixel-change checks, while the task failed. The current
source corrects replay-state initialization and visual acceptance. A new image
must pass fresh digest-bound B200 and RTX qualification before promotion; the
historical release is not evidence for these corrections.

The image excludes Isaac Sim/Lab/Omniverse Kit runtime payloads, Arena policy
checkpoints and replay datasets, operator inputs, Lightwheel registry assets,
credentials, evaluation results, and populated runtime caches. Its inherited
open-source Python environment includes public dependency examples and test
fixtures, such as the Newton sample policy and ONNX conformance models, with
their installed package licenses. Those fixtures are not a ready-to-run Arena
policy or validation evidence.

On first use, `/isaac-sim/python.sh` fetches the pinned Isaac runtime from NVIDIA
under the operator's `ACCEPT_EULA` setting. The compatible NPA baseline is Isaac
Lab `3.0.0b2.post1` with Isaac Sim `6.0.1.0`.

The SDK is not the asset license. `gr1_open_microwave`,
`franka_put_and_close_door`, and `press_button` use Lightwheel-backed fixtures;
`put_item_in_fridge_and_close_door` uses a Lightwheel kitchen. Those bytes are
resolved only during the operator's run. The upstream provider controls access,
selectors, and terms; NPA supplies no Lightwheel credential, redistributes no
registry asset, and makes no claim that other Lightwheel objects are available
or licensed. The capability JSON exposes this requirement on each applicable
environment.

## Supported policies

```bash
npa workbench isaac-arena capabilities
```

This JSON command and `npa.sdk.workbench.isaac_arena.capabilities()` enumerate
the pinned alpha surface with explicit `implemented`, `input_required`,
`unsupported`, and `upstream_alpha` states. Digest-specific `live_validated`
claims deliberately remain in the readiness and release records: an image
cannot pre-assert qualification of its own not-yet-published digest. Both Arena
entries are listed by the authenticated Agent UI's `/api/tools` endpoint; their
`/api/tools/{tool_ref}` detail responses embed the same payload and name the CLI
command. Together with those digest-bound records, it
is the source of truth when a broader upstream feature exists but is not an NPA
claim.

The evaluator supports the three local policy adapters registered by upstream:

- `zero_action`: no input; a simulator and metric baseline, never task-qualified
  visual evidence. Requested video must still pass the capture and coherent-motion
  checks; a static or noise-only baseline fails video qualification.
- `replay`: `--input-path` resolves to an Isaac Lab episode HDF5 file. NPA
  selects the first sorted episode and requires finite multi-step actions and a
  finite `initial_state` group. Recorded-state histories and source success are
  optional diagnostics, never the current run's outcome. Zero-action recordings in compatible action spaces
  and missing or static state histories remain valid ordinary evaluation inputs;
  replay visual qualification additionally requires measured nonzero source
  actions. The generic upstream loader otherwise
  copies every recorded observation/state tensor to the execution device even though its replay
  adapter consumes only actions and the initial state, so NPA materializes
  exactly those required fields in owner-private scratch. The runner applies
  the recorded initial state through Isaac Lab's `reset_to(is_relative=True)`
  and executes the source actions once, without padding, repetition, truncation,
  or a final-action hold. Replay evaluates one recorded episode in one
  environment.
- `rsl_rl`: `--input-path` resolves to `model*.pt` or a directory containing
  exactly one such checkpoint with sibling `params/agent.yaml`.

Replay follows Isaac Lab's dataset format metadata: absent or integer `0` means
legacy WXYZ quaternions; integer `1` means XYZW. Unknown or malformed versions
are rejected. The pinned Lab loader converts legacy root poses but does not
convert quaternions embedded in Pink actions. For the resolved `gr1_pink`
embodiment, NPA requires its 36-column action layout and single-environment
initial robot root pose. It converts both hand-target quaternion slices (`3:7`
and `10:14`) and initial-state root quaternions to XYZW in the private execution
file, then marks that file version 1. Quaternion norms must be nonzero. The
conversion preserves physical orientations, positions, hand commands, other
state values, and the number and order of actions. Source bytes remain
untouched; the result records both file hashes and the exact representation
change. Native version-1 Pink inputs are not converted again. Other embodiments
retain upstream root-pose handling and require actions that already match their
native controller; NPA does not infer a Pink layout from action width or
numerical values.
The [pinned dataset loader](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py)
defines those format versions and its root-pose conversion boundary.

When a Pink recording also supplies native
`obs/datagen_info/target_eef_pose/left` or `right` matrices, NPA checks each
provided hand target against the corresponding action's position and declared
quaternion interpretation before creating execution input. Malformed matrices,
different coordinate frames, misaligned samples, and contradictory quaternion
conventions are rejected. A mixed-format recording requires an explicit,
source-specific normalization with retained provenance; changing its version
label alone can corrupt root poses. NPA never chooses a convention from a
replay's apparent success. The consistency result describes source coordinates
and does not establish current-run motion or task success.

The pinned source also contains separate OpenPI, GR00T, Cosmos, and DreamZero
remote-policy packages. NPA does not expose those adapters through this public
runner: each requires a separately operated model server, compatible
embodiment adapter, and its own model/runtime terms. Arbitrary dynamic policy
and external-environment import paths are also unsupported.

The replay qualification example uses an operator-owned copy of upstream's
Apache-2.0 `test_demo_gr1_open_microwave.hdf5` fixture at the exact pinned
revision. The input is never baked into or redistributed with the public image.

```bash
npa workbench isaac-arena evaluate \
  --output-path ./arena-results \
  --environment gr1_open_microwave \
  --policy-type replay \
  --input-path ./test_demo_gr1_open_microwave.hdf5 \
  --embodiment gr1_pink \
  --object tomato_soup_can \
  --execution-device cpu \
  --record-video \
  --run-id "<run-id>"
```

Inputs and outputs may be local or operator-owned S3 paths. NPA removes cloud,
model, and HTTP admission secrets from the simulator subprocess environment,
including the Agent's artifact-reader and observability credentials. A completed
evaluation contains raw episode JSONL, upstream static HTML, the current run's simulator
ground-truth HDF5, the credential-isolated simulator log, and `result.json` with
aggregate metrics, GPU identity, byte sizes, and hashes. After evaluation setup
succeeds, handled runtime or evidence failures write a failed `result.json`,
retain the artifacts actually produced, and attempt publication. A storage
publication failure retains the private local copy. Artifacts may be partial logs, scalar phase journals, raw video,
a capture sidecar, or initial/terminal PNGs; completed-episode JSONL, scored HDF5,
and an HTML report are not guaranteed. An unfinished recorder buffer is retained
separately as unscored diagnostic data when finalization runs. Request validation
or input/setup failures can occur before a result tree exists, and interrupted
workers may leave only workflow logs.
The original operator input and the normalized execution HDF5 are never output
artifacts. The result redacts their location and binds both the source SHA-256
and prepared execution-input SHA-256, along with source and prepared step counts.
Observed action counts and task outcomes come from the simulator evidence.
A completed evaluation may truthfully report a success rate of zero. Nonzero
actions and movement metrics never substitute for the upstream task result.

Replay evaluation requires `--num-episodes 1` and `--num-envs 1`, including
state-only replay. NPA selects the first sorted episode in the input HDF5.
`--record-video` also requires one episode in one environment and H.264 at least
320×240 and one second long. The untouched source MP4 is retained beside a
labeled FFmpeg-denoised derivative. Acceptance checks temporal-median samples
for coherent spatial change and binds that interval to the same run's simulator
metric trace. Task-qualified video currently supports only
`gr1_open_microwave` with `replay` or `rsl_rl`: the upstream JSONL, numeric
`success_rate > 0`, and HDF5 success flag must agree, the final door openness
must exceed the upstream threshold of `0.8`, and maximum openness must increase
at least `0.5` from the initial state. The acceptance interval runs from the
first measured door progress to the first threshold crossing, excluding idle
and reset frames. Other registered scored environments remain available for
ordinary evaluation; their nonzero-policy video qualification is unsupported.

Successful video qualification retains `simulator-video-evidence.json`, an actual
initial PNG, and an actual terminal PNG captured before automatic reset. The sidecar binds each
PNG's file and decoded-RGB SHA-256 to its action step; contiguous capture indices
must match the simulator HDF5. The terminal PNG must also match the corresponding
decoded source-MP4 frame within encoding tolerances. The result records these
bindings, initial/final state, measured changes, thresholds, source and derivative
hashes, run identity, and input hashes. A static scene with rendering noise fails
the gate. A zero-action baseline is never task-qualified, even if it passes the
capture and coherent-motion checks. Failed qualification retains diagnostic
artifacts. State-only evaluation reports scored outcomes, including zero success,
without requiring successful visual qualification.

NPA's context-bound source patch applies replay initial state, retains the
simulator metric-recorder HDF5, captures the initial revolute-joint state, and
records actual viewport frames after each action and before automatic reset.
The Gym recorder uses those cached simulator frames. Capture freezes Kit's
simulation updates and checks native PhysX step-event counts and elapsed event
time, the Lab physics step counter, and
uncached robot/object state before and after every render. The version-2 capture
sidecar must contain one matching check for the initial frame and every action.
The retained native subscription must observe progress between real actions;
an inactive observer cannot provide a freeze proof. Its elapsed time starts at
capture setup and is not an absolute simulation clock.
Texture streaming and asset loading must finish before capture requests 32
samples per pixel on each of four frozen path-tracing updates with the
NGX-independent OptiX denoiser. The proof records settings and update counts;
it does not measure the renderer's actual accumulated sample count.
The initial and terminal PNGs must also contain nonblack pixels; real task
progress and coherent motion remain separate required checks. Renderer settings
are read back and verified. The patch preserves early Kit
camera enablement required by `env.render()`, but prevents that recorder choice from also adding
the embodiment's unused observation cameras. Upstream camera-observation video
is not exposed by this adapter. The patch is context-bound to the pinned source
and the image build fails if that source block changes.

When the action sequence ends before an episode finishes, a separate unscored
diagnostic retains the remaining metric buffer and simulator state. It does not
create a completed episode, success flag, or scored HDF5 record. This keeps a
failed replay diagnosable while preserving the upstream result.
Replay input metadata reports `prepared_steps` and `prepared_sha256`; these
describe the private execution input prepared before launch. They do not count
actions actually executed or imply a task result, including when setup fails.

When emitted, `simulator-phases-rank*.jsonl` records fixed phase/event labels,
monotonic timestamps, rank, action/render counters, and observed readiness
booleans. It distinguishes policy calls, Pink IK, action application, scene
writes and updates, simulator steps, native physics waits and steps, capture,
and individual render calls without retaining their arguments, input arrays,
or exception text. Native classmethod binding, arguments, return values, and
exceptions remain unchanged; journal I/O adds wall-clock overhead. The rollout
finalizer restores only its own method bindings, including on failure.
Bindings without a writable instance attribute dictionary remain untouched and
emit an `unavailable` event for that method phase. Their enclosing simulator
operation can still be observed; unavailable hooks do not block a rollout or
claim that the underlying operation ran.
An unfinished phase identifies the last observed operation;
it does not establish a timeout, a failed task, or successful progress. Journal
writing is best effort and does not replace the native result or exception.
In particular, a native physics-step phase includes simulation, result fetching,
and callback handling; entry alone does not isolate a defect in any one of them.

The viewport path also fails closed on the target graphics stack. NPA first
probes native NVIDIA EGL, Vulkan, and `libnvoptix.so.1`, and requires a readable,
regular, nonempty `/usr/share/nvidia/nvoptix.bin`. NVIDIA documents that data
file in its [driver component reference](https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/README/installedcomponents.html).
A CUDA-capable target can omit these libraries or weights. In that case, the
worker queries the one loaded driver version and downloads only the exactly
matching `libnvidia-gl-<branch>-server` package from Ubuntu's signed archive.
It validates package name, version, architecture, SHA-256, and the packaged ICD
metadata, then derives a canonical run-private EGL ICD because NVIDIA documents
EGL as the headless Vulkan entrypoint.

The image contains an empty `/usr/share/nvidia` directory owned by its non-root
runtime user. Missing weights are copied from the validated package only when
that directory belongs to the same private root overlay as the container's `/`.
Symlink, shared-writable, foreign-owned, and mounted destinations are refused.
The copy is atomically published without overwriting an existing file and its
hash is read back; an existing different payload is rejected. Libraries remain
in private scratch, and copied weights remain only for the worker container
lifetime. NPA never installs these bytes on the node, bakes them into the public
image, publishes them as run artifacts, or redistributes them. The result records
safe package/manifest/weights hashes, byte size, and native or container-overlay
placement. File and settings readiness do not prove denoising: actual OptiX
renderer errors reject video qualification, which still requires clean visible
task progress. This correction requires a fresh immutable image and GPU
qualification before promotion. State-only runs do none of this graphics setup.

The Python SDK is the same implementation:

```python
from npa.sdk.workbench.isaac_arena import evaluate

result = evaluate(
    output_path="./arena-results",
    environment="gr1_open_microwave",
    policy_type="replay",
    input_path="./test_demo_gr1_open_microwave.hdf5",
    execution_device="cpu",
    object_name="tomato_soup_can",
    record_video=True,
    run_id="<run-id>",
)
```

## Environment and task surface

The pinned registry contains 18 Python environments. This table identifies the
task and default embodiment/object rather than implying that every combination
has been live-qualified.

| Environment | Task | Default embodiment / object | Metrics | NPA state |
| --- | --- | --- | --- | --- |
| `cube_goal_pose` | goal pose | `franka_ik` / `dex_cube` | success, object moved | implemented; historical live baseline |
| `dexsuite_lift` | DexSuite lift | `kuka_allegro` / task object | success | implemented; unvalidated |
| `droid_table_multi_object_placement` | no scored task | `droid_abs_joint_pos` / object pool | none | unsupported as evaluation |
| `franka_put_and_close_door` | sequential pick/place + close door | `franka_ik` / `dex_cube`; Lightwheel microwave required | success, movement, subtask | implemented; input required; unvalidated |
| `galileo_g1_locomanip_pick_and_place` | pick/place | `g1_wbc_pink` / `brown_box` | success, object moved | implemented; unvalidated |
| `galileo_pick_and_place` | pick/place | `gr1_pink` / `power_drill` | success, object moved | implemented; unvalidated |
| `gr1_open_microwave` | open door | `gr1_pink` / none; Lightwheel microwave required | success, joint moved | implemented; input required; replay qualification is digest-bound in readiness evidence |
| `gr1_table_multi_object_no_collision` | no scored task | `gr1_joint` / object pool | none | unsupported as evaluation |
| `gr1_turn_stand_mixer_knob` | turn knob | `gr1_pink` / none | success, joint moved | implemented; unvalidated |
| `kitchen_pick_and_place` | pick/place | `franka_ik` / `cracker_box` | success, object moved | implemented; unvalidated |
| `lift_object` | lift / RL lift | `franka_joint_pos` / `dex_cube` | success | implemented; unvalidated |
| `peg_insert` | assembly | `franka_ik` / `peg` → `hole` | success, object moved | implemented; unvalidated |
| `pick_and_place_maple_table` | pick/place | `droid_abs_joint_pos` / RoboLab cube | success, object moved | defaults only; unvalidated |
| `press_button` | press | `franka_ik` / selected pressable; Lightwheel coffee machine required | success | implemented; input required; unvalidated |
| `put_item_in_fridge_and_close_door` | sequential pick/place + close door | `gr1_pink` / HOPE dressing bottle; Lightwheel kitchen required | success, movement, subtask | implemented; input required; unvalidated |
| `gear_mesh` | assembly | `franka_ik` / medium gear | success, object moved | implemented; unvalidated |
| `tabletop_place_upright` | place upright | `agibot` / `mug` | success, object moved | implemented; unvalidated |
| `tabletop_sort_cubes` | sort | `franka_ik` / red + green cubes | success | implemented; unvalidated |

Embodiment and object overrides are registry names, not arbitrary strings. The
selected embodiment must match the environment and policy action space; for
example, `gr1_open_microwave` explicitly accepts only `gr1_joint` or
`gr1_pink`. Objects must also implement the task-required asset/affordance type,
and some RoboLab/SimReady assets need an operator-authorized runtime source.

Upstream additionally ships 31 kitchen-benchmark graph specs, three Maple-table
specs, and 38 RoboLab task plus 17 RoboLab scene specs. They remain upstream
alpha and are listed by the capability payload as unsupported because this NPA
surface does not yet accept `--env_spec`. This is not a broad graph-catalog or
external-plugin support claim.

The capability payload also lists the broader upstream workflow families under
`upstream_workflows`. Each is explicitly `unsupported`, `input_required`, and
`upstream_alpha` in this Arena integration:

| Upstream workflow | Required inputs and NPA boundary |
| --- | --- |
| Agentic environment generation | A prompt or graph specification, with model access for prompt resolution and authorized assets for building. NPA does not expose the experimental CLI, review GUI, or generated-scene build workflow. |
| Experiment runner | Experiment configuration, compatible policies, and task assets. Upstream local experiment orchestration and OSMO submission are not exposed by NPA's SkyPilot evaluation workflows. |
| Sensitivity analysis | Episode outcomes with recorded variation factors. Ordinary evaluation metrics do not constitute upstream posterior estimation or sensitivity reports. |
| Teleoperation | A registered task, compatible input device, and assets. The upstream Isaac Lab demonstration recorder is not an NPA Arena command. |
| Data generation | Demonstrations, task annotations, and Isaac Lab Mimic configuration. Replay evaluation does not annotate or generate demonstrations. |
| Imitation learning | A demonstration dataset, training configuration, and authorized model access. The upstream GR00T conversion and fine-tuning workflow uses a separate training environment. |
| Reinforcement learning | A registered Arena task and Isaac Lab training configuration. Evaluating an RSL-RL checkpoint does not train it. |

These entries reference files in the [pinned upstream revision](https://github.com/isaac-sim/IsaacLab-Arena/tree/ed0fd12be862078be316c73eb7cf423ba9b1c5cd).
Schema and catalog inspection need neither a prompt nor model access. Separate NPA
training or teleoperation tools retain their own supported contracts; their
existence does not qualify these Arena workflows.

## Supported workflows

- `workflows/testing/isaac-arena-evaluation-b200.yaml` runs a sequential
  four-seed zero-action `cube_goal_pose` state regression on one B200. B200 has
  no RT cores, so this path makes no render or meaningful-motion claim.
- `workflows/testing/isaac-arena-evaluation-rtxpro.yaml` replays the nonzero
  operator-owned trajectory on RTX PRO 6000 and requires successful task
  execution plus simulator-ground-truth-bound viewport motion. It selects CPU
  physics and replay tensors following the upstream GR1 tutorial, with RTX GPU
  viewport rendering. The NPA source patch avoids constructing unrelated
  embodiment-camera sensors;
  native graphics is preferred and the exact-driver private extraction above is
  used only when required by the target.

Upstream documents CPU physics for its generated demonstrations and recommends
the CPU replay path, while warning that reset nondeterminism can still prevent
reproduction. The small test fixture does not record its source physics device;
the selected workflow must pass fresh task and visual validation. CPU physics
with RTX rendering is qualified only by a readiness record for the exact image
digest, target hardware, and task; selecting the device alone establishes no result.
See the [pinned upstream replay guidance](https://github.com/isaac-sim/IsaacLab-Arena/blob/ed0fd12be862078be316c73eb7cf423ba9b1c5cd/docs/pages/example_workflows/static_manipulation/step_3_data_generation.rst#L124-L143).
  The result records both the execution device and measured GPU identity.

Validate and plan before submission, substitute an operator-owned bucket, and
pass the standard S3 credentials through workflow secret handling. No model
credential is required for this replay. Its exact HDF5 input must already exist
in operator-owned storage, and the worker needs outbound access to the upstream
Lightwheel registry for the runtime-only microwave asset.

The live-submit test harness downloads the exact pinned upstream replay without
operator credentials, checks its SHA-256 and HDF5 content, and stages it under
the invocation's E2E prefix. It rewrites the test's input URI to that private
key and verifies the stored bytes again. The reusable workflow itself retains
an operator-supplied input URI; production inputs are never bundled into tests
or the public image.

```bash
npa workbench health preflight --checks nebius,s3
npa workbench workflow validate-spec workflows/testing/isaac-arena-evaluation-b200.yaml
npa workbench workflow plan-spec workflows/testing/isaac-arena-evaluation-b200.yaml
```

On a cluster whose SkyPilot accelerator spelling is already verified, pin it
explicitly to keep submission identity stable. Use the corresponding RTX name
and YAML for the video workflow.

```bash
export NPA_WORKFLOW_GPU_ACCELERATOR='B200:1'
npa workbench workflow submit \
  workflows/testing/isaac-arena-evaluation-b200.yaml \
  --runtime --durable-s3 --max-wait-seconds 0 \
  --project "<project-alias>" --infra "k8s/<context>" \
  --var "bucket=<operator-owned-bucket>"
```

## Historical workload evidence and current qualification

The first exact release digest completed zero-action baseline evaluations on
B200 and RTX PRO 6000. Those runs remain valid simulator/report evidence, but
their success rate was 0.0 and the RTX recording was not inspected for temporal
motion. It is historical decodability evidence only and no longer satisfies a
meaningful visual claim.

That release's B200 zero-action workflow completed its four-state seed sweep.
All four episodes ran 1,050 scored steps at capability `(10, 0)` and retained
five independently hash-verified non-video artifacts each: 20 task artifacts /
344,330 bytes in total. Every live job was terminal or repeat-safely cancelled,
and an independent pod audit found no run-owned worker. Run-created controllers
were removed; the pre-existing shared controller, clusters, and operator storage
were retained.

The historical r2 release ran two isolated, concurrently launched
workflows against reserved capacity. `arena-replay-rtx-7dd3a2bf-r1` ran the
`replay` adapter for 250 steps on RTX PRO 6000. Its source fixture was 365,180
bytes with SHA-256
`154ebea7839ec53e6ac441e18f1404b3fe140c3f004ad7e309519ba37274fa50`:
80 nonzero action steps, 36 action dimensions, nonzero fraction 1.0, mean
absolute step delta 0.0183097, and a recorded-success trajectory. The private
execution copy held the final action for the remaining 170 task steps without
truncating or publishing the source. Upstream reported one 250-step episode and
`revolute_joint_moved_rate=1.0`. Success rate 0.0 remains the policy result; it
is retained unchanged. This run is rejected as meaningful visual/task evidence:
the source state was not applied, most execution steps held the final action,
the task failed, and render grain could satisfy the old pixel thresholds.

The run retained 18 durable objects / 72,726,403 bytes. All six declared task
artifacts were downloaded again and hash-verified. The input- and run-bound
MP4 was 72,519,371 bytes, H.264, 1280×720, 5.04 seconds, and 252 frames. Twenty
independently decoded samples produced 19 changed frame pairs, maximum changed
pixel ratio 0.5391667, and maximum mean absolute luma delta 12.5636111. These
measurements exceeded the former raw-pixel thresholds, which did not distinguish
render noise from task-relevant motion. These byte and decode measurements
remain historical facts; they do not qualify the visual result.

`arena-state-b200-7dd3a2bf-r1` ran the same digest through the four-seed B200
state regression. Seeds 42–45 each completed one 1,050-step episode and reported
`object_moved_rate=1.0`, for 4,200 total steps. Independent validation re-read
49 durable objects / 502,340 bytes, including 24 task files / 358,401 bytes.
This zero-action regression truthfully records `meaningful=false` and emits no
video or `.rrd`.

The authenticated Agent UI discovered the RTX run through its artifact-first
run list, retained the exact server-issued source tuple, selected the MP4 as a
video, and returned the same size and SHA-256. Unauthenticated tool, detail,
load, status, and media requests returned 401; authenticated requests returned
200, and `bytes=0-1023` returned 206 with `Content-Type: video/mp4` and range
support. Exact terminal jobs were cancelled before their run-owned controllers
and local APIs were removed. No run-owned worker or controller remained; the
shared clusters, operator storage, and reserved GPU nodes were retained.

Arena emits MP4 and HTML/JSON evidence, not a native `.rrd`. The Agent UI must
use its authenticated video renderer for this workload; an unrelated Rerun
recording must not be substituted or relabeled.

Inspection of the selected legacy replay also found a controller compatibility
defect: its WXYZ hand-target quaternions reached the installed Lab 3 Pink
controller's XYZW parser without conversion. Interpreting the same targets in
the wrong order changes their physical orientations. The replacement normalizer
handles this declared representation change explicitly; a successful source
recording still does not establish a successful live replay.

The replacement candidate's image/security gates, B200 state regression, RTX
successful task video, independent artifact readback, and authenticated playback
remain pending in the adjacent workflow readiness records until executed. No
historical digest or run is reused to claim the changed implementation passed.

For exact build, scan, qualification, and cleanup rules, use
`skills/tools/isaac-arena/SKILL.md`.
