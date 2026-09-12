# Isaac Lab-Arena policy evaluation

NPA supports the official Isaac Lab-Arena 0.3.0 policy evaluator as a pinned,
artifact-producing workbench job. The integration invokes upstream
`policy_runner.py`; it does not substitute a synthetic environment or infer
success from imports.

Upstream describes 0.3.0 as alpha, with unstable and incomplete APIs, and says
not to use it in production. NPA therefore supports only the immutable
evaluation contract documented here. This is a production-quality packaging
and operations surface around an explicitly pre-release upstream—not a claim
that the broader upstream project is production-ready.

## What the image contains

The public `npa-isaac-arena` image adds the Apache-2.0 Arena source at commit
`ed0fd12be862078be316c73eb7cf423ba9b1c5cd` and the upstream-declared
Apache-2.0 `lightwheel-sdk==1.0.3` client to the accepted payload-clean Isaac
Lab image. The source archive and Python wheels are SHA-256 verified. Upstream
tests, sample checkpoints, demonstration data, and documentation media are
removed.

The accepted release is
`npa-isaac-arena:0.3.0-isaaclab3-20260912`, manifest
`sha256:f07a7fd0f44e22ba3366437b0d0973869a0590919951d516150094220939416f`.
It was promoted without rebuilding from source revision
`22783a16abcd424df540b71e94600d705b317f9b` after complete payload/security,
SBOM, provenance, bootstrap, anonymous-pull, and two-platform workload gates.

The image contains no Isaac Sim/Lab/Omniverse Kit payload, weights, replay data,
operator checkpoints, Lightwheel registry assets, credentials, results, or
populated runtime cache. On
first use, `/isaac-sim/python.sh` fetches the pinned Isaac runtime from NVIDIA
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
entries in the authenticated Agent UI's `/api/tools` catalog embed the same
payload and name the CLI command. Together with those digest-bound records, it
is the source of truth when a broader upstream feature exists but is not an NPA
claim.

The evaluator supports the three local policy adapters registered by upstream:

- `zero_action`: no input; useful as a simulator and metric baseline, but never
  meaningful visual evidence merely because its MP4 decodes.
- `replay`: `--input-path` resolves to an Isaac Lab episode HDF5 file. NPA
  rejects effectively zero actions or a trajectory with no changing recorded
  state before starting the simulator. The generic upstream loader otherwise
  copies every recorded observation/state tensor to CUDA even though its replay
  adapter consumes only actions and the initial state, so NPA materializes
  exactly those required fields in owner-private scratch. When a recording is
  shorter than a known task horizon, `--replay-target-steps` may hold its final
  recorded action through that horizon; it refuses to truncate source actions.
- `rsl_rl`: `--input-path` resolves to `model*.pt` or a directory containing
  exactly one such checkpoint with sibling `params/agent.yaml`.

The pinned source also contains separate OpenPI, GR00T, Cosmos, and DreamZero
remote-policy packages. NPA does not expose those adapters through this public
runner: each requires a separately operated model server, compatible
embodiment adapter, and its own model/runtime terms. Arbitrary dynamic policy
and external-environment import paths are also unsupported.

The meaningful replay example uses an operator-owned copy of upstream's
Apache-2.0 `test_demo_gr1_open_microwave.hdf5` fixture at the exact pinned
revision. The input is never baked into or redistributed with the public image.

```bash
npa workbench isaac-arena evaluate \
  --output-path ./arena-results \
  --environment gr1_open_microwave \
  --policy-type replay \
  --input-path ./test_demo_gr1_open_microwave.hdf5 \
  --replay-target-steps 250 \
  --embodiment gr1_pink \
  --object tomato_soup_can \
  --record-video \
  --run-id "<run-id>"
```

Inputs and outputs may be local or operator-owned S3 paths. The simulator
subprocess receives no cloud, model, or HTTP admission credentials. Each result
contains raw episode JSONL, upstream static HTML, the credential-isolated simulator
log, and `result.json` with aggregate metrics, GPU identity, byte sizes, and hashes.
The original operator input and the normalized execution HDF5 are never output
artifacts. The result redacts their location and binds both the source SHA-256
and exact executed-input SHA-256, along with source, executed, and final-action
hold step counts.
`--record-video` additionally fails unless upstream writes H.264 at least
320×240 and one second long, and independent FFmpeg sampling finds at least two
frame pairs with both mean absolute luma delta ≥1.0 and ≥0.5% of pixels changing
by at least eight luma levels. The result records full video metadata, decoded
sample counts, measured change, thresholds, video hash, exact run id, upstream
run directory, policy adapter, and policy-input hash. A technically valid static
MP4 therefore fails closed. State-only evaluations retain their original
contract and do not need video.

The Python SDK is the same implementation:

```python
from npa.sdk.workbench.isaac_arena import evaluate

result = evaluate(
    output_path="./arena-results",
    environment="gr1_open_microwave",
    policy_type="replay",
    input_path="./test_demo_gr1_open_microwave.hdf5",
    replay_target_steps=250,
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

## Supported workflows

- `workflows/testing/isaac-arena-evaluation-b200.yaml` runs a sequential
  four-seed zero-action `cube_goal_pose` state regression on one B200. B200 has
  no RT cores, so this path makes no render or meaningful-motion claim.
- `workflows/testing/isaac-arena-evaluation-rtxpro.yaml` replays the nonzero
  operator-owned trajectory for the 250-step task horizon on RTX PRO 6000 and requires a
  motion-validated viewport MP4.

Validate and plan before submission, substitute an operator-owned bucket, and
pass the standard S3 credentials through workflow secret handling. No model
credential is required for this replay. Its exact HDF5 input must already exist
in operator-owned storage, and the worker needs outbound access to the upstream
Lightwheel registry for the runtime-only microwave asset.

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

## Accepted workload evidence

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

The current release gate requires the genuine replay workflow above on RTX PRO
6000, plus the established B200 state-only regression because common upstream
execution and result parsing changed. Record exact live measurements
here only after both runs and the authenticated Agent UI checks complete.

Arena emits MP4 and HTML/JSON evidence, not a native `.rrd`. The Agent UI must
use its authenticated video renderer for this workload; an unrelated Rerun
recording must not be substituted or relabeled.

For exact build, scan, qualification, and cleanup rules, use
`skills/tools/isaac-arena/SKILL.md`.
