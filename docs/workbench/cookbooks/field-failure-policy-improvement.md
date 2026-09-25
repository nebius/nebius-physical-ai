# Field failure to navigation policy improvement

[Workflow](../../../workflows/testing/field-failure-policy-improvement.yaml) ·
[Readiness](../../../workflows/testing/field-failure-policy-improvement.readiness.json)

This workflow validates field captures, executes reconstruction and navigation
fine-tuning adapters, runs the existing and candidate policies against identical
held-out scenarios, and writes an evidence-backed promotion recommendation.
The YAML owns all six stages and runs through the standard SkyPilot runtime.
There is no deployment stage or additional orchestration service.

**GPU acceptance has not been run.** Proprietary robot/scan integration remains
operator input. The repository supplies executable validation, adapter invocation,
durable handoffs, comparison, and native reference adapters described below.
The reference requires the metric RGB-D reconstruction and shared-scene Isaac
navigation components in the same reviewed source distribution. It does not
establish compatibility with a proprietary robot or trainer. Missing components,
adapters or invalid evidence fail the stage. Schema validity does not establish readiness.

## Native metric-capture and Isaac reference

The concrete adapter entrypoints are
`npa.workflows.field_failure.native_reconstruction:reconstruct`,
`npa.workflows.field_failure.native_policy:train`, and
`npa.workflows.field_failure.native_policy:evaluate`.
They call the actual metric RGB-D reconstruction, OpenUSD scene assembly and
native Isaac Lab learner; they do not substitute fixture scores. This draft
integration depends on the scan and shared-scene navigation implementations
being included together. Their absence is an import failure, never a fallback.

Reconstruction needs Open3D 0.19, Pillow, NumPy and OpenUSD in its selected
Python runtime. This TSDF stage runs on CPU. The training and evaluation image
must provide the pinned Isaac Sim 6.0.1 / Isaac Lab 3 beta runtime at
`/isaac-sim/python.sh`, with an RTX GPU for native simulation and rendering.
Use a reviewed content-addressed NPA source overlay containing all three
components, or bake that exact source into an immutable operator image.
Source overlay bytes are separate from the base-image digest. The protocol
additionally pins the navigation module inventory digest, and every recipe must
match it before native adapter import. Preserve the standard submission's full
source-archive identity alongside the run evidence.

Each capture artifact is a regular-file-only tar bundle containing `capture.json`,
its hash-bound RGB/depth files, and `recipe.json`. The recipe selects training
cases and native hyperparameters; reconstruction replaces its scene filename and
digest with the actually derived USDZ. Each untouched held-out artifact is a tar
bundle containing `scene.usdz` and `recipe.json`. Its `eval_cases` must contain
exactly the declared held-out seeds, one per robot. Archive links, traversal,
duplicate members and caller-selected initial checkpoints are rejected.

The common protocol is a JSON artifact with exactly these fields:
`schema_version: npa.field-failure.native-protocol.v1`, `navigation_image`,
`task`, `adapter_module`, `adapter_sha256`, and `source_bundle_sha256`.
The last field is the exact installed navigation-module inventory digest, not
the full source archive. Pin identical protocol bytes for both policy arms.
The scene recipes separately seal the training/evaluation settings, sensor mode,
physical controls and reset cases; paired arms consume identical held-out bytes.

Training loads the supplied baseline checkpoint before native PPO. With multiple
training scenes, each subsequent scene continues from the previous result. Each
scene also receives a baseline and candidate diagnostic replay of its declared
training cases, with complete native artifacts retained. These replays are
explicitly **training-exposed** and never contribute to promotion. Supply one
training case per robot so the original scenarios can be replayed exactly.

Held-out evaluation reloads each exact checkpoint and recomputes outcomes from
the measured native trajectories. Supported metric names are `success`,
`collision_steps`, `peer_collision_steps`, `physical_failure_steps`,
`path_length_m` and `goal_distance_m`. Physical failures such as falling or losing
measured floor support remain failed episodes even when the robot reaches the
goal's horizontal coordinates. Their raw upright and ground-clearance
measurements remain in the retained native trajectories.
Failed and timed-out rollouts remain completed comparison evidence. The native
standalone quality threshold does not discard a losing baseline arm; the sealed
paired comparison determines the recommendation. Both the original failure
replay and the untouched held-out comparison must be shown when claiming improvement.
Each native held-out evaluation also retains its complete simulator output in
`evaluation-<index>.tar` under that adapter attempt. Its companion
`native-evaluation-<index>.json` binds the archive digest, policy, protocol,
scenario and measured report. Native rendered frames/video and full trajectories
therefore remain available after the worker's temporary directory is removed.

## Required runtime and adapters

Use an existing authorized RTX PRO 6000 Kubernetes target for the declared
reconstruction, training, and evaluation profiles. NuRec and Isaac rendering need
RT cores; do not substitute H100/H200. Configure credentials and vendor access
through the normal [workflow preflight](../npa-workflow-guide.md), outside the
spec. This change builds no images and provisions no infrastructure.

Provide three installed Python `module:function` entrypoints in immutable
`image@sha256:<64 lowercase hex>` BYOF images. Each function accepts one request
dictionary, performs the real workload synchronously, publishes artifacts, and
returns the report dictionary specified below. Package dependencies in the
image; use the [BYOF guide](byof-isaac-lab/README.md) for packaging. The function
must return only after artifact writers have stopped. Upload to its supplied
attempt prefix with `StorageClient.put_bytes_conditional(..., if_none_match=True)`;
do not overwrite existing objects or launch detached writers.

| Adapter | Actual work required | Request-specific fields |
| --- | --- | --- |
| `reconstruct_adapter` | Decode the failure captures, reconstruct the observed environment, and produce simulator-loadable navigation scenes with calibrated scale, collision geometry, robot/sensor frames, and traversability. | `captures`, `protocol` |
| `train_adapter` | Load the exact existing baseline checkpoint and actually fine-tune it on the reconstructed training scenes. Retain native training logs and a changed checkpoint. | `baseline`, `scenes`, `reconstruction_sha256`, `protocol` |
| `evaluate_adapter` | Load the requested checkpoint and run every held-out scenario/seed using the same simulator, sensors, termination rules, and metric definitions. Retain trajectories and per-episode measurements. | `policy`, `held_out`, `protocol`, `metrics` |

All requests also contain `bundle_sha256`, `adapter`, `attempt_id`,
`inputs_sha256`, and `output_prefix`. Reconstruction and training requests exclude
held-out assets and the bundle URI. Training receives only training scenes.
Evaluation receives one policy at a time and no other policy's scores.

An NRE-capable reconstruction adapter can invoke the existing
[`npa workbench nurec reconstruct`](../guides/neural-reconstruction.md) or its
Python SDK after staging the capture as NCore. The adapter must additionally
supply navigation collision geometry and dynamics: an NRE Gaussian USDZ alone
does not establish physical traversability. Existing public PPISP object captures
can exercise reconstruction, but do not establish a navigation workload.

An Isaac navigation adapter should run its own registered navigation environment
and robot controller in the configured image; stock manipulation or Cartpole
scores cannot satisfy this task. For native Isaac Lab 3, use the version's
`--visualizer none` execution path. A separate public navigation implementation
reference is [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), whose
repository includes navigation training/evaluation and test PointNav inputs.
It is an adapter-development reference, not an integrated or accepted runtime
here; its upstream README also documents the end of internal maintenance after
v0.3.4. No upstream assets, datasets, models, or source are bundled by this change.

## Identity and trust boundary

The input bundle seals each adapter's exact `entrypoint`, `source_sha256`, and
`runtime_image`. Workflow configuration must match all three identities. Both
policies use the same sealed evaluation entrypoint and image. Changing the
function within the same file, image digest, or source bytes fails closed.

Source resolution uses `PathFinder` without executing package initializers. The
leaf source digest is checked before import. Parent initializers, dependencies,
import hooks, cached code, and native libraries belong to the **trusted immutable
BYOF image boundary**. This is not a sandbox for arbitrary imports. Do not mount
writable replacement adapter code or modify import paths at runtime. The image
identity is supplied by trusted workflow configuration, used both for scheduling
and for stage verification; it is not a hardware or registry attestation measured
inside the worker. Operators must independently verify image pullability,
provenance, runtime compatibility, and any image overrides. `source_overlay: true`
stages this checkout's NPA implementation, not the operator adapter.

Hashes establish byte identity and detect inconsistent evidence; they do not
prove an adapter honestly measured a physical robot. Operators own adapter
correctness, baseline training-history declarations, site/capture grouping,
calibration, artifact access controls, and protocol qualification. This workflow
checks declared separation; it cannot discover undisclosed training exposure.

## Sealed input bundle

Store one UTF-8 JSON object in S3 and pin its actual byte SHA-256 using
`bundle_sha256`. Unknown fields, duplicate JSON keys, non-finite numbers,
coercible string/boolean numbers, unsafe object keys, and missing evidence are
rejected. URIs must be exact S3 objects with no query, fragment, whitespace,
traversal, or percent-encoded segments. Archives remain opaque to this validator;
the consuming adapter must safely unpack and validate its native format.

The executable contracts live in
[`contracts.py`](../../../npa/src/npa/workflows/field_failure/contracts.py).
All fields below are required. SHA-256 values are 64 lowercase hex digits.
Identifiers use letters, digits, underscores, periods, or hyphens and start with
a letter or digit. Lists described as nonempty cannot be omitted or empty.

| Type | Fields |
| --- | --- |
| Artifact | `uri`, `sha256`; referenced bytes must exist, be nonempty, and hash correctly |
| Policy | `policy_id`, `checkpoint` (Artifact) |
| Adapter | `entrypoint` (`module:function`), `source_sha256`, `runtime_image` (immutable digest reference) |
| Capture | `scenario_id`, `group_id`, `asset` (Artifact containing native capture data) |
| Held-out scenario | `scenario_id`, `group_id`, `asset` (Artifact containing executable scene), nonempty unique nonnegative integer `seeds` |
| Metric | `name`, `direction` (`higher` or `lower`), nonnegative finite `minimum_improvement` and `maximum_regression`, finite `minimum` and `maximum` bounds with minimum less than maximum |

The bundle has `schema_version: npa.field-failure.bundle.v1`, `task: navigation`,
`baseline` (Policy), `baseline_training_groups` (list of group IDs), nonempty
`captures`, nonempty `held_out`, `protocol` (Artifact), `adapters` (exactly
`reconstruct`, `train`, `evaluate` mapped to Adapter), nonempty `metrics`, and
`primary_metric` (one metric name).

Use `group_id` for the physical site/capture family, including related scene
variants, to prevent leakage under renamed scenario IDs. Training and held-out
IDs, groups, source URIs, and source hashes must be disjoint. Held-out groups
must also be absent from the baseline's declared training history. Use a protocol
artifact that fixes robot identity, units/frames, sensors, action interface,
simulator/task version, goals, timestep, episode completion rules, native rollout
format, metric computation, training settings, and evaluation configuration.
Native episode horizons belong in this operator protocol; the workflow invents
no job, cost, or time budget.

## Reports and machine-readable decision

Every adapter report contains `bundle_sha256`, the exact `adapter`, `attempt_id`,
and `inputs_sha256` copied from the request. Only measured adapter results may
populate reports; successful schema validation is not a workload result.

| Report | Additional required fields |
| --- | --- |
| `npa.field-failure.reconstruction.v1` | `scenes`: one entry per capture with `scenario_id`, `group_id`, `capture_sha256`, and `asset`; `evidence` (native conversion/calibration log Artifact) |
| `npa.field-failure.training.v1` | `reconstruction_sha256`, `initial_checkpoint_sha256`, `candidate` (Policy), exact unique `training_scenario_ids`, `training_group_ids`, `training_scene_sha256`, and `evidence` (native learning log Artifact) |
| `npa.field-failure.evaluation.v1` | `protocol_sha256`, `policy`, nonempty `episodes` |

Each episode has `scenario_id`, `scene_sha256`, `seed`, `status: completed`,
`termination` (`success`, `failure`, or `timeout`), positive integer `steps`,
`metrics` (exactly every sealed metric), and `evidence` (Artifact).
A failed or timed-out navigation episode is completed evidence, not a missing
row; a crashed or unfinished evaluator is invalid evidence.

Episode evidence is a JSON object with `schema_version:
npa.field-failure.episode.v1`, all episode fields except `evidence`, plus
`bundle_sha256`, `checkpoint_sha256`, `protocol_sha256`, and `trajectory`
(Artifact containing the native measured rollout). Summary values must equal
these hash-bound measurements, and underlying trajectory bytes are verified.
The adapter owns decoding native trajectories and deriving metrics correctly.
Every episode must have distinct wrapper and underlying trajectory hashes,
including across policy arms. This is intentionally strict: byte-identical
trajectories are rejected even when honest deterministic policies produce them.
Such a cohort cannot support promotion through this contract; use independent
measured evidence and review the rejected cohort. Adding policy/seed wrappers or
copying raw bytes to different object paths does not establish independence.
Uniqueness is an evidence-integrity check, not proof that metrics are truthful.

Durable records under `s3://<bucket>/field-failure/<run-id>/` are
`validated.json`, `reconstruction.json`, `training.json`,
`baseline-evaluation.json`, `candidate-evaluation.json`, and `decision.json`.
Adapter artifacts use stage-private `attempt-<id>/` subdirectories.
Claims are under `claims/`. Every report reference is verified before use.

The final `npa.field-failure.decision.v1` includes both policy identities,
input/report hashes, evaluator identity, episode counts, direction-adjusted
paired metric deltas, means, regression rows, and `recommendation`. Promotion
requires a strictly positive primary mean improvement at least its sealed
`minimum_improvement`, and no individual scenario/seed/metric degradation beyond
that metric's `maximum_regression`. Secondary metrics are regression guards;
their `minimum_improvement` is informational unless selected as the primary.
All comparisons use the predeclared bounds and tolerances, with no post-hoc
threshold selection. This is a deterministic paired comparison, not a statistical
significance claim or an assertion of safety outside the held-out cohort.

Valid evidence yields `status: verified_comparison` and either `promote` or
`retain_baseline`; `decision` is compatible with the standard runtime reader.
Invalid or missing comparison evidence writes `status: invalid_evidence` with
`promote_checkpoint: false`, then raises so the workflow fails. All decisions
set `deployment_authorized: false`. If storage itself is unavailable, writing a
rejection can also fail; no successful decision is fabricated.

## Resume and concurrency

A completed stage is reused only after its current inputs, permanent claim,
identity, output hashes, and report contents are reverified. The adapter is not
called again. Provider-conditional S3 writes grant one permanent attempt claim;
concurrent invocations cannot both enter the adapter. Each attempt receives a
unique output generation. A claim is never expired, deleted, or taken over.
An interrupted attempt without a completed report fails closed on retry and
requires a fresh workflow run ID. This deliberately avoids an old writer
resuming into another attempt's output. A changed claim blocks late publication.

Use fresh run IDs after changed bundles, images, protocols, or failed partial
attempts. The normal workflow runtime may resume completed waves; it does not
recover partially executed operator training. Do not manually edit completed
records or claims. Retain the whole prefix for review; apply your existing
storage retention policy when finished. Cancel outstanding workflow jobs through
the standard runtime before any operator-led teardown.

## Plan and execute

From an installed checkout, planning is local and does not verify adapter/data
access:

```bash
npa workbench workflow validate-spec workflows/testing/field-failure-policy-improvement.yaml --json
npa workbench workflow plan-spec workflows/testing/field-failure-policy-improvement.yaml --run-id preview --json
```

The spec's `bucket`, `bundle_uri`, empty digest/entrypoints, and
`operator-required` images are deliberate unresolved inputs. Set these fields in
an operator-owned copy or with standard `--var` overrides. The required keys are
`bucket`, `bundle_uri`, `bundle_sha256`, `reconstruct_adapter`, `train_adapter`,
`evaluate_adapter`, `reconstruction_image`, `training_image`, `evaluation_image`.
`prefix` defaults to `field-failure/{{run.id}}` and must end in the exact run ID.
`output_root` and `decision_uri` derive from it. No secret belongs in the bundle,
spec, entrypoint, or URI. Forward workload credentials via `--secret-env`.

For a prepared operator copy:

```bash
npa workbench workflow submit "$FIELD_FAILURE_SPEC" \
  --run-id "$RUN_ID" --runtime --stage-src --infra "$NPA_INFRA" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The submit matrix registers an executing GPU runtime case, excluded from daily
rotation because inputs cannot be synthesized honestly. Its generic seeder
points to the dedicated opt-in test. To run that complete path, store exactly
the nine required config keys above in private JSON and set:

```bash
export NPA_INTEGRATION_E2E=1
export NPA_FIELD_FAILURE_LIVE=1
export NPA_FIELD_FAILURE_LIVE_CONFIG="$FIELD_FAILURE_CONFIG"
export NPA_FIELD_FAILURE_INFRA="$NPA_INFRA"
npa/.venv/bin/python -m pytest npa/tests/e2e/test_field_failure_policy_live_e2e.py -q
```

`NPA_FIELD_FAILURE_LIVE_CONFIG` is a local private configuration file, not an
artifact URI; `NPA_FIELD_FAILURE_INFRA` is the authorized target. All four
variables are required and have no live default. Existing AWS credentials and,
when present, `NGC_API_KEY`/`HF_TOKEN` are forwarded by name. The test submits the
actual six-state workflow, verifies real decision evidence, and re-runs the
comparison to prove durable reuse. A retain-baseline result can be a valid live
acceptance outcome; acceptance does not guarantee improvement. Local contract
tests use explicitly synthetic fixtures and never constitute GPU acceptance.
