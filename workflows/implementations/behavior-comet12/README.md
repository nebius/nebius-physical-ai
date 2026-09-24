# Comet12 BEHAVIOR transfer adapter

This directory documents the narrow server adapter for the published Comet12
checkpoint. The adapter runs a pinned 2025 BEHAVIOR policy through the existing
2026 evaluator wire protocol. It is a transfer experiment, not an author
reproduction or a claim about challenge performance.

## Immutable inputs

- Source: [`mli0603/openpi-comet`](https://github.com/mli0603/openpi-comet) at
  `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5` (Apache-2.0).
- Checkpoint: [`sunshk/openpi_comet`](https://huggingface.co/sunshk/openpi_comet)
  folder `pi05-b1kpt12-cs32` at revision
  `a3d85eb978b58501c99f6c927a18d52ec6c1532c`.
- Model configuration: `pi05_b1k-base`, with the upstream receding-horizon
  wrapper and a 32-action queue.
- Declared checkpoint tasks: `0, 1, 6, 17, 18, 22, 30, 32, 34, 35, 40, 45`.

The checked-in checkpoint inventory records all 5,948 relative paths, byte
sizes, and upstream content identities. The adapter accepts a previously
fetched private checkpoint only after validating that entire inventory. It also
requires the exact clean source commit and verifies the critical loader files.

## Observation and action boundary

The evaluator-facing server accepts the existing messagepack WebSocket
protocol. It sends only three onboard RGB images and the 61-element R1Pro
proprioception vector to the upstream wrapper. Depth, segmentation, task-state,
instance metadata, and all other evaluator fields are excluded. Comet12 does
not use depth in this configuration. The simulation-only proprioception import
is replaced in an ephemeral source overlay with the reviewed 2026 index map;
the pinned checkout stays unchanged.

The upstream wrapper must return one finite numeric action with shape
`(1, 23)`. The server removes only that batch dimension and emits the original
23-action evaluator response. A connection start, an explicit reset message,
and connection teardown each reset the wrapper's action queue. The upstream
wrapper reset does not restore the underlying JAX policy RNG. Reproducible case
starts therefore require a fresh server process per case; a WebSocket reset is
only an in-process queue reset.

## Private runtime fetch and licensing

No checkpoint weights are stored in this repository or in this documentation.
The operator fetches the immutable Hugging Face revision using their own
authorized access and keeps the resulting cache and weights private. The model
card does not declare a checkpoint license. The checkpoint derives from
Gemma/PaliGemma, so its use remains subject to the applicable upstream terms
and the operator's recorded competition-use scope. Source licensing does not
grant permission to redistribute the weights.

## Managed serving

Select `--policy-kind comet12`, pass all four managed policy paths, and bind the
single campaign task with `--policy-task-name`. Only the native execution
variant is accepted. `comet_policy.prepare_policy` verifies the full private
Zip64 archive, its extraction, the frozen per-file inventory, source identity,
and task mapping before staging the server. The serving identity covers both
adapter modules, the inventory, and task name. Until a real GPU smoke pass
exists, this is a validated loader and protocol adapter, not a completed
evaluation result.

## Native full-training reference

The native-training reference uses the real upstream OpenPI loader, model loss,
direct JIT `train_step`, optimizer, and Orbax checkpoint APIs. It contains no
weights, datasets, normalization statistics, credentials, provider locations,
or live run receipts.

The operator supplies immutable archives and a
`npa.behavior.comet-native-training-admission.v1` document that binds the
load-bearing upstream source files, full static training reconstruction,
released non-resumable parent, data identity, and checkpoint space floor.
The admission also binds an explicit Comet data reconstruction: dataset
revision, supported task ID and name, RGB modalities, prompt behavior, and
alignment tolerance. The adapter constructs this data factory directly and
rejects unknown upstream base-config names; it never accepts OpenPI's fallback
config. The same contract supports the declared task 0, 1, and 22 readers.
`train_comet_native.py preflight` installs the same verified source overlay and
follows the training config/data entrypoint, stopping before policy
initialization. The dependent GPU stage uses the same entrypoint and arguments.
The locked scientific Python never imports the storage SDK; restore and each
milestone publication execute under the declared control Python.

The portable example is
`workflows/testing/behavior-comet-native-full-training.yaml`. Its placeholder
identities intentionally fail until an operator supplies exact authorized
inputs, image, PVC, and storage prefix. Model and data access remains the
operator's responsibility.

The checked-in `ADMISSION.example.json` documents both required free-space
floors. It is an identity template and contains no live artifact locations.
