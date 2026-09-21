# Paired NNX stock-anchor helper

This directory contains the reusable, publication-safe pieces of a prospective
stock-anchored BEHAVIOR training experiment:

- `anchored_objective.py` defines the backend-neutral demonstration and teacher
  consistency objective.
- `native_nnx_anchor.py` captures an independent parent action state, evaluates
  student and parent in one mapped NNX call with shared stochastic inputs, and
  reduces those outputs through the shared objective.
- `native_trainer.py` provides the executable filtered value-and-gradient,
  optimizer-update, frozen-state, state-inventory, and parent-sharding wiring.
- `anchor_preflight.py` runs the compiled initialization gates before an
  optimizer update. The paired topology is the gate; the separate-call result
  remains a diagnostic.
- `selection.py` validates a complete, sample-aligned stock-plus-checkpoint loss
  panel and applies the frozen no-drift and late-proxy selection rule.
- `export_qualification.py` validates an immutable serving export, finite
  actions, the documented task-specific `-inf` stage mask, and fixed-output
  byte identities without importing campaign infrastructure. It also stores
  raw diagnostic arrays and typed JAX keys with exact byte manifests so an
  independent loader can distinguish observation reconstruction drift from
  output drift.
- `recipe.json` records the frozen experiment proposal.

The operative integration order is:

1. initialize the native train state and install the explicit FP32 correlation;
2. initialize `NativeTrainer` with the exact 23-leaf action filter and the
   native detailed-loss seam that returns `_flow_velocity`;
3. run `compiled_initialization_preflight` on a real batch;
4. configure `nnx.Optimizer` with the same filter and call
   `NativeTrainer.update`;
5. use `preserve_frozen_state` for the export EMA and validate its changed
   leaves against the same allowlist.

Differentiate and donate only the student. Keep the captured parent outside
optimizer, EMA, donation, and checkpoint state. Use
`NativeTrainer.parent_input_sharding` when the native compiled train step passes
the parent state as a distinct argument.

The helper validates the real `nnx.Intermediate` correlation type and checks
ordered action paths, variable types, shapes, and dtypes before the mapped
forward. It does not select the trainable partition or modify a third-party
model. Use the existing 23-leaf filter from the matched-training implementation.

See [the training note](../../../docs/workbench/behavior-stock-anchored-training.md)
for the objective, recipe, integration order, tests, and current limits. The
repository now publishes and tests the reusable native trainer wiring, including
two updates across the real 23-leaf partition. A concrete third-party model must
still supply its initializer, detailed-loss seam, data loader, schedule, and
canonical correlation artifact. A private pinned-runtime trial completed the
recipe and improved its frozen offline held-out action-loss measure on all three
training tasks. Historical native raw outputs were not retained, so export
parity and simulator evaluation remain pending.
