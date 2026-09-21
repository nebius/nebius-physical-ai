# Paired NNX stock-anchor helper

This directory contains the reusable, publication-safe pieces of a prospective
stock-anchored BEHAVIOR training experiment:

- `anchored_objective.py` defines the backend-neutral demonstration and teacher
  consistency objective.
- `native_nnx_anchor.py` captures an independent parent action state, evaluates
  student and parent in one mapped NNX call with shared stochastic inputs, and
  reduces those outputs through the shared objective.
- `recipe.json` records the frozen experiment proposal.

Install the explicit FP32 correlation state before calling
`capture_parent_action_state`. Pass the same native detailed-loss method used by
the student into `paired_detailed_losses`; that method must return its flow
velocity under `_flow_velocity`. Differentiate and donate only the student.
Keep the captured parent outside optimizer, EMA, donation, and checkpoint state.

The helper validates the real `nnx.Intermediate` correlation type and checks
ordered action paths, variable types, shapes, and dtypes before the mapped
forward. It does not select the trainable partition or modify a third-party
model. Use the existing 23-leaf filter from the matched-training implementation.

See [the training note](../../../docs/workbench/behavior-stock-anchored-training.md)
for the objective, recipe, integration order, tests, and current limits. The
pinned native trainer adapter, flow-velocity model seam, real 23-leaf two-update
test, and accelerator regression are not yet published, so this directory does
not claim a complete runnable trainer. A private pinned-runtime trial completed
the recipe and improved its frozen offline held-out action-loss measure on all
three training tasks; export parity and simulator evaluation remain pending.
