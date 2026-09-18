# BEHAVIOR matched-stage training

This implementation compares two action-only RLC fine-tunes with identical
examples and optimizer settings. The `teacher` arm conditions actions on the
released equal-time stage bin. The `replay` arm conditions them on the frozen
parent controller's stage after chronological replay of its native filter.

Read the [training guide](../../../docs/workbench/behavior-matched-training.md)
for the complete prefix, training, holdout-selection, and export commands. A
runtime must satisfy [`runtime-inputs.schema.json`](runtime-inputs.schema.json);
[`runtime-inputs.example.json`](runtime-inputs.example.json) shows the expected
path layout.

The command sequence is:

1. Run `gpu_preflight.py` on one B200.
2. Run `generate_prefix_records.py` and `build_replay_trace.py` for the frozen
   training and holdout splits.
3. Run `freeze_inputs.py` and `matched_train.py` once for each configuration.
4. Run `evaluate_holdout.py`, then `export_selected.py`, for each arm.

The repository does not include upstream source, weights, demonstrations,
credentials, or runtime caches. Native GPU validation and official rollout
evaluation remain pending; these helpers do not establish a policy gain.
