# BEHAVIOR matched-stage training

This implementation compares two action-only RLC fine-tunes with identical
examples and optimizer settings. The `teacher` arm conditions actions on the
released equal-time stage bin. The `replay` arm conditions them on the frozen
parent controller's stage after chronological replay of its native filter.
Replay decisions use the parent's canonical native inference path with batch
size one and one denoising step; the sampled action is discarded. Direct
15-way classifier logits are retained only as diagnostics. `PanelDataset`
wraps the pinned LeRobot v3 selected reader with validated episode boundaries.
Ordered CPU decoding keeps observations, actions, and batch order unchanged.
Optional complete-episode shards let two GPUs generate the shared prefix traces;
`merge_prefix_records.py` verifies their membership before restoring canonical
order. The optimizer also preserves frozen moving-average parameters explicitly.

Read the [training guide](../../../docs/workbench/behavior-matched-training.md)
for the complete prefix, training, holdout-selection, and export commands. A
runtime must satisfy [`runtime-inputs.schema.json`](runtime-inputs.schema.json);
[`runtime-inputs.example.json`](runtime-inputs.example.json) shows the expected
path layout.

The training guide also documents opt-in final-stage voting and adaptive
short-chunk execution. Native execution remains the default, and these
execution variants carry no aggregate-gain claim.

The command sequence is:

1. Run `gpu_preflight.py` on one B200.
2. Run `generate_prefix_records.py` and `build_replay_trace.py` for the frozen
   training and holdout splits.
3. Run `freeze_inputs.py` and `matched_train.py` once for each configuration.
4. Run `evaluate_holdout.py`, then `export_selected.py`, for each arm.

The repository does not include upstream source, weights, demonstrations,
credentials, or runtime caches. Both matched training arms and their holdout
selection completed on B200 GPUs. The selected replay checkpoint passed serving
consistency after restoring its receipt-bound native BF16 correlation
intermediate before policy/JIT construction. Candidate rollout evaluation
remains pending; these helpers do not establish a policy gain. The current
freeze filter excludes non-parameter statistics from the native initializer's
BF16 weight conversion. Completed checkpoints retain the earlier filter's
behavior; see the training guide before reusing them.
