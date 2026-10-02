# Task-1 semantic monitor data boundary

This package prepares a TRAIN-only dataset for a future completion and grasp
persistence monitor for `picking_up_trash`. It does not train a model or change
RLC execution.

`TrainingTraceFrame` accepts exact task-1 TRAIN membership and offline simulator
predicates for the three cans. `collect_episode_labels` converts those traces to
persistent grasp, transport, release, placement, and failure-boundary labels.
Inputs naming DEV, REPORT, holdout, validation, or evaluation sources are
rejected. Episode splits are deterministic and episode-disjoint.

The caller must verify the frozen TRAIN membership and observation bytes before
calling the labeler. The labeler validates supplied records and digests; it cannot
prove that a caller's split declaration or observation digest is truthful.

`grasp_established` describes a currently persistent grasp, not a historical
latch. Any release resets it, and a later re-grasp must satisfy persistence
again. Placement means persistently contained while released; re-grasping the
can invalidates placement even while the can remains geometrically inside.

`PromotionRequest` is the proposed runtime integration boundary. It accepts only
the official RGB cameras, optional depth cameras, finite 61-value proprioception,
and internal current/proposed stage integers. Unknown fields are rejected, so
privileged TRAIN predicates cannot enter monitor inference. `PromotionMonitor`
returns a recommendation and has no policy mutation method. A later reviewed
integration may decide how to apply that recommendation and refresh action
queues.

The interface is explicitly unbatched and represents one simulator environment.
Official evaluator-wire arrays include a leading batch dimension, so a future
integration must add and validate an N=1 extraction adapter before calling this
interface. Batched arrays are rejected today. Monitor scores are only required
to be finite and within `[0, 1]`; empirical calibration is a future TRAIN-only
acceptance requirement and is not claimed by this package.

Real collection still needs frozen task-1 TRAIN episode membership, observation
bytes, native stage proposals, and simulator-derived predicates for proximity,
grasp/contact, release attempts, and `inside(can, ashcan)`. Boundary requirements
that bind the six RLC stages to can-specific semantic events must be frozen before
collection.
