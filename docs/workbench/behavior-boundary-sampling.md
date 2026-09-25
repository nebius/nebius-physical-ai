# BEHAVIOR gripper-boundary sampling

The boundary sampler builds matched TRAIN-only sample lists for action-expert
fine-tuning. It does not change policy inference, action values, source cadence,
episode boundaries, or DEV and REPORT selection.

## Frozen action definition

The sampler uses the released 23-coordinate B1K action layout:

- coordinate `14`: left gripper;
- coordinate `22`: right gripper;
- `-1.0`: close;
- `1.0`: open.

These signed values apply to the raw expert action before model normalization.
The pinned BEHAVIOR `v3.9.3`
[R1Pro configuration](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.3/OmniGibson/omnigibson/eval/r1pro.yaml)
uses smooth gripper controllers with the default `[-1, 1]` input range. The
[controller base](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.3/OmniGibson/omnigibson/controllers/controller_base.py)
and [multi-finger controller](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.3/OmniGibson/omnigibson/controllers/multi_finger_gripper_controller.py)
map the lower endpoint to the closed position limit and the upper endpoint to
the open position limit.

The source manifest
`e7610329a274501b1c53d033f12a5ed61900f3121f4e5ec2a236af862bf7aeb3`
binds the packed TRAIN data. One fully hashed task-1 Parquet file
`0ad51a4a16cbe48b8d122109963e905819f92000380b1e0fdf64fae61e305795`
contains 225,534 rows. Coordinate 14 contains 44,382 `-1` and 181,152 `+1`
commands; coordinate 22 contains 167,586 `-1` and 57,948 `+1` commands. These
counts describe that file rather than the full dataset. The sampler deliberately
accepts only exact endpoint commands, so zero is invalid at its input boundary;
the controller itself supports the continuous `[-1, 1]` input range.

Pinned Comet commit `4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`
[passes the 23-coordinate action through its B1K input transform unchanged](https://github.com/mli0603/openpi-comet/blob/4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5/src/openpi/policies/b1k_policy.py).
The
quantile normalization asset
`d66ed16830a98f90dde8a315058b4a0df59f5e05734c1686d8b3f66787d0a929`
has `q01=-1` and `q99=+1` for both gripper coordinates, so normalization
preserves their signs. As part of the sampler contract, inputs with another
width or any command outside exact `{-1.0, 1.0}` fail before sampling.

## Transition and sampling rules

A transition is the first frame of a changed command with three identical
source frames immediately before it and three identical source frames starting
at it. A simultaneous left/right change is ambiguous and is not a boundary.
A chunk enters a boundary pool only when it contains exactly one accepted
transition and that frame is the chunk's only raw left-or-right gripper-change
frame. A valid transition therefore cannot hide an additional rejected change,
including a simultaneous two-hand change. This also excludes overlapping
close/open or left/right changes.

Every eligible chunk consists of consecutive source frames from one episode.
Uniform control draws are generated once with NumPy PCG64. Candidate anchor
tasks reuse those exact draws. The focal task reuses a declared prefix of its
control draws and fills its separately declared close/open quotas from the
boundary pools. Repeated draws use independently permuted passes through a
pool, so small boundary pools remain deterministic without changing cadence.
One final PCG64 permutation is applied identically to the control samples,
candidate samples, and candidate-stratum labels. This shared interleaving keeps
matched positions aligned and prevents task-blocked training order. It does not
claim that each minibatch is task-balanced.

The plan must supply exact, disjoint fit and TRAIN-calibration episode
memberships for every quota task. Supplied episodes must equal the complete fit
membership; calibration episodes cannot enter either sample. The caller remains
responsible for verifying that these identities came from the reviewed split
manifest bytes and expected manifest digest before constructing the plan. The
sampler does not infer that provenance from integer IDs.

Quotas, seed, the explicitly required action horizon, memberships, and
persistence are fixed before reading any TRAIN-calibration result. The action
horizon must come from the pinned trainer binding; the sampler has no default.
Rollout outcomes and privileged simulator state are not accepted by the sampler
API.
