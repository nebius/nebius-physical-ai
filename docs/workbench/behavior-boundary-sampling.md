# BEHAVIOR gripper-boundary sampling

The boundary sampler builds matched TRAIN-only sample lists for action-expert
fine-tuning. It does not change policy inference, action values, source cadence,
episode boundaries, or DEV and REPORT selection.

## Frozen action definition

The sampler uses the released 23-coordinate B1K action layout:

- coordinate `14`: left gripper;
- coordinate `22`: right gripper;
- `0.0`: fully open;
- `1.0`: fully closed.

These values apply to the raw expert action before model normalization. They
come from the pinned Comet source at commit
`4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5`. Its B1K transform
(`a82c15dece0242a94cb25d94b7fe4ed04c1d0fcfc508ed6d9d5e28b0f32a075e`)
passes the dataset's 23-coordinate action through unchanged; its normalization
contract (`62e4a39ea7fcdae53b4466942db348c3e38fc016b7f7a5273cafed649ebe614b`)
defines `0.0` as open and `1.0` as closed. The independently pinned RLC source at commit
`ca556f74a455cef7987a2be4537b5ac85cc56dd7` names coordinates 14 and 22 in
both its per-coordinate loss
(`55faeee1c5e3f3c47f1dd02ca88e66270cb2f1bffb36b132fe90fa40855d2585`)
and correction-rule
(`943ac883b65527d7d1a63bafbb8ee84f19bde8de78785565c3ee8e118af4b857`)
implementations. Inputs with another width or a nonbinary gripper command fail
before sampling.

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
