# Narrow terminal-evidence qualification

This is a four-control qualification of the default hosted model's terminal
decisions, not broad acceptance of its generated explanations. Earlier failed
panels and strict rationale reviews remain rejected; this report does not
replace them or qualify alternate models.

The [earlier control review](vlm-missing-terminal-control-review.md) rejected
the same default `MiniMaxAI/MiniMax-M3` model on rationale grounding under the
previous rubric. That rejection remains intact: the qualification here covers
only four terminal decisions with the ordinal-framed prompt, not broad rationale
acceptance or a retrospective pass for the earlier panel.

## Frozen execution

Production source `4cbaef3273e210b92ec0937af8b2e22575848de2` issued four real
requests to `MiniMaxAI/MiniMax-M3`, once per control, after freezing the full
request bodies. No local GPU or container was involved; provider hardware is
unattested. The change added general grounding instructions and interleaved
`Frame N` anchors. These are supplied-order ordinals, not source frame indices
or timestamps. Original source labels, pixels, order, task, default rubric,
expected labels, and the `0.8` threshold remained unchanged.

The controls were a complete seven-frame sequence, its four-frame and five-frame
prefixes, and seven uniform RGB `(128, 128, 128)` images. All 23 submitted image
occurrences matched the frozen normalized submitted bytes and original RGB
pixels; PNG encodings need not be identical. Independent read-only review
verified the request, model, response, hash, and ordinal bindings against actual
provider evidence and inspected the pixels and rationales.

| Frozen control | Score | Decision |
| --- | ---: | --- |
| Complete terminal evidence | 1.0 | Pass |
| Progress without terminal evidence | 0.0 | Fail |
| Near-terminal prefix without withdrawal | 0.0 | Fail |
| Blank evidence | 0.0 | Fail |

The complete rationale correctly identifies the cube inside the box and the
withdrawn gripper in Frames 6 and 7. Neither prefix establishes withdrawal.
These four decisions had no observed false positive or false negative. This
small, source-matched panel does not estimate general error rates, calibrate the
threshold, establish continuous execution, or certify physical safety.

## Separate rationale review: not fully met

Generated explanations still contain real inaccuracies:

- The blank control is gray, but the model calls it white.
- Calling the physical task incomplete overstates missing evidence; actual
  completion is unknown for blank images.
- Motion blur in Frame 4 does not establish a definite lift.
- A near-terminal contact claim is stronger than the partly occluded pixels
  establish.

None of these statements creates an observed gate error in this panel: the
positive terminal decision is supported independently by Frames 6 and 7, and
missing withdrawal supports rejection of both prefixes. Nevertheless, strict
all-claims-grounded acceptance is **not fully met**. Do not treat explanations as
independently reliable physical-state evidence or describe this result as
flawless grounding. The operator accepted a separately named, narrow
terminal-evidence qualification for this feature and exact model only.

## Public evidence bindings

- Frozen full-wire panel:
  `5f3d9a304467a99d46bf557c21833a3637b93d0dd19bc27b5981342df64da9f4`.
- Numeric and provider-binding summary:
  `eb85a69954ba3dc1a7f944b656c540a8cf0be50ececf27b60f4596b008d515df`.
- Distinct review and narrow disposition:
  `4c21952592758bb211b7e9d316b4fa1e7a4d0f10b5efd7da77d0aace5f24e86a`.
- Disposition's retained-result hash manifest:
  `5e6243394d707ff15b3280affa0878a8d57918936053d4df32ec58c169c05063`.

Complete requests, responses, source pixels, private mappings, and transport
identities remain in access-controlled evidence outside the repository. These
bindings name the actual executed source, not a later composition commit.
