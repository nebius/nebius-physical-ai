# Task: generate, grade and curate a Cosmos3 campaign

Read the [common benchmark contract](README.md). Planning estimate: **3–6 hours
per arm**, subject to GPU calibration. This is the Workbench-specific Cosmos3
VDA path, not an assertion that NVIDIA's separate IAA/EVG workflows ran.

## Agent goal

Build and operate a source-conditioned video campaign with real quality-driven
refinement, resilient publication, immutable campaign reuse and independent
dataset reconciliation. Make changes when execution exposes problems. Deliver
reviewable videos and an accepted/rejected/review partition whose contents can
be traced to source, prompt, seed, model and actual evaluator evidence.

Use [the PAIDF skill](../../../../skills/workflows/physical-ai-data-factory/SKILL.md)
and [canonical Cosmos3 workflow](../../../../workflows/main/paidf-cosmos3.yaml).
Generation must invoke Cosmos3-Nano video2video; grading, Cosmos Curator,
FiftyOne and Rerun must execute their real implementations. Pin actual images
and model revisions after access preflight. A generated manifest is insufficient.

## Frozen workload

Select eight rights-cleared source clips, each with useful visible manipulation
or motion and an identified camera/timeline. Freeze exact bytes and durations.
Use three coherent appearance profiles, with all four supported fields:
`lighting`, `background`, `color_grade`, `surface_finish`. Cross each source and
profile with generation seeds `17` and `29`: **48 initial variants**.

Freeze prompts, captioning instructions, native conditioning parameters and
evaluator settings before measurement. Retain and charge the actual captions
produced by each arm. Both arms receive identical inputs and the same
quality requirements. Use separate ranking frames during refinement and held-out
frames for final acceptance. Fix those partitions before either arm, without
exposing held-out grader inputs to the workers.

Preserve source duration and timing through the supported normalization path.
Record all additional variants produced by legitimate refinement; 48 is the
initial matrix, not an attempt limit. This task adds no time/cost/job cap.
Keep any existing workflow iteration setting explicit and identical across arms;
exhaustion is a terminal quality outcome, not permission to relabel a rejection.

## Engineering milestones

1. Implement the campaign-level mapping from source/profile/seed identities to
   actual Workbench runs, immutable generations and durable status. Execute
   input preparation and confirm source normalization/caption provenance.
2. Generate and evaluate the complete matrix. Diagnose failed stages and quality
   gates; use recorded feedback for supported parameter changes. Preserve
   the failing generation and its configuration. Do not weaken thresholds or
   quietly replace hard source clips to obtain acceptance.
3. Recover from the two controlled faults below and repair the relevant
   Workbench boundary. Integrate specialist changes and rerun affected stages
   with final-source receipts and unchanged evaluation rules.
4. Execute real curation and visualization. Freeze source, generation and
   evaluation evidence into a base campaign, then run a second curation/viewer
   derivative from that immutable base without new generation or evaluation.
   Deliver both derivative receipts and independently reconcile their outputs.

Use [the existing campaign contracts](../../../../docs/workbench/guides/paidf-campaign-reuse.md)
for base and derivative identities. That module validates envelopes; it is not
an executor. Implement and test the missing orchestration rather than claiming
that calling the envelope builder executed a campaign.

## Controlled recovery exercises

After one generation completes, the harness interrupts its campaign controller
before local acknowledgement. The recovered campaign must reconcile provider
state and real output bytes, preserving the successful generation without a
duplicate paid request.

In a disposable curation-import copy, inject one duplicate candidate ID and one
decision whose media digest refers to another generation. The agent must reject
that import before publication, diagnose the identity problem and rebuild it
from authoritative evidence. It must not edit the immutable base or invent an
acceptance decision for a missing or rejected candidate.

Ordinary model/provider failures encountered during execution also count. A
synthetic fault is labeled as such; it is not evidence of a provider outage or
a previously unknown production defect.

## Independent acceptance

Prepare a trusted campaign verifier before the measured pair. Existing workflow
tests and envelope validation are useful but do not establish all of these:

- Every one of the 48 source/profile/seed cells has real generation/evaluation
  evidence or an explicit failed outcome. Complete task success requires the
  predeclared quality coverage floor, with per-source coverage, rather than a
  small cherry-picked accepted subset. Set numeric floors in calibration and
  freeze them before inference; without those floors the trial is not ready.
- Decode every delivered video completely. Verify source/model/prompt/seed,
  effective conditioning, duration/timestamps and source-to-output lineage.
  Independent frame checks supplement model grading. Visual appearance alone
  does not establish motion/contact fidelity or downstream policy quality.
- Recompute candidate membership from actual artifacts. Each terminal candidate
  appears exactly once as accept/reject/review. Only acceptance enters the
  curated dataset; rejected and unresolved cases remain reviewable evidence.
- Confirm that Cosmos Curator and FiftyOne actually processed the data, and
  decode the Rerun recording to match sampled media and reported decisions.
- Verify both injected failures, recovery without duplicate generation, and
  preservation of the immutable base. The second derivative must emit no
  generation or evaluation requests and no writes to base objects.
- Include every refinement and hosted evaluator call in cost and latency.
  Reusing a frozen base is application-level artifact reuse; it is not KV or
  prompt-cache evidence, and both comparison arms have the same reuse rights.

## Suggested specialist ownership

- **Generation operations:** fanout, runtime diagnosis and durable job reconciliation.
- **Quality integration:** production evaluator/refinement and curation wiring.
- **Dataset evidence:** identity reconciliation, derivative publication and viewer.

Likely source includes `npa/src/npa/workflows/data_factory_*.py`,
`npa/src/npa/workflows/paidf_campaign.py`, the Cosmos3 adapter and canonical
workflow. Freeze exact disjoint source grants after inspecting module ownership.
Give neither arm permission to edit the quality floor, immutable fixtures or
trusted graders during measurement.
