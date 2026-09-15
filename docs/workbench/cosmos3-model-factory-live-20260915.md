# Cosmos 3 data factory on reserved RTX capacity

[Workbench](README.md) · [Model factory plan](../architecture/cosmos3-model-factory.md)

**Live exercise: 2026-09-15 UTC.** Runtime code baseline: `d743f1853`.

This exercise runs the existing `workflows/main/paidf-cosmos3.yaml` in a newly
provisioned project. It tests the video-production foundation for the model
factory plan: source conditioning, generation, evaluation, and quality-driven
refinement. Both generation passes produced videos, but neither batch met the
shipped quality criteria. The second pass received a durable `rejected`
disposition. This exercise does not train a Cosmos model or evaluate a robot policy.

## Infrastructure and software

- One reserved RTX PRO 6000 Blackwell Server Edition GPU and one CPU node in
  Nebius Managed Kubernetes, with project-scoped Object Storage.
- Provider inventory confirmed the GPU's reservation binding. Both nodes became
  Ready; the CUDA smoke succeeded and GPU health remained stable throughout the
  default stability window.
- Project, storage, Hugging Face, and Token Factory preflights passed. Exact
  Cosmos3-Nano and Cosmos-Guardrail1 access checks passed. All five workflow
  images were pullable and bootstrap-compatible.
- A Linux Python 3.12 operator environment ran SkyPilot 0.12.2. NPA's isolated
  controller uses Linux process inspection; direct macOS discovery failed before
  workload launch. After moving to Linux, NPA regenerated the kubeconfig and
  verified the exact project/cluster identity.
- Generation used `npa-cosmos3:1.2.2-cu130-r7`, resolved to image digest
  `sha256:d8e1fe370f75e5433455a221b70ae6211c30369255a3bb111d03e5c07240e010`.
  Its native framework revision is
  [`5e67049cd94acb667786f1e6dd0dab821cb90c97`](https://github.com/NVIDIA/cosmos-framework/tree/5e67049cd94acb667786f1e6dd0dab821cb90c97).
  Worker logs recorded Nano snapshot `7a312c868bcce8e40b3eb40861300a9d0ba3fde1`.
- The standard submit path staged the checkout's NPA source automatically.
  Credentials, model caches, resource identities, and raw operational evidence
  remain outside Git.

## Input and settings

The run used a freshly downloaded copy of the public RoboPro MP4 in the setup
guide's [local-video example](../../workflows/guides/paidf-cosmos3.md#r3b-run-the-full-pipeline-from-a-local-mp4).
Both run input/output prefixes were empty before submission. The original's
SHA-256 matched
`caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198`.

| Media | Decoded frames | Frame rate | Duration | Resolution |
| --- | ---: | ---: | ---: | --- |
| Retained original | 169 | 50 fps | 3.38 s | 640×480 |
| Prepared reference | 81 | 24 fps | 3.375 s | 832×480 |

The canonical settings were retained: two variants, sequential generation,
native edge conditioning, guardrails enabled, and no source/model pixel blending.
The first pass used seeds 17 and 18, 24 sampling steps, and guidance 5.0. The
shipped thresholds were 0.2 for the aggregate and required hallucination scores,
and 0.25 for attribute verification,
with the `all-variants` batch policy and the existing two-pass refinement setting.
These are exploratory criteria; passing them does not qualify a training corpus.
Higher values are better for the evaluator's hallucination quality score.

## Observed quality feedback

The first pass published two videos totaling 12,697,120 bytes. Both passed full
decoding and source-relative frame/timestamp alignment. The evaluator completed
all eight attribute questions without API errors.

| First-pass variant | Evaluator score | Required hallucination score | Attribute checks passed | Clip disposition |
| --- | ---: | ---: | ---: | --- |
| Seed 17 | 0.304449 | 0.608897 | 0/4 | Rejected |
| Seed 18 | 0.421188 | 0.342376 | 2/4 | Passed |

The aggregate score was 0.362819. The batch was rejected because every variant
must pass. The recorded decision was `loop_back`; the runtime launched a fresh
generation pass instead of promoting the first outputs. First-pass videos,
metadata, and the evaluator report were retained separately for comparison.

The failed attributes concerned background, color grade, lighting, and surface
finish. Midpoint visual previews retained the main scene layout; one variant
introduced strong amber lighting and color changes. Still-image inspection does
not establish motion or contact fidelity. The attribute report and required
temporal-alignment checks answer separate questions.

The configured retry used seeds **1017 and 1018**, **28 steps**, and **guidance
4.5**. It published another two videos totaling **15,427,093 bytes**, each with
81 decoded frames at 24 fps, lasting 3.375 seconds at 832×480. The second
evaluator report's video hashes matched those newly published files. Again,
all eight attribute questions completed without API errors.

| Retry variant | Evaluator score | Required hallucination score | Attribute checks passed | Clip disposition |
| --- | ---: | ---: | ---: | --- |
| Seed 1017 | 0.336664 | **0.173328** | 2/4 | Rejected |
| Seed 1018 | 0.376435 | 0.252870 | 2/4 | Passed |

The retry's aggregate score was **0.356550**. Appearance verification improved
for the first variant, but its required hallucination score fell below 0.2.
Consequently, the `all-variants` batch policy still rejected the batch. The
advisory temporal-consistency metric also reported low scores; it was not the
required check that caused rejection. Decoded timestamp alignment verifies
media correspondence, not physical plausibility.

## Runtime recovery and evidence

A SkyPilot controller health probe failed with a log-stream request-not-found
error before launching `reject-quality`. The first 12 stage executions and their
durable outputs had completed. Standard `--resume-run` reused those completed stages
and launched only the terminal rejection job. No generation or evaluation work
was repeated. This exercises recovery between workflow stages, not restoration
of an in-progress Cosmos optimizer checkpoint.

The terminal job then raised the expected quality-rejection error. The final
workflow status is **`failed`**, with **`quality_status: rejected`**. This is a
completed rejection path, not a passing end-to-end dataset-production result.

A read-only audit of the rejected-run artifact contract passed. It reused the
repository's existing live media assertions and separately checked the rejection
branch and resume evidence:

- The original execution reached all 12 completed stage executions without
  replay or adoption. Their job identities, timestamps, outputs, and immutable
  identities were unchanged after resume. The final history contains 13 stage
  executions across 10 distinct states.
- Source hashes, fresh object timestamps under both run prefixes, decoded
  video timelines, native edge-control hashes, verified control loading, and
  enabled text/video guardrails matched the recorded artifacts.
- The final Rerun recording is **15,673,261 bytes**, with 17 entities. Its
  application/recording identity matched the expected contract. Both embedded
  video hashes matched the published MP4s; all **162 video-frame timestamps**
  matched the decoded timelines. `rerun rrd verify` passed.
- The workflow retained rejected-candidate evidence. It did not run augmented
  annotation, Cosmos Curator, or FiftyOne, and produced no accepted final report
  or `sim2real.rrd`. This run therefore does not validate those downstream stages.

The reference and both generation passes are retained in a synchronized review
gallery with dataset attribution and sanitized measurements. Raw Rerun records
retain full private run provenance and are stored separately from that gallery.

## Timing scope

The span from the first stage's start through the final rejection was **80 min
14 s**, including the controller-recovery gap and CPU stages. Provisioning and
cleanup are outside that interval. The GPU node remained allocated during CPU
stages; this elapsed span is not GPU sampling time.

| Stage execution | Observed elapsed time |
| --- | ---: |
| Prepare input, including cold worker setup | 11 min 21 s |
| First generation pass | 13 min 42 s |
| Retry generation pass | 13 min 6 s |
| First evaluation | 5 min 5 s |
| Retry evaluation | 4 min 22 s |

These stage timings include submission, worker setup, and artifact handling.
Four generated clips contain **13.5 seconds of video** in total; **zero clips
were promoted** under the batch policy. This short exercise exposes material
orchestration overhead. It does not establish warm inference throughput or
reserved-pool efficiency. Qualify resident generation for the same conditioning
path and measure useful accepted output across a representative workload before
sizing a factory.

## Reproduce and inspect

Follow the [setup and run guide](../../workflows/guides/paidf-cosmos3.md), including
its exact GPU discovery, project-scoped preflight, fresh run ID, separate SkyPilot
directory, local MP4 submission, and saved pre-submission UTC timestamp. Execute
through the standard runtime using actual quality decisions. The guide's
`--assume-decision` example is a planning preview only.

When moving to another operator environment, regenerate its saved kubeconfig
through `npa cluster kubeconfig`; copying state containing the original host's
paths does not establish a valid execution identity.

For an accepted complete run, the guide's read-only
[`test_paidf_cosmos3_mp4_live.py`](../../npa/tests/e2e/test_paidf_cosmos3_mp4_live.py)
checks fresh object timestamps, stage execution without replay/adoption, source
identity, decoded timelines, curation, and both Rerun recording identities.
Require a passed test, rather than a skip. A quality-rejected run has a different
terminal artifact contract; retain its rejection and review evidence.

Use the standard [cancel → controller → cluster cleanup](../teardown.md) and
preserve the output storage needed for review. Exact cleanup identities belong
in private operational evidence.

This exercise completed that cleanup sequence. Controller removal was verified;
the final provider inventory contained **zero clusters, compute instances,
compute disks, or filesystems**. The dedicated project, its default network,
and output storage were retained. The quality recording remained readable after
compute teardown.

## Implications for the model factory

This run exercises an actual quality-feedback path. Successful generation and
timeline alignment do not imply that requested appearance attributes are present.
Changing seeds, steps, and guidance improved some appearance checks while the
required hallucination score worsened. Scene preservation and appearance targets
need joint qualification before data is admitted to a training corpus. Retain
attempt-specific videos and evaluator reports so improvement claims can be
checked against the exact bytes from each pass.

The retry changed seeds, steps, and guidance together, so it does not isolate
which change caused the quality difference. The next qualification should hold
source clips and seeds fixed when comparing settings, then verify the selected
configuration on held-out clips with human review of motion and contacts.

The remaining factory work is the native training environment, a validated
training-data layout, checkpoint save/resume/export, matching policy evaluation,
and failure-driven retraining described in the [implementation plan](../architecture/cosmos3-model-factory.md).
Cold cluster, image, and model startup in this exercise are not a throughput or
reserved-pool efficiency benchmark. Measure cold and warm operation separately.

## Dataset attribution

Source: [RoboPro dataset card](https://huggingface.co/datasets/Hoshipu/RoboPro/blob/90ec789bf4018eb9c0f75da9f69aab5c185f0fd0/README.md),
revision `90ec789bf4018eb9c0f75da9f69aab5c185f0fd0`, declared
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The downloaded card and
its author citation accompany the retained media. Changes comprise the normalized
reference and model-generated appearance variants; retain this attribution when
sharing derivatives.
