# Real NuRec workflow experiment

[Aggregate results JSON](../../npa/examples/specialists/workflows/results/2026-09-23-nurec.json)

The agents operated real Workbench workflows, but **this experiment did not
demonstrate that Astra + Token Factory outperforms Astra alone**. Both original
runners ended with **0/4 agent-verified completions** after control-plane
failures. The hybrid produced four complete artifact sets subsequently verified
by an operator. The baseline produced no fully verified artifact set, so there
is no paired quality result or paired verified-video comparison.

This measures execution reliability and orchestration overhead for
parameterizing supplied workflows; it does not measure autonomous workflow
design or source-code bug repair.

| Measure | Astra + Token Factory | Astra alone |
|---|---:|---:|
| Original agent-verified workflows | 0/4 | 0/4 |
| Independently verified artifact sets | 4/4, operator verification after the run | 0/4 |
| Recorded input / output tokens | 16,032,854 / 34,602 | 12,059,444 / 10,217 |
| Recorded model-price equivalent | $20.49–$35.10 | $14.30–$28.35 |
| Recorded usage completeness | Incomplete | Complete provider counters |
| Coordinator elapsed time | 72.75 minutes | 52.26 minutes |
| Full autonomous completion time | Unknown: completion requirement unmet | Unknown: completion requirement unmet |
| Attributed GPU allocation | 3.01430–3.01432 GPU-hours, four admitted scenes | 2.30969–2.30971 GPU-hours, three admitted scenes |

The shorter failed baseline lifecycle is not a speed win. The token-price
ranges do not establish end-to-end savings.

## What the models actually did

Both arms ran [frozen Workbench commit `0231005`](https://github.com/nebius/nebius-physical-ai/commit/0231005c76226e35125ceae96859d56346c88a5e). The pinned viewer used an older public build. Later observation-recovery changes were not exercised by the original trials.

The workload used NVIDIA's public [PhysicalAI-NuRec-PPISP dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with the four `auto` scenes. The native recipe used one epoch with 30,000 training samples and one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU per workflow. Renders used `camera1` and `camera2`, with rig Y translations of +0.25 metres for Struktur28/Toro and −0.25 metres for Huerstholz/Valiant. The fixed model IDs were `gpt-6-astra` at `medium` reasoning effort, `zai-org/GLM-5.3` and `deepseek-ai/DeepSeek-V4-Pro-0813`. The controller was self-hosted; routing was explicit and Jev was not used.

The hybrid used an Astra supervisor with GLM-5.3 and DeepSeek-V4-Pro specialists
through explicit Token Factory endpoints. Each specialist had a separate
workspace and fixed Workbench operation grants. They edited workflow
parameters, validated and planned the result, submitted it through Workbench,
and monitored durable receipts. Astra could inspect progress and take over an
explicit escalation. The Astra-only arm used the same supervisor model,
reasoning effort and direct workflow tools.

| Scene | Hybrid specialist | Changes made in both arms |
|---|---|---|
| Struktur28 | GLM-5.3 | None; the supplied template already matched |
| Toro | GLM-5.3 | Set the requested scene |
| Huerstholz | DeepSeek-V4-Pro | Set the scene and rig Y offset to −0.25 metres |
| Valiant | DeepSeek-V4-Pro | Set the scene and rig Y offset to −0.25 metres |

Each arm changed five parameters across three workflow files; `variant: auto`
remained unchanged. The specialists used five edit calls and Astra used three.
All first validation, planning and local submission-acceptance checks passed.
Local acceptance did not guarantee native launch or completion. Neither arm
needed to correct a rejected parameter edit.

The hybrid's six-stage pipelines performed environment checks, fetched NCore
captures, trained real neural reconstructions, rendered offset views, produced
Rerun recordings and finalized outputs. Operator verification confirmed four
USDZ reconstructions, 351 images, eight videos containing 351 decoded frames,
and 189 Rerun image rows bound to the requested offsets.

| Hybrid scene | Native test PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Struktur28 | 24.7556 | 0.7334 | 0.4724 |
| Huerstholz | 19.8799 | 0.4722 | 0.5894 |
| Toro | 23.9722 | 0.6679 | 0.5503 |
| Valiant | 24.9919 | 0.7014 | 0.4380 |

These are native reconstruction test metrics, not ground-truth quality scores
for the offset videos. Novel outputs were checked through provenance, exact
offsets, full decoding and Rerun image-byte binding. No quality threshold or
ranking was introduced after the run. Operator visual review noted blur and
ghosting in parts of the hybrid renders; valid media is not a quality-win claim.

## Why the trials did not complete

Three hybrid workers stopped after local SQLite I/O failures interrupted
observation receipts; the underlying OS cause is unknown. Toro attempted
verification, encountered grader incompatibilities, then escalated through a
truncated DeepSeek fallback to Astra. Original journals preserve these failures.
Three disclosed grader amendments corrected native status interpretation,
temporary artifact staging paths and compatibility with valid checkpoint and
older viewer formats. Operator verification after those corrections established
the four artifact passes, without changing the original agent score.

The baseline's fourth scene was blocked before native launch because operator
configuration lacked the replacement cluster's identity metadata. That metadata
was repaired, but the original failed launch was preserved without a retry.
The other three native drivers later failed a SkyPilot API credential-fingerprint
guard after a shared authentication-cache change. Their executing Kubernetes
identity still matched the initial identity; the changed cache entry belonged
to another identity. The cache writer is unknown. This is distinct from model
inference caching and is not attributed to the model.

The three admitted baseline scenes did produce reconstruction and render
outputs. Their retained runtime records are failed, and the pipelines never
passed complete artifact verification. Those partial outputs remain diagnostic
evidence and are not scored as successful workflows or paired quality results.

Both coordinators stayed within their granted tool scope and reported blocked
results. Workbench initialization, preflight, recovery and verifier code fixes
were authored by the development/operator team, not by the measured GLM,
DeepSeek or Astra trial agents.

## Cost and interpretation

Hybrid recorded price equivalents are Astra $14.97–$29.58, GLM $2.01 and
DeepSeek $3.50. Rejected and fallback responses remain counted, but interrupted
journals leave the hybrid's full usage and model cost unknown. Astra's aggregate
counters do not expose every request's pricing tier, so both arms retain a
range. Token Factory input uses the full input rate without an assumed cache
discount. These are trial recorded model-price equivalents, not invoices or an
end-to-end experiment/customer bill. Rates were recorded on 23 September 2026
from [OpenAI Astra model pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Token Factory model information](https://tokenfactory.nebius.com/api/public/models_info).
[Nebius compute prices](https://nebius.com/prices) were consulted, but shared-host
GPU allocations were not converted into an invented incremental invoice.

Operator and development-agent inference outside the controlled runners,
including coding and debugging, is unmetered. Human effort, whole-host billing,
CPU/controller use, storage, network and other setup costs are also excluded.
Three failed pilots are retained separately: recorded model-price equivalents
$1.08–$2.12, $1.45–$2.86 and $13.72–$24.09; the last has incomplete usage.
Four setup warm probes used 1,304.43–1,309.44 requested GPU-seconds, separately
from the arms. Requested GPU intervals on shared existing hosts are not kernel
time, incremental billing or demonstrated savings.

The hybrid ran four workflows concurrently; the baseline setup failure left
three. The baseline also received the corrected grader and amended placement
for one scene. Unequal concurrency, feedback, placement, cache warmth,
shared-host contention and fixed run order rule out a clean causal comparison.

A future controller should wait for durable state changes without repeated
model calls and wake models for actionable decisions. This run recorded 234
specialist waits plus 78 supervisor waits and 58 supervisor status inspections.
The new opt-in interrupted-observation recovery applies only to calls classified
before execution. It cannot recover this trial's unclassified calls, fix the
OS I/O cause or change historical completion counts. Any evaluation of that
improvement requires a separately declared experiment.

All 28 admitted workload GPU intervals are closed, with no recorded observer
gaps or unattributed owned GPU pods. The blocked fourth baseline task is not a
successful zero-cost task; the totals cover unequal workloads. Post-run cleanup
removed the three admitted baseline controllers and stopped their invalidated
and cleanup API generations; unrelated workloads were preserved. No workflow
was resumed and no new payload was launched. All 22 original sealed files remain
byte-identical. The never-launched task had no runtime resource to clean up,
and its native cancellation refusal remains recorded.

## Incomplete baseline diagnostics — excluded from accepted quality results

Retained evidence shows exact name/size/SHA256 equality for all 12 original
input shards across the three admitted scenes. The failed pipelines also
contain native reconstruction metrics and partial render outputs:

| Baseline scene | Native test PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Huerstholz | 19.8947 | 0.4715 | 0.5908 |
| Toro | 23.9824 | 0.6677 | 0.5492 |
| Valiant | 24.9144 | 0.7004 | 0.4391 |

All three runtime records are failed, and full artifact verification did not
pass. These are diagnosis-only measurements. They do not populate accepted
quality deltas, establish parity or produce a paired verified-video result.
The original baseline completion count remains 0/4.

## Operator intervention accounting

Counts below use each original runner's actual start and end, not only ledger phase labels. Shared grader amendments can occur after one arm and before the other. Read-only observations are excluded from these human-event counts.

| Arm | Before runner | During original runner | After runner | After-run cleanup subset |
|---|---:|---:|---:|---:|
| Astra + Token Factory | 0 | 1 | 3 | 0 |
| Astra alone | 5 | 1 | 1 | 1 |

The cleanup subset is already included in the after-run count. Post-run grader corrections, operator artifact verification and cleanup do not complete original agent tasks. These event counts do not measure all operator/development effort.
