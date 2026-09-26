# Agents repairing real NuRec stages

[Aggregate results JSON](../../npa/examples/specialists/workflows/results/2026-09-23-nurec-repair.json)

Both Astra alone and Astra supervising Token Factory specialists repaired and
verified **2/2 supplied Workbench defects**. The hybrid took 10.54 minutes versus
10.96 minutes, but its model-price equivalent was higher: **$6.14–$8.90 versus
$2.93–$5.77**. This single pair demonstrates source repair and recovery on real
stages; it does not demonstrate savings, general parity or superior output quality.
The earlier [workflow execution experiment](specialists-nurec-experiment.md)
remains a separate, unchanged result.

| Measure | Astra alone | Astra + Token Factory |
|---|---:|---:|
| Source repairs verified before coordinator exit | 2/2 | 2/2 |
| Agent/tool lifecycle | 657.61 s | 632.10 s |
| Recorded input / output tokens | 1,834,534 / 3,600 | 4,152,050 / 29,291 |
| Model-price equivalent | $2.93–$5.77 | $6.14–$8.90 |
| Requested GPU allocation | 262.000–262.441 s | 260.000–260.289 s |
| Native render launches | 1 | 1 |
| Operator intervention during measured execution | 0 | 0 |

## What the agents repaired

**Render isolation and publication.** An existing render directory contained old
media. A native invocation produced 45 new frames, but the original wrapper
reported 46 and the CLI published from the reused parent. Each arm changed the
Python render wrapper and CLI, ran source checks, submitted a fresh real render,
waited for completion and verified the result against its final source hashes.
Both passed seven source/adapter checks and a separate native artifact gate:
45 decoded PNGs, a 45-frame MP4, and matching inventories and bytes for all 49
native, selected and published files. Previous files were
preserved. This used an existing trained scene; neither arm retrained it.

**Faithful viewer tracks and sampling.** Grouping images only by their immediate
parent merged RGB, distance and opacity tracks for the same camera. Sampling also
omitted the final frame. Each arm changed the Workbench Rerun producer to retain
full relative paths and numeric source frame identities, and include both
endpoints when at least two frames are requested. Both passed eight artifact
cases, decoding 322 rows, plus 24 existing PAIDF regressions. The genuine Toro
fixture with a cap of 24 produced eight distinct entities and 192 rows instead
of the original four entities and 96 rows. These are visualization correctness
checks, not improved reconstruction quality or calibrated depth measurements.

The models were given the observable defects and narrowly scoped source files.
This measures repair of known problems, not unrestricted bug discovery or
autonomous workflow design.

## Authorship and recovery

| Repair | Astra alone | Hybrid |
|---|---|---|
| Render wrapper and CLI | Astra, two successful edits | GLM-5.3, two successful edits |
| Viewer grouping and sampling | Astra, two successful edits | Astra after specialist escalation, two successful edits |

DeepSeek-V4-Pro attempted the hybrid viewer task through 13 model calls but made
no edits. Five directory-listing calls were rejected because the configuration
granted exact files while the tool also required a directory grant. Its last
response was truncated; the GLM fallback's first response was also truncated.
Both provider responses reported `finish_reason=length` and 8,192 output tokens.
The client sent no output-token limit. The cause beyond that reported boundary
is unknown; rejected response bodies were not retained in the journal.

Astra inspected the failure, took over the resolved assignment and completed
the viewer repair itself. The original worker task remains recorded as
`cancelled`, including in the unfinished-worker inventory; it is not relabeled
as a successful DeepSeek task. Both repaired artifacts nevertheless passed
verification before the coordinator exited. GLM's render history also retains
one malformed edit rejected before mutation. Failed observations, fallback
responses and takeover costs remain counted.

## Matched procedure and real execution

Both arms used source base `39b9508b2ce616dd07bbdc8eaaf084303128b6a4`, the same
task text, source grants, operation grants, fixtures and independent graders.
The trusted runtime tree was `b86e2b537bbfcf5a48a436939ed9b3fdcdda0ad1`.
All 61 frozen protocol files remained unchanged. Checks and fresh artifact
verification had to pass after the final edit. Candidate source could not
access credentials, network, held-out graders or reference repairs.

Astra used `gpt-6-astra` with medium reasoning in both arms. The hybrid assigned
render work to `zai-org/GLM-5.3` with low reasoning and viewer work to
`deepseek-ai/DeepSeek-V4-Pro-0813` with reasoning disabled, with reciprocal
fallbacks. LangGraph workers and SQLite journals ran on an operator host;
Token Factory served the specialist models. Routing was explicit; Jev was not
used. The baseline had the same direct tools and concurrent-operation support.

The candidate Python wrapper and CLI ran in a CPU sandbox. A trusted broker
validated their native arguments and submitted an actual NRE render through a
Workbench GPU workflow, using an immutable runtime and existing checkpoint.
The native renderer ran on one RTX PRO 6000 Blackwell Server Edition GPU; the
candidate Python was not installed inside that GPU image. The broker returned
the actual native files to the candidate, checked local CLI publication and
published the verified output. All render inventories were independently bound
to source and artifact hashes. The viewer stage executed the candidate producer
on genuine retained output plus six held-out synthetic cases; its verifier
decoded Rerun files without importing the candidate's grouping or sampling code.

The public source data was NVIDIA's
[PhysicalAI-NuRec-PPISP](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP)
Toro scene, revision `2521064a3af6ab1c1caa2ba1b01ddde7eecded69`, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The native renderer was pinned to
`nvcr.io/nvidia/nre/nre-ga@sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`.
Both renders selected camera 2, rig Y offset −0.25 m, frame step 1, image scale
0.5 and 30 FPS. Runtime access and dataset use remain operator responsibilities.

## Costs and limitations

Hybrid model-price equivalents were Astra $2.84–$5.60, GLM $2.55 and DeepSeek
$0.75. Hybrid Astra input stayed near the baseline (1.84 million versus
1.83 million tokens), while the specialists added 2.31 million input tokens.
Delegation did not materially reduce coordinator input in this pair.
All recorded pricing-relevant input/output counters were available;
Token Factory did not expose separate cache-write or reasoning counters. Its
input is priced at the full input rate without assuming a cache discount.
Astra's aggregate turn counters do not identify each request's context pricing
tier, so the estimate retains both published rate possibilities. Subscription
usage valued at API rates is not an invoice.

Rates were recorded on 23 September 2026 from
[OpenAI Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Token Factory model information](https://tokenfactory.nebius.com/api/public/models_info).
GPU allocation is reported separately from model inference and is neither GPU
kernel time nor incremental whole-host billing. Human work, development-agent
inference, the original retained-checkpoint training, host compute, storage and
network are excluded from both these arm and setup totals. Development-agent
inference outside the measured runners was not metered.

Preparation is separate: three native probes used 777.000–777.865 GPU-seconds;
two model-readiness smokes cost $0.45336–$0.88022 and $0.46530–$0.90211 in
model-price equivalents. Preparation failures are retained: an overly strict
preservation oracle was corrected to permit new files while preserving old
bytes; an incorrect workflow artifact declaration was fixed; source checks
were changed to use complete snapshots after a partial workspace lacked a
required module. These corrections preceded the frozen pair. No measured
candidate or grader was changed while either arm ran.

The baseline ran first. Shared-host load, cache warmth and one trial per arm
prevent a causal speed claim; a 25.51-second lifecycle difference is not proof
of a repeatable advantage. Both arms met this task's acceptance criteria, but
that does not establish broad model parity. Owned experiment controllers and
API processes were cleaned up; unrelated workloads were preserved.

## Changes adopted after review

The production change retains GLM's render isolation/publication repair and
Astra's hybrid viewer repair. Separate developer changes reserve a fresh render
generation atomically on every invocation, including concurrent first use,
preserve metadata-only output directories, and reject duplicate numeric frame
IDs within one viewer entity. These additional edge cases were outside the
frozen task fixtures and do not alter either measured score. Reusing an output
publication URI can still merge older destination files; use a fresh URI per
render as the [NuRec guide](guides/neural-reconstruction.md) describes.

Runtime follow-ups allow scoped listing of explicitly granted files, encourage
line-range reads and small edits, and omit repeated source text from supervisor
status responses unless explicitly requested. Full journals, source hashes,
operation failures and uncertain effects remain available. These improvements
were made after the comparison; no savings claim is attributed to them.

Separate production validation on code commit `e5360fc30` passed the same seven
render source/adapter checks and native artifact gate: 45 PNGs, 45 decoded video
frames and 49 matching files. Its one additional GPU allocation was
262.000–262.287 seconds, outside the measured arms. A collector path assumption
failed in the nested post-review evidence directory; the unchanged independent
graders then checked the retained native files and submitted source directly.
No GPU retry or source change was needed, and the failed collector receipt is
preserved. Owned resources were cleaned up.

The final viewer also produced eight fresh accepted Rerun artifacts, totaling
322 decoded rows, and passed 24 PAIDF regressions. The committed public verifier
accepted both genuine-input recordings with embedded settings and full image
coverage. An additional hosted-model smoke paused and restarted both workers:
GLM repaired a public Cosmos3 PAIDF workflow and DeepSeek repaired Sim2Real,
then passed real Workbench validation and planning. This smoke used 11 model
responses, cost $0.11136 in model-price equivalents and submitted no GPU jobs.
These are separate post-review checks, not replacement trials or score changes.

The [comparison runner](../../npa/examples/specialists/workflows/README.md),
[artifact verifier](../../npa/examples/specialists/workflows/VERIFICATION.md),
and public regression tests are reusable Workbench material. Exact operational
configuration, raw model transcripts and infrastructure receipts are retained
outside Git; this aggregate report alone is not a one-command reproduction of
the operator's controlled benchmark.
