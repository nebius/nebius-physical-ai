# Choosing Token Factory models for workflow repairs

For this predefined pair of repairs, **GLM-5.3-Flash with Astra only on escalation**
completed both tasks in two fresh comparisons with **zero Astra calls**.
It took **100.3s and 106.4s versus Astra's 141.4s and 151.3s**, and cost an
estimated **$0.01008 and $0.01167** in Token Factory usage. That is **29–30% faster
and at least 98.49% lower model API-equivalent cost** on these tasks. These are
verified task-specific results, not subscription invoices or a general guarantee.

This follow-up to the [robot workflow experiment](specialists-robot-workflow-experiment.md)
keeps the same two injected source defects, acceptance criteria, native execution
and independent artifact checks. It screens cheaper models, confirms the chosen
recipe on fresh workspaces, and separately tests direct dispatch of predefined
specialist assignments.

The task is real local CPU MuJoCo Fetch simulation followed by native LeRobot
export and loading. Each arm must repair camera/state ordering before action and
exclude failed episodes from training while retaining raw evidence. These are
controlled defects in disposable copies, not newly discovered production bugs.
Each passing task verifies 555 physics transitions, 1,110 raw camera frames,
370 accepted training timesteps, 740 aligned camera samples, and a native
four-sample training batch. No cloud workflow, GPU training or policy quality
improvement is claimed.

## Model screening

The accessible catalog contained 24 models. Price and advertised throughput
suggested two candidates; only these two were screened. Advertised throughput
was a selection hint, not measured task performance. Both models used explicit
Token Factory endpoints. Jev was not invoked.

| Pair | Arm | Verified repairs | Elapsed | Model API-equivalent cost |
| --- | --- | --- | --- | --- |
| 4 | Astra | 2/2 | 184.4s | $0.774–$1.508 |
| 4 | Lightning specialists + Astra | 0/2 | 345.2s | $0.907 |
| 5 | Astra | 2/2 | 151.1s | $0.905–$1.768 |
| 5 | Flash specialists + Astra | 2/2 | 127.5s | $0.352 |

The screened models were `nvidia/Nemotron-3_5-Lightning` with thinking disabled,
and `zai-org/GLM-5.3-Flash` with low reasoning. Each used the other as its
configured fallback. Neither run actually invoked its fallback.
Lightning made no source edits and repeated 32 failed verification checks.
Both workers were explicitly stopped after the repeated no-progress behavior;
the retained Astra recovery turn also made no repair. That run is unsuccessful
and operator-assisted, and its entire recorded cost remains in the comparison.
This result does not establish how Lightning performs with other settings.

Flash authored one correct edit per task and passed both native checks without
intervention. In pair 5 it was 15.6% faster and 61.1–80.1% cheaper in model
API-equivalent cost. Its specialists cost $0.01289890; Astra delegation cost
$0.339486. The four screening arms together cost an estimated $2.937–$4.534,
including $0.38327812 of Token Factory usage and the failed Lightning attempt.
These are descriptive matched runs on a shared host, not a population estimate.
Arm order was Astra first in pair 4 and hybrid first in pair 5.

[Screening costs and outcomes](specialists-model-screening-results.json),
[independent artifact evidence](specialists-model-screening-outcomes.json), and
[rate snapshot](specialists-model-selection-prices.json) retain the comparison.

## Confirmation with Astra delegation

Pair 6 froze the exact pair 5 model recipe and runtime before either arm and
reversed the order to Astra first. Both arms again passed 2/2 repairs. Flash
with one Astra delegation turn took **129.3s versus 157.8s** (18.1% shorter).
Recorded CLI-plus-specialist usage prices at **$0.23041980 versus
$0.932278–$1.823256**; the Flash portion was $0.01139180.
[Independent confirmation evidence](specialists-model-confirmation-outcomes.json)
retains both actual patches and the original-source/artifact bindings.

New request-level Codex telemetry exposed an additional 11,103-input-token,
zero-output completion in **each** arm that the CLI aggregate omits. Its shape
is consistent with Codex connection prewarming, but the retained metadata does
not directly label it as such or establish billing. Its standard input-price
sensitivity is **$0–$0.111030 per arm**, separately from the estimates above.
It is not silently discarded. A separate [request-level cost view](specialists-model-confirmation-costs.json)
reconciles the CLI counters after removing that event and proves short request
contexts: $0.932278–$1.043308 for Astra and $0.23041980–$0.34144980 for the hybrid,
including the independent setup sensitivity. The earlier runs have no usable request-level
telemetry and retain their original conservative context-tariff ranges.

## Runtime changes motivated by the experiment

The opt-in `handoff_on_failure` operation policy preserves a completed failed
receipt and hands the next inference step to a configured backup model. It never
replays an operation or treats an unresolved effect as a retry opportunity.
This addresses the observed verification loop without adding a time, token or
job-count limit. Existing operations keep their prior behavior by default.

For already defined workflows, `--coordination specialists-first` directly
submits all configured assignments and uses Astra only for escalation. Profiles
must have explicit instructions and required checks. This avoids paying Astra
to repeat a decomposition already supplied by the operator. Arbitrary new
requests still benefit from the default Astra delegation mode. See the
[runner instructions](../../npa/examples/specialists/workflows/README.md).

The direct-dispatch candidate uses these endpoint fields in each existing
operator-owned profile:

```json
{
  "model": "zai-org/GLM-5.3-Flash",
  "model_options": {"chat_template_kwargs": {"reasoning_effort": "low"}},
  "fallback_models": [
    {
      "model": "zai-org/GLM-5.3",
      "model_options": {"chat_template_kwargs": {"reasoning_effort": "low"}}
    }
  ]
}
```

Keep that profile's instructions, workspace grants and real verification command.
Set `handoff_on_failure: true` on its trusted `verify` operation and run the
workflow runner with `--coordination specialists-first`. The fallback is a
configured recovery path; a successful Flash-only run does not test it.

## Direct-dispatch architecture comparison

A separate iteration froze runtime `fd85feadeaaac72151dcf2d6dfef793f88e27db1`
before two fresh matched pairs. Both specialist profiles used Flash with low
reasoning, full GLM-5.3 as backup, and the opt-in verification-failure handoff.
The host dispatched the two operator-defined assignments directly; Astra
remained available for escalation. Source defects, common task text, playbooks,
operation grants, concurrency and independent native acceptance stayed fixed.

| Pair | Arm | Verified repairs | Elapsed | Model cost estimate |
| --- | --- | --- | --- | --- |
| 7 | Astra | 2/2 | 141.4s | $0.666748–$0.777778 |
| 7 | Flash, Astra on escalation | 2/2 | 100.3s | $0.01007720 |
| 8, confirmation | Astra | 2/2 | 151.3s | $0.987136–$1.098166 |
| 8, confirmation | Flash, Astra on escalation | 2/2 | 106.4s | $0.01166725 |

Flash authored one edit per task and passed all four repairs, with one native
attempt per task and no fallback, intervention or Astra call. Positive evidence
binds the configured dispatch to exact task goals, policies and journals, and
binds host completion to those same verified native artifacts. Empty logs alone
are insufficient. The two timing reductions were **29.03% and 29.70%**. Model cost was at least
**98.49% and 98.82% lower** than the corresponding baseline estimates.
The table uses request-level standard-rate estimates for Astra, with the upper
value including the full input-only setup sensitivity. Recorded request prompts
were all below the long-context boundary, and all CLI counters reconcile after
removing that separately retained input-only event. This tightens the earlier
conservative turn-only ranges; actual billed service tier remains unknown.
See the [first pair cost audit](specialists-direct-dispatch-costs.json),
[confirmation cost audit](specialists-direct-dispatch-confirmation-costs.json),
and [independent outcomes](specialists-direct-dispatch-outcomes.json).

Pair 7 ran specialists first; pair 8 ran Astra first. Host load was retained in
private evidence; these are two descriptive comparisons on a shared host.
They do not establish general superiority on other tasks or open-ended planning.

The original pair 8 preparation stopped before inference because a selected
local audit had not yet been copied to the execution host. Its failed preparation
is retained. The fresh confirmation directory has distinct identities and no
reused execution state; it is not a hidden retry of a model or native workload.

## Exploration cost and validation

The [complete follow-up ledger](specialists-model-selection-results.json) joins
all ten measured arms, independent outcomes and the separate preparation failure.
The ten-arm exploration cost was **$5.7756–$7.8167** in model-price equivalents,
using request-level context evidence where available and including the observed
setup-token sensitivity. The **$0.41641437 Token Factory estimate is part of
that total**, including failed Lightning usage. Original conservative turn-only
ranges remain in the ledger; no failed candidate was replaced by a success.
This follow-up total excludes earlier experiments documented in their own reports.

Runtime `fd85feadeaaa` passed the complete Linux suite: **27,419 passed,
122 skipped, 1 xpassed**, zero failures and **76.33% source coverage**. Required
security regressions passed **859 tests** with the pinned CPU Torch runtime.
The focused specialist suite passed **509 tests**, with two opt-in live tests
skipped; the paid native comparisons above are separate. The final documentation
and sanitized evidence do not change that runtime.

## Cost interpretation

The [Codex audit](specialists-codex-cost-audit.json) distinguishes a visible
Codex turn from the underlying model requests. Every recorded turn and specialist
response is counted, including failures. The long-context tariff applies per
request; aggregate turn tokens alone cannot select it. Earlier runs retain a
range when request-level telemetry is missing. Reasoning tokens are a subset of
output tokens and are not charged twice.

Astra used ChatGPT authentication. Dollar figures apply standard API prices to
recorded model usage; they are not measured subscription bills, credit deductions
or invoices. Included Codex quota can mean that adding Token Factory increases
cash spending while saving quota. Token Factory input is priced in full, with
no assumed prefix-cache discount. Host compute, storage, human work and the
supervising development/research conversation are excluded; their actual cost
has not been established. See [official Codex billing documentation](https://learn.chatgpt.com/docs/pricing)
and [Astra API pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).
