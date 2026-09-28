# Matched Workbench workflow repair benchmark

On September 27, 2026, we compared a single Astra agent with Astra delegating
to three concurrent Token Factory specialists through Workbench and LangGraph.
The benchmark replays known public regressions in physical-trace validation,
transactional dataset publication and simulation-to-LeRobot validation. Models
repair source, run native tools, inspect failures and verify their final patches.
The source fixes already exist in this PR; these are fresh repair attempts on
historical copies, not newly discovered production bugs.

## Observed results

Both arms passed all three repeats, including all three repair lanes and each
combined native workflow. The hybrid had **74.8% lower median model cost** under
the conservative comparison (its upper estimate against Astra's lower estimate).
Its median elapsed time was **2.5% shorter**.
The frozen all-correct/lower-median-cost/lower-median-time criterion was
**met**. This is equal measured
correctness; it does not establish higher output quality.

| Pair | Arm | Elapsed | Astra | Token Factory | Total including setup uncertainty |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | astra-only | 382.6s | $2.004092 | $0.000000 | $2.004092–$2.115122 |
| 1 | astra-tofa | 373.1s | $0.246146 | $0.142142 | $0.388288–$0.499318 |
| 2 | astra-tofa | 368.8s | $0.264932 | $0.151635 | $0.416567–$0.527597 |
| 2 | astra-only | 364.5s | $2.093528 | $0.000000 | $2.093528–$2.204558 |
| 3 | astra-only | 457.3s | $2.524680 | $0.000000 | $2.524680–$2.635710 |
| 3 | astra-tofa | 528.7s | $1.474946 | $0.138612 | $1.613558–$1.835618 |

| Median | Elapsed | Total model estimate |
| --- | ---: | ---: |
| astra-only | 382.6s | $2.093528–$2.204558 |
| astra-tofa | 373.1s | $0.416567–$0.527597 |

The six measured arms together used **$9.040713–$9.817923** in
standard API-equivalent model usage. This includes baseline and hybrid runs,
not the separately excluded development or environment costs.

Across all three repeats, the hybrid's total was
**$2.418413–$2.862533**, versus
**$6.622300–$6.955390** for Astra.
Even with the recovery outlier, the conservative aggregate saving was
**56.8%**.

The hybrid was faster in **1/3 matched pairs**. Mean elapsed time was
**423.5s hybrid versus 401.5s Astra**.
Read these alongside the median; the experiment does not prove a consistent
speed advantage.

Actual routing selections: `zai-org/GLM-5.3-Flash`: 8.
The full-GLM endpoint was available as a candidate; no routing decision was
overridden to force a desired model mix.

Retained failed native attempts: 1.

- Pair 2, astra-tofa, publication.

The second hybrid publication worker initially left a linked staging artifact.
The independent verifier rejected it despite passing regressions. The worker
repaired cleanup and completed a fresh native attempt without operator edits.
All diagnosis failures, retries and their model usage remain in the evidence.

The third hybrid run hit a transient ownership refusal while delegating the
publication lane. The old bridge returned an undifferentiated error, so Astra
returned for a recovery turn. That delay and all recovery tokens remain in the
measured result. This PR now reports a pre-submission refusal with explicit
`submission_attempted: false` and `safe_to_retry: true`; later errors retain
uncertainty. Real-lock and post-submission failure tests cover both boundaries.
The fix was made after freezing this experiment and is not credited with any
measured latency improvement.

[Machine-readable results, patches and receipts](specialists-matched-repair-results.json)
include every declared arm. The [public runner and instructions](../../npa/examples/specialists/repair_benchmark/README.md)
provide the frozen protocol, preparation, calibration, paid execution and cost
scoring. Price sources are the [Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Token Factory catalog](https://tokenfactory.nebius.com/model-catalog.md).

## Protocol and quality

The protocol was frozen before measured inference at runtime commit
`52ea0359e`, which incorporated main `50e6dd0fe`. Later publication validation
uses the branch after merging main `5e8ad57be`; the measured snapshot stayed
unchanged. Three pairs ran serially on
the same Linux host, with order Astra/hybrid, hybrid/Astra, Astra/hybrid. Each
arm received the same starting files, detailed requirements, 128 immutable
regression checks, three-case scene matrix and native tool grants. Both could
run operations concurrently. Astra used medium effort in both arms; native
shell/file tools and additional Codex agents were disabled in both. This does
not compare a fleet of Astra agents against a fleet of cheaper models.

The hybrid used an actual Astra delegation call, a real Lightning model
selection request per specialist and LangGraph's durable worker/tool loop.
The host waited for completion outside the model and required source-bound
receipts before completing. All Astra delegation/recovery and Token Factory
classification, generation, rejection and fallback usage count. Jev remains an
optional external route; this measured run used the public Token Factory router.

Each lane needed current-source passing checks plus a native three-case run.
The host then combined all three patches, repeated the 128 checks, and executed
the six-case integration matrix. Every successful combined run had six MuJoCo
Fetch episodes, 1,110 physics transitions, four accepted demonstrations and two
failed-grasp controls retained as raw evidence. Independent replay checked the
state, action, contacts, two camera streams and encoded artifacts. Native
LeRobot 0.5.1 loaded 740 training timesteps, 1,480 aligned camera samples and a
real four-sample training batch.

This is native CPU Workbench simulation and dataset conversion, using the
implementation behind robot SDG. It does not submit a GPU/cloud workflow,
train a policy or demonstrate physical robot transfer. The supplied scenes and
known repair requirements make this a controlled engineering benchmark, not an
open-ended discovery or general agent-intelligence benchmark.

## Accounting and limits

Elapsed time includes coordinator startup, routing, source edits, all repair
attempts, worker shutdown and final combined native verification. Environment
installation and reference calibration precede the measured arms. No failed
or expensive arm is discarded. No operator edited candidate files during the
measured sequence.

Astra costs use per-request counters reconciled against all five CLI counters.
Cached tokens receive their published cached rate once, and context tariffs
apply per request. Initial input-only response records omitted from CLI turn
totals remain an explicit zero-to-full-price uncertainty interval. Token Factory
usage receives the full input tariff with no assumed caching discount.

These are standard API-equivalent model estimates, not verified Codex invoices
or subscription charges. The development conversation, setup/calibration and
host compute, storage and network charges are excluded. Recurring model savings
do not establish total project expenditure. Three repeated pairs are a small
sample; a lower observed median does not establish a statistically significant
speed advantage or generalize to other tasks, models or provider loads.

Two environment preparation attempts failed before inference: one bypassed a
virtualenv interpreter symlink; another referenced a missing native-reader
installation. Both were retained and fixed before the passing calibration. They
made no generation calls. The measured campaign used a fixed private copy of
the previously verified filtered telemetry receiver; the public runner now
includes its portable counterpart with privacy and loopback transport tests.
Raw transcripts, exact runtime paths and artifacts remain in operator storage.

## Publication validation

At source commit `4fbfa193ca19`, the full Linux suite passed with **32,762
passed, 128 skipped and one non-strict XPASS**, at **76.88% package coverage**.
The separate serial security gate passed **878 tests with no skips**, using the
required CPU Torch 2.14.0 build. Local validation also passed 649 agent-evaluation
tests (three skips), 234 Cypress tests, 28 native-protocol tests, 266 precheck
tests and CLI documentation drift checks. The aggregate diff passed secret and
infrastructure-confidentiality scans.

Validation prerequisites were corrected before the passing run: the checkout
and temporary test homes use isolated locations, the validation host has the
required Git version, and the readiness fixture supplies an explicit synthetic
local target on supported Linux hosts. Earlier failed validation logs remain
in private evidence. The passing test and coverage receipts are hashed in the
machine-readable results; the subsequent publication commit changes only
reports and documentation.
