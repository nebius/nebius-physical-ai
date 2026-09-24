# Robot workflow repair: model cost and verified results

The subsequent [model-selection experiment](specialists-model-selection-experiment.md)
tests cheaper Token Factory models and direct dispatch of predefined assignments.
The original six runs and their results remain below.

In the final matched pair, Astra with Token Factory specialists completed the
same two repairs as Astra alone for **$0.405 versus $0.754–$1.469 in standard
model API-price equivalents**. This is at least **46% lower model cost**, but
the hybrid took **241.4 seconds versus 154.4 seconds**. These are measured
task-specific results, not a general savings guarantee or subscription invoices.

The earlier provider-failure run is retained below. Its hybrid cost was worse
even before accounting for missing usage, so the combined experiment does not
establish overall savings. Live Jev routing was not tested: no router credential
was available. All measured profiles used explicit Token Factory endpoints.

## What the agents did

Each arm repaired two deliberately injected defects in disposable copies of
existing Workbench source:

- Simulation recording: restore camera images and joint state at time `t`
  before action `t`, while retaining the resulting state at `t+1`.
- Dataset export: exclude physically rejected episodes and assign contiguous
  LeRobot indices to accepted episodes in their original order.

These were supplied benchmark defects, not newly discovered production bugs.
The agents inspected source, made edits, executed the real local CPU Workbench
MuJoCo Fetch simulator and LeRobot exporter, and verified the resulting
artifacts against their final source revision. The execution uses the same
production components as `token-factory-robot-sdg.yaml`, with prepared scenes
instead of an additional hosted scene-planning call. It does not submit a cloud
workflow, train a policy or demonstrate transfer to a physical robot.

Every repaired task ran three prepared cases: accepted grasp/place, a rejected
open-gripper control, and another accepted grasp/place. Independent verification
checked **555 physics transitions, 1,110 raw camera frames, 555 preview frames,
and 370 exported training timesteps with 740 aligned camera samples**. A separate
native LeRobot reader also formed a four-sample training batch. Failed grasps
remained inspectable in raw artifacts and were excluded from training data.

## All six runs

| Pair | Arm | Verified repairs | Model API-price equivalent | Agent/tool elapsed |
|---|---|---:|---:|---:|
| 1 | Astra | 2/2 | $0.765–$1.488 | 177.1s |
| 1 | Astra + Token Factory | 2/2, Astra recovery | Unknown; $1.823–$3.475 recorded | 398.1s |
| 2 | Astra | 2/2 | $0.655–$1.271 | 148.3s |
| 2 | Astra + Token Factory | 2/2 | $0.572 | 185.0s |
| 3, final runtime | Astra | 2/2 | $0.754–$1.469 | 154.4s |
| 3, final runtime | Astra + Token Factory | 2/2 | $0.405 | 241.4s |

The [cost records](specialists-robot-workflow-costs.json) retain every arm,
known token counter, missing usage, rate source and protocol comparison. The
[outcome records](specialists-robot-workflow-outcomes.json) retain actual repair
authorship, generic patches, source/artifact hashes and native acceptance counts.
Operational configuration, raw transcripts and media remain in private evidence.

Both coordinators used `gpt-6-astra` at medium effort. Specialist profiles used
`zai-org/GLM-5.3` and `deepseek-ai/DeepSeek-V4-Pro-0813` with reciprocal fallback.
**GLM authored both repairs in each successful specialist-led pair.** DeepSeek
participated, but its export generation was truncated at the provider's 8,192
output-token boundary; GLM completed that assignment. Rejected output was
charged in the comparison. No caller-added token, runtime or job budget was used.

Pair 1 instead required Astra to author both repairs after two GLM HTTP 404
responses. GLM had returned two successful responses before its error, and the
authenticated catalog still listed that exact model. The underlying serving
cause was not established. Both failed requests lacked usage counters; they
remain unknown charges rather than zero-cost calls.

The pair 2 GLM export patch also skipped malformed accepted records that lacked
simulation metadata. That behavior was outside the measured acceptance contract;
passing this benchmark did not establish the patch's production suitability.
The final pair's patch contained no such additional filters. Candidate patches
were not applied to production robot source.

## What changed to reduce cost

The first two pairs used fresh Astra delegation and review turns, compact
specialist context, and host-side waiting without model calls. Their order was
Astra first in pair 1 and hybrid first in pair 2.

After retaining those results, we declared a third pair with two additional
runtime changes. A confirmed unavailable endpoint can hand off to a configured
backup; failed requests retain sanitized diagnostics and missing usage. Hidden
client retries are disabled for specialists, and ambiguous transport outcomes
require attention. This 404 recovery path passed injected transport tests; the
final live pair had no provider error, so it did not exercise that new path.

The host can also finish directly when every configured assignment has fresh,
successful required-operation receipts, matching policy and current recorded
edit hashes, and no unresolved tool effects. It preserves original failures
after takeover and rejects recovery checks that predate a later worker edit.
Model claims alone cannot satisfy this gate.

The final hybrid used **one Astra delegation turn and no review turn**. Its
host-generated completion record followed both native verification receipts.
Astra stopped while specialists worked; routine polling and the final acceptance
gate invoked no model. The final model-cost split was Astra **$0.213138**, GLM
**$0.09924560**, and DeepSeek **$0.09223368**. Savings came from reduced Astra
work and cheaper specialist inference, with no assumed Token Factory cache
discount.

## Controls and limits

Each matched pair used fresh, disjoint workspaces, identical initial faults,
scene/reset parameters, common task, playbook, grants and acceptance criteria.
Pair 3 ran hybrid then Astra and retained the original four runs. Its runtime
amendment was frozen before either arm started; it is a separate development
iteration, not a replacement sample for either earlier pair.

Candidate code ran without network, credentials or access to the reference and
grader. The verifier replayed every control transition and raw image using
upstream MuJoCo/Gymnasium-Robotics, independently recomputed acceptance, and
checked exact decoded frames and timestamps against re-encoded verified raw
images. Native LeRobot checks covered every exported sample. Calibration proved
the unchanged reference passed and both supplied defects failed. Source hashes
bound submission, execution, verification and final completion.

Prices were retrieved on 2026-09-24 UTC from the official
[Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra.md)
and [Token Factory catalog](https://tokenfactory.nebius.com/api/public/models_info).
The rate snapshot is included in the cost records. Astra used ChatGPT login;
these numbers translate recorded usage into standard API prices and do not
measure the operator's subscription bill or service tier. Where aggregate turn
usage cannot bound every request below the 272,000-token pricing threshold, the
report retains both published context tariffs. Fresh hybrid turns below that
bound use the short-context rate. Reported Astra cached-input counters receive
the published cache tariff; all Token Factory input receives its full input rate.

A subsequent [Codex cost audit](specialists-codex-cost-audit.json) confirmed the
cache arithmetic and that reasoning tokens were already included in output.
The baseline's 336,912 input tokens include 299,264 cached tokens; its short-context
API equivalent is `(37,648 × $10 + 299,264 × $1 + 1,566 × $50) / 1,000,000`,
or $0.754044. The upper bound covers unknown per-request context tariffs; it
does not mean the entire accumulated turn necessarily received long-context
pricing. The ephemeral runs retained no request-level token records.

One Codex delegation turn can contain several model requests. The recorded turn
totals include their usage; calling it one API request would be inaccurate.
Neither actual Codex credit deductions nor invoices were measured. Within an
included ChatGPT allowance, the hybrid can add a separate Token Factory charge
while saving Codex quota rather than cash. Its $0.19147928 Token Factory component
is also a token-tariff estimate, not an invoice. The supervising development and
research conversation is outside these measured arms.

Elapsed time uses `execution.agent_tool_seconds`, including worker shutdown.
All recorded model attempts within each arm are included or marked unpriced.
Host allocation, storage, setup and work outside these measured arms are not
included in model-price estimates. The host was shared, and the small
fixed task set cannot establish general speed, quality or savings. Jev support
is implemented and unit-tested but contributes no live routing evidence here.

To run the native workload and independent verifier, see the
[robot workflow example](../../npa/examples/specialists/robot_workflow/README.md)
and [coordinator procedure](../../npa/examples/specialists/workflows/README.md).
