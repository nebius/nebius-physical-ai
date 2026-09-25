# Token Factory routing for Workbench repair fanouts

The public specialist runtime now supports a Token Factory model as its endpoint
selector. `nvidia/Nemotron-3_5-Lightning` makes one structured classification per
task; LangGraph runs the selected worker against its existing Workbench grants.
The classifier cannot add tools or workspace access. This path needs the ToFa
credential and does not require Jev or a TypeSafe service.

With Astra delegating the work, the routed stack passed both repairs in
**134.1s versus 181.1s**, with **at least 53.8% lower model API-equivalent cost**.
The predefined dispatch mode was cheaper still and needed no Astra invocation.

## Measured repair task

Each arm receives fresh copies of two deliberately broken Workbench source
files. One repair restores camera/state capture before its corresponding action.
The other excludes failed physical episodes from the LeRobot training dataset
while retaining raw evidence. The two assignments can run concurrently.

Each assignment reruns three CPU MuJoCo Fetch cases: two successful pick-and-place
cases and an open-gripper failure control. A separate verifier checks actual
physics and image alignment, then loads the exported dataset through native
LeRobot. Every passing assignment verifies 555 transitions, 1,110 raw camera
frames, 370 accepted training timesteps, 740 aligned camera samples and a real
four-sample training batch. The verifier is outside the agents' editable scopes.

These are repairs in disposable source copies, not newly discovered production
bugs. No GPU job, learned-policy training or physical robot transfer is claimed.

## Predefined fanout, Astra available for escalation

Two matched pairs froze runtime, original faults, task text, tools and acceptance
before execution. The second pair reversed arm order. Astra alone had the same
Workbench operations, including concurrent `run_operations`. Both specialist
profiles offered Flash and full GLM; the classifier selected Flash each time.

| Pair | Astra alone | Routed stack | Astra model cost | Stack model cost |
| --- | ---: | ---: | ---: | ---: |
| 1, Astra first | 165.0s | 101.4s | $0.69479–$0.80582 | $0.010462 |
| 2, stack first | 164.5s | 99.3s | $1.07453–$1.18556 | $0.010011 |

All four arms passed both repairs. The stack reduced measured duration by
38.5–39.6% and standard model API-equivalent cost by at least 98.49–99.07% in
the corresponding pairs. Each stack run made two paid classifier calls and
used **zero Astra invocations**: the predefined assignments completed without
escalation. This is not evidence of zero-cost open-ended task decomposition.

## Astra delegates to the routed specialists

A separately declared matched pair uses `--coordination completion`. Astra
actually assigns both workspaces; each receives a paid Lightning routing
decision. Flash then edits source, runs simulation/export and completes the
required checks while the host waits without additional Astra calls.

| Arm | Verified repairs | Agent/tool elapsed | Model API-equivalent cost |
| --- | ---: | ---: | ---: |
| Astra alone | 2/2 | 181.1s | $0.72666–$0.83769 |
| Astra + routed ToFa stack | 2/2 | 134.1s | $0.22436–$0.33539 |

This run used one Astra delegation turn, two Lightning classifier calls and
Flash-authored repairs in both lanes. Host verification completed the run
without an Astra review turn or model fallback. It was **26.0% faster and
53.8–73.2% cheaper** across independently varied setup-token sensitivities.
The CLI-recorded model cost alone was $0.22436407, comprising $0.213658 of Astra
usage and $0.01070607 of ToFa worker/classifier usage. This is one additional
pair, not a repeated statistical estimate for the delegation mode.

All six measured arms together cost $2.74082–$3.18494 in these model
API equivalents, including setup uncertainty. The machine-readable evidence
retains all six; no measured arm was dropped. Runs occurred on 2026-09-25 UTC.

Separately, the committed opt-in live test passed both copied PAIDF Cosmos3 and
Sim2Real specification repairs through actual routing, generation, Workbench
validation and planning. Its two classifier and 14 Flash responses cost an
estimated $0.01198848; they are outside the timed comparisons. It submitted no
GPU workload.

## How to run the stack

Use the [ToFa routing configuration](specialists.md#model-driven-routing-through-token-factory)
with `require_model_route: true`, Lightning as `routing_model`, Flash as the
low-cost endpoint and full GLM as the more capable candidate. Retain each
profile's real instructions, workspace grants and required verification
operations. The runtime also supports other explicitly configured compatible
endpoints; this experiment does not benchmark them.

The [workflow runner](../../npa/examples/specialists/workflows/README.md) supports
two useful coordination modes: `completion` invokes Astra for delegation and
waits in the host between actionable events; `specialists-first` directly
dispatches already configured assignments and invokes Astra only when needed.
Use fresh workspaces and state for comparisons. Required routing rejects invalid
or abstained choices before generation, and a lost routing response is never
silently replayed. Decisions and usage survive worker restarts.

## Cost and evidence boundaries

The [machine-readable results](specialists-token-router-results.json) retain
actual patches, source/artifact bindings, classifier choices, latency, model
counters and separate Astra accounting. Routing and every recorded generation
are included. Timing covers model calls, tool execution, native verification and
worker shutdown after initial configuration preparation. Infrastructure setup
and the final evidence snapshot are outside that timer.

Codex used ChatGPT authentication. Dollar figures apply published **standard
API-equivalent tariffs**, not actual invoices or subscription charges. Captured
request contexts fit the short-context tariff. An additional input-only setup
completion omitted from CLI usage is retained as a $0–$0.11103 sensitivity per
observed Astra invocation; its billing is unknown. Astra cached input is priced
once and reasoning is already included in output. ToFa input uses the full
published tariff, without an assumed cache discount.

Rates come from the [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
and [Token Factory catalog](https://tokenfactory.nebius.com/api/public/models_info).
CPU hosting, storage, networking, development conversations and separate live
validation are excluded from these timed-arm model costs. Actual service tier
and customer cash savings are not established.

This small fixed-task experiment does not establish general superiority or a
calibrated routing policy. All observed selections chose Flash, so it does not
measure the router's marginal value against always selecting Flash. One Flash
export patch additionally skips accepted records missing simulation metadata;
that malformed-input policy was outside the acceptance checks and needs review
before production adoption. Earlier [model screening and failures](specialists-model-selection-experiment.md)
remain separate, with their original results unchanged.
