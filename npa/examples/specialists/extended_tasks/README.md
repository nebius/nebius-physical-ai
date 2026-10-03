# Workbench tasks designed for hours of agent work

These task briefs extend the short
[independent-checkout repair benchmark](../../../../docs/workbench/specialists-independent-experiment.md)
into sustained engineering and workflow operation. They ask agents to implement
missing behavior, integrate changes across components, operate real workloads,
recover from controlled failures, and deliver independently verified artifacts.
They do not reintroduce the earlier three historical regressions.

**Status: task definitions, not completed experiments or a runnable suite.** The
existing specialist runtime, workflow tools and artifact readers are reusable.
Campaign adapters, fault injection, input snapshots and task-specific independent
checks must be prepared and calibrated before these briefs become measured runs.
In particular, the existing repair benchmark's `prepare.py` does not load them.

| Task | Planning estimate per arm | Concrete result |
| --- | --- | --- |
| [Resumable robot data campaign](robot-campaign.md) | 2–4 hours | 216 simulated cases, restart-safe campaign code, verified LeRobot shards and a dataset index |
| [NuRec reconstruction campaign](nurec-campaign.md) | 4–8 hours | Four fresh reconstructions, 32 render products, cross-run isolation and a verified comparison viewer |
| [Cosmos3 generation and curation campaign](cosmos-campaign.md) | 3–6 hours | 48 initial video variants, evaluated refinement, immutable campaign reuse and a curated dataset |

These are uncalibrated estimates, including engineering, execution and review.
Hardware, caching, failures and parallelism change elapsed time. The simulator
estimate uses the earlier native workload as a rough scaling reference; the GPU
estimates are planning envelopes, not measured throughput promises. An arm that
finishes sooner passes on correctness. Do not add sleeps or repeat successful
work to reach a duration. The corpus sizes define comparable tasks; they impose
no time, cost, retry or job-count limit.

## Common agent assignment

Give each arm this README and one task brief. Astra alone gets the same complete
assignment and tool concurrency as Astra coordinating Token Factory specialists.
The hybrid may split work among the three suggested roles, each with its own full
Workbench checkout, operation namespace and outputs. Its coordinator owns the
integrated result. Role suggestions are not a requirement to call three different
models; retain the router's actual choices and any escalations.

Implement the requested behavior in the granted Workbench source. First reproduce
the missing behavior or failure, retain the evidence, then make a general repair.
Use real Workbench tools, inspect actual execution and repair problems along the
way. A passing plan, job submission, model final answer or manifest by itself is
not completion. Finish only after the integrated source passes the frozen checks
and its artifacts pass independent decoding and provenance checks.

Keep a durable handoff after every milestone: source digest, task and attempt
identities, verified outputs, unresolved failures, and the next operation. Let
native jobs and host waits run without model polling. Wake the coordinator for
an integration decision, failed quality gate, blocked dependency or ambiguous
operation; include its tokens and latency. Per-profile ownership locks remain
necessary alongside separate checkouts. Never blindly resubmit an uncertain job.

## Freeze before either arm starts

1. Pin one source revision, dependencies, container/model identities, input bytes,
   public requirements, independent graders and semantic failure triggers. Use
   current source; record any deliberately injected failure as a fixture rather
   than calling it a discovered production bug. An acceptance defect genuinely
   found in current source remains a separately described finding.
2. Prepare identical independent checkouts and direct operation grants for both
   arms. Candidate code runs away from credentials and graders. A trusted broker
   performs authorized external operations and records their source/spec hashes.
   Test authorship and receipt collection stay outside model write access.
3. Calibrate every native path with reference code and a small input, verify that
   each failure trigger fires, and prove the judge rejects corrupt artifacts,
   duplicates, missing cases and stale-source results. Freeze quality thresholds
   using separate calibration inputs. Do not tune them after seeing either arm.
4. Record the actual CPU/GPU allocation, equivalent concurrency entitlement,
   cache preparation, routing policy, exact seeds, prices and data split. Reuse
   the prior telemetry/accounting modules; do not assume the previous scorer
   understands new artifacts, GPU charges or completion criteria.
5. Check exact runtime/model access and storage readiness through the applicable
   root skills. Keep operational locations, credentials and access records outside
   Git. NuRec's NRE runtime is separately licensed; the public task code does not
   make that vendor runtime open source.

Use the same fixed workload and quality threshold for both arms. Do not give the
hybrid additional GPU capacity or make Astra execute independent tools serially.
Preserve all attempts, including abandoned changes, rejected model responses,
failed jobs, retries and operator interventions. Any intervention is visible in
the score; a human repair cannot be credited as autonomous completion.

## Failure and recovery protocol

The trusted harness injects failures at semantic milestones, once per declared
fixture per arm, rather than at elapsed-time offsets. This gives both arms the
same recovery problem even when they progress at different speeds. Trigger only
owned experiment processes or disposable object copies. Record the trigger,
affected identity and observed failure; do not fabricate a provider outage.

A killed controller does not imply that its GPU job was cancelled. Reconcile
actual execution before resuming. Preserve completed output bytes and distinguish
a partial output, a completed output with a lost acknowledgement, and an artifact
whose source or input digest changed. A successful recovery must prove which
work was reused, invalidated or rerun and why.

## Score complete outcomes

Report all milestone outcomes before comparing cost or speed. Each task is an
all-gates completion result; partial artifact counts are diagnostic. A campaign
that silently drops hard cases or lowers its quality gate fails. A legitimate
negative control or quality rejection remains evidence, not a forced acceptance.

Measure first coordinator launch through integrated artifact verification and
owned-worker shutdown. Include diagnosis, edits, test runs, queueing, workflow
execution, recovery, final integration and verification. Show model-active time,
model-free waiting and CPU/GPU job intervals separately; overlapping intervals
cannot be summed into wall-clock latency. Report installation, input staging,
calibration and between-arm integrity checks separately and show whole-experiment
elapsed time as well.

Account for every Astra, worker, router, fallback and evaluator call. Reconcile
Astra request telemetry with CLI totals. Preserve missing-usage uncertainty,
price cached subsets only once and assume no Token Factory prefix-cache discount
without billing evidence. API-equivalent Codex cost is not a subscription bill.
Include GPU/CPU allocation, storage and network estimates in a separate total;
unknown infrastructure rates prevent a total-cost savings claim. Development
conversation and preparation remain explicitly separate.

Start with one matched pilot per task, report it as a pilot, and calibrate actual
duration. For a repeatable comparison, freeze a fresh cohort of three pairs in
the order Astra/hybrid, hybrid/Astra, Astra/hybrid. Retain the pilots separately;
do not replace an unfavorable measured arm with a retry. Equal verified quality,
lower end-to-end latency and lower full cost must all hold to claim this stack is
better on a task. Report per-task outcomes and paired differences; do not hide a
GPU-heavy loss inside an aggregate model-cost saving.

## Deliverables retained for each arm

- Source patch and file hashes, integration checks and meaningful regression tests.
- Workflow specifications, effective configurations, stage receipts and attempt history.
- Complete artifact inventory, decoded verification reports and a viewable example.
- Recovery trace proving preservation, invalidation and absence of duplicate effects.
- Timing and usage ledger, infrastructure accounting, limitations and operator interventions.

Publish only reviewed Workbench code and sanitized aggregate evidence to the
existing PR. Raw conversations, captures, infrastructure details and vendor
payloads stay in private evidence storage. The agent layer uses Workbench's
public specialist runtime and explicitly configured model endpoints.
