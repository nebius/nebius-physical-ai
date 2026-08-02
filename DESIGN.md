# DESIGN — real parallel execution and real runtime control flow for `npa.workflow/v0.0.1`

This document covers the two tiers added to the `npa.workflow` engine:

1. **Parallel execution** — a spec can declare a fan-out group that launches as
   genuinely concurrent SkyPilot jobs, with a barrier and bounded concurrency.
2. **Runtime control flow** — an orchestrator above `build_scheduler_task` that
   submits a wave, polls it to a terminal state, reads the *actual* decision
   artifact from S3, and replans: real early-exit, real data-dependent branching,
   retry/idempotency/resume, and a trigger/watch pattern.

Both are **additive and opt-in**. Every pre-existing spec plans, renders and
submits exactly as before, and the plan-time `--assume-decision` path is retained
as the offline/plan-only mode.

---

## 1. Where the engine was

| Layer | Behaviour before this change |
| --- | --- |
| `interpreter.build_plan` | Statically unrolls loops (`for iteration in range(1, max+1)`) into a flat serial plan; the loop-exit predicate is fed a **plan-time assumption** (`--assume-decision`) |
| `interpreter.run_workflow(execute=True)` | A dynamic walker that *does* read `config.decision_uri` from S3 (`decision_reader`) — but it executes every step locally with `subprocess` |
| `scheduler.build_scheduler_task` | Portable per-step task doc (name, resources, command, image, outputs) — the seam |
| `skypilot_render.render_skypilot_yaml` | Emits `execution: serial` pipelines and **rejects** anything else |
| `submit.prepare_npa_workflow_for_submit` | One-shot: load → plan → render → `sky jobs launch`. No monitor, no replan |
| `decisions.py` + `write_*_decision` toolRefs / `data_factory_stages.grade_gate` | Already *produce* and normalize decision artifacts; only a runtime *consumer* was missing |

So the missing pieces were: a concurrent submission shape, and a driver that
consumes decisions between stage submissions.

---

## 2. Spec surface (all optional, all `npa.workflow/v0.0.1`)

```yaml
states:
  sweep:                                  # (1) fan-out group
    parallel: [v-lr-1e-3, v-lr-3e-4, v-entropy-0, v-entropy-0-01]
    maxConcurrency: "{{config.max_concurrency}}"   # int or config ref
    needs: [prepare]
    next: select-best                     # barrier edge, owned by the group

  v-lr-1e-3:                              # (2) per-state config overlay
    resources: trainer-gpu
    params:
      variant: lr-1e-3
      overrides: "agent.algorithm.learning_rate=1.0e-3"
      variant_uri: "s3://{{config.bucket}}/{{config.prefix}}/variants/lr-1e-3/"
    run: { shell: "... {{config.variant}} {{config.variant_uri}} ..." }

  ingest:                                 # (3) trigger / watch
    trigger:
      uri: "s3://{{config.bucket}}/{{config.prefix}}/inbox/"
      pollSeconds: 30
      maxPolls: 20
      minObjects: 1
    toolRef: workbench.dataset.ingest
```

### Why these shapes

* **`parallel:` is a sibling of `sequence:`.** Fan-out already existed in the
  catalog (`av-night-scene-hardening.yaml` lists two per-view branches under
  `sequence:`), and `docs/workbench/npa-workflow-guide.md` already named
  `parallel` as the field advanced scheduling should use. Rejected: Jinja loops,
  `foreach` templating.
* **`params:` instead of `{{item.*}}` matrix templating.** A `toolRef`'s argv
  template is fixed, so sweep members must differ *somehow*. A per-state config
  overlay needs **no new token scope** — `{{config.learning_rate}}` simply
  resolves against `config | params` for that state — and the resulting spec is
  the direct analogue of the SkyPilot template's four explicit task documents.
  `params` is also useful outside sweeps (the gate-loop's `escalate` state reuses
  one dashboard toolRef with a different output prefix).
* **`trigger:` gates a state that does work.** A wait-only state would render as
  an empty scheduler task; validation rejects it. Watermarks live in the runtime
  ledger so a resumed run does not wait again for data it already saw.

### Validation rules (all fail at `validate-spec`)

* `parallel` and `sequence` are mutually exclusive; a `parallel` state may not
  also carry `run`/`toolRef`, and may not carry `loop` (wrap it in a `sequence`).
* Members must exist, be unique, be leaf states (no `sequence`/`parallel`/`loop`),
  must not be `terminal`, and must not declare `next`/`transitions` — the group
  owns the barrier edge, so the graph stays deterministic.
* `maxConcurrency` requires a `parallel` group and resolves to `>= 1`.
* `params` values and `trigger.uri` are token-resolved at validate time (they see
  the same per-state overlay the command does).
* `trigger` requires `run`/`toolRef` on the same state.

### Why not bump the apiVersion

`SUPPORTED_API_VERSIONS`, `blueprints.py`, the submit matrix, ~36 shipped specs,
the skills and the guide all key on `npa.workflow/v0.0.1`. The new fields are
optional and ignored by every existing spec, so a `v0.0.2` fork would duplicate
the catalog for zero semantic gain. The guide's "out of scope (v0.0.1)" list said
these features "belong in spec v0.0.2+ as explicit fields (`parallel`, `gang`,
`foreach`)"; that section is updated in this change — we adopt the *explicit
field* direction it asked for, inside v0.0.1, and note it in the doc.
`gang` and `foreach` remain out of scope.

---

## 3. Parallel model

### Planning: one plan, two lenses

`build_plan` **flattens** a `parallel:` group in declared order, exactly like
`sequence:`. Consequences:

* `--plan-only` output, `plan-spec`, and every existing plan-only guardrail
  (including "every rendered header says `execution: serial`") are unchanged for
  serial **and** parallel specs.
* Offline preview of a parallel spec is still meaningful (it is the serialized
  execution of the same work).

`waves.build_wave_plan` is the second lens: it folds the flat step list into
**waves** using `PlanStep.group`:

| Wave kind | Contents | Submitted as |
| --- | --- | --- |
| `serial` | exactly one step | SkyPilot pipeline document (`execution: serial`) |
| `parallel` | the members of one group | SkyPilot **JobGroup** (`execution: parallel`), chunked by `maxConcurrency` |

`plan-spec --waves` prints this shape offline.

### Rendering: a separate entry point, serial output unchanged

`render_skypilot_yaml` still raises on anything but `execution: serial` — the
historic guardrail test passes verbatim — and its **output is byte-identical** to
before (verified by planning three dynamic specs on the base commit and diffing).

Its *body*, however, was refactored: it now delegates to a shared `_render_docs(...,
execution="serial")` that the parallel renderer also uses. So the accurate claim is
"**serial output unchanged**", not "serial renderer untouched": there is one shared
code path, and the guard plus the byte-identical-output check are what protect it.
Parallel rendering is a separate public entry point,
`render_skypilot_job_group_yaml` (with the dispatching
`render_skypilot_steps_yaml`). Every task doc is still produced by
`build_skypilot_task_doc` → `scheduler.build_scheduler_task`, so the portable-task
seam is intact and the runtime tier never reaches into rendering internals.

### Why a SkyPilot JobGroup

SkyPilot 0.12 treats a multi-document YAML whose header sets
`execution: parallel` as a JobGroup:

* all tasks share **one managed `job_id`** but each launches **its own cluster
  concurrently** (`sky/jobs/scheduler.py`: *"For JobGroups, multiple tasks share
  the same job_id but each launches a different cluster in parallel"*);
* `primary_tasks` is intentionally **omitted**, which marks every task primary, so
  the group only reaches a terminal state once **all** members do.

That last property is the barrier, enforced by SkyPilot itself rather than by
bookkeeping in the driver. Polling stays a single `workflow_status(job_id)` call,
and the existing aggregation ("SUCCEEDED only if all tasks SUCCEEDED") already
handled multi-task jobs.

### Barrier and bounded concurrency

* **Barrier**: the wave boundary. The runtime tier submits the downstream state
  only after the group's wave reached a terminal state. Live evidence is the
  per-task timeline: the barrier task's `submitted_at` is later than every member's
  `end_at`.
* **Bounded concurrency**: JobGroups have no concurrency cap, so a group larger
  than `maxConcurrency` is **chunked into batches**, each batch one JobGroup,
  batches submitted in order. The declared bound is therefore authoritative and
  observable (`plan-spec --waves` prints the batch count). `--max-concurrency`
  overrides every group at submit time (cost control for GPU sweeps).

---

## 4. Runtime tier

### The state machine

```
for each wave produced by the traversal:
    key = f"{sequence}|{group}|{loop_label}:{state}:{iteration},..."
    if resuming and ledger says key succeeded  -> replay record, continue
    render wave  (1 step -> serial doc, N steps -> JobGroup batches)
    submit_workflow(...)                        -> job_id            [ledger]
    poll workflow_status(job_id) every poll_seconds until terminal
        while polling a parallel wave, sample per-task statuses      [ledger]
        SUCCEEDED  -> record per-task timeline                       [ledger]
        FAILED     -> retry (<= --retries, backoff) else fail the run
        timeout    -> cancel job + cluster, fail the run
    if the state writes a decision or has transitions:
        load the decision artifact from S3 (existing contract)       [ledger]
    traversal decides: iterate / break early / goto / next / terminal
```

### It is the existing traversal, not a second engine

`interpreter._execute_state_machine` already implements loops, `loop.until`,
`transitions`, `needs`, depth guards and *S3 decision refresh*. The runtime tier
injects two things into it:

* `step_executor` — `SkyPilotWaveExecutor`, whose `execute(step)` /
  `execute_parallel(steps, group, max_concurrency)` replace the local
  `subprocess` execution;
* `trigger_waiter` — the driver-side S3 watcher.

The default (`step_executor=None`) still calls the module-level `_execute_step`,
so local `--execute` behaviour and the tests that monkeypatch it are untouched.

The alternative — a new iterative interpreter with a serializable cursor — was
rejected: it would duplicate loop/branch semantics and the two implementations
would drift. A unit test pins the equivalence directly: for a fixed decision
sequence, the runtime submission order equals
`build_plan(assume_decision=<same>)`.

### Decision contract (consumed, not reinvented)

Decisions are read through the existing helpers:

* `decisions.refresh_context_decision(context, reader)` → `config.decision_uri`
* `decisions.load_decision` → `decision_from_payload` (`decision` /
  `last_decision` / `action`) → `normalize_decision`
* producers unchanged: `workbench.*.write_*_decision` toolRefs and
  `data_factory_stages.grade_gate` (which derives the decision from a **real** VLM
  eval score).

`RecordingDecisionReader` wraps the reader so every runtime gate read (URI, raw
body, normalized decision, timestamp) lands in the ledger — that is what makes a
live claim auditable after the fact.

Failure modes are deliberately asymmetric: a gate artifact that **does not exist
yet** falls back to the plan-time assumption (recorded in the ledger with
`source: assume_decision_fallback`), because that is the documented offline
behaviour; a gate artifact that exists but is **unreadable or malformed** fails the
run, because silently looping on corrupt JSON would be worse than stopping.

Two control-flow shapes fall out of the existing semantics:

* **Bounded loop with early exit** — `loop.until: promote_checkpoint` on a
  `sequence` whose member sets `writesDecision: true`. The decision is re-read
  from S3 after each iteration, so a promoting gate breaks the loop *before*
  submitting the remaining iterations.
* **Data-dependent branching** — `transitions: [{when: promote_checkpoint, goto: publish},
  {when: loop_back, goto: escalate}]` on a state outside a loop body. The runtime
  reads the artifact and follows the matching edge.
  (Inside a loop body, `follow_transitions=False` — the loop drives control flow;
  that is pre-existing engine behaviour, documented in the catalog's blueprints.)

### Durability, idempotency, resume

`RuntimeLedger` writes `npa.workflow.runtime.v1` to
`<config.prefix>/npa-workflow/runtime.json` through the existing
`RunStateStore` (reader/writer injection = unit-test seam), containing every wave
attempt (key, states, kind, job id/name, sky status, timings, per-task timeline,
concurrency observations, outputs, error), every decision read, and every trigger
watermark. `RunManifest` (`npa.workflow.run.v1`) is untouched.

Resume is **memoized replay**: `--resume` re-runs the traversal, and any wave
whose key is already recorded as `succeeded` is replayed from the ledger instead
of resubmitted. This works because the traversal is deterministic given the same
decision artifacts, which are re-read from the same S3 objects. Re-running a
completed run is therefore a no-op, and a run that failed at wave *k* resumes at
wave *k*.

Wave keys embed the loop label, state name, iteration and a monotonic sequence
number, so a loop body that runs three times produces three distinct keys.

**Determinism constraint (important).** Because the key carries that in-process
sequence number, replay is only sound while the traversal is deterministic. It is
deterministic when the inputs are: decisions are re-read from the same S3 objects
and trigger watermarks are already satisfied. If a gate artifact *changes* between
runs (a later run reads `promote_checkpoint` where the first read `loop_back`), the
graph legitimately diverges, keys shift from the divergence point onwards, and the
waves after it re-run. That is the correct outcome — the plan really is different —
but it means `--resume` is "continue this run", not "reproduce this run".

**Never resubmit work that may still be running.** A wave is recorded `running` the
moment it is submitted, so a driver that dies mid-poll leaves a job that may still
be billing. On `--resume`, such a record is *reconciled* rather than replaced: the
recorded job is queried and then adopted if it already succeeded, attached to (kept
polling) if it is still alive, or replaced only once it is observably terminal-failed.
If its state cannot be determined at all, the run fails instead of launching a second
copy. `--resume` without `config.bucket` (i.e. without a ledger) fails fast for the
same reason.

**Never leave a job running after an abort.** Every wave failure path — workflow
error, unexpected tooling error, `KeyboardInterrupt` — goes through `_abort_wave`,
which cancels the managed job (by id, or by cluster name when the submit reported no
id) before recording the failure. Status queries are treated as unreliable rather
than fatal: up to `MAX_CONSECUTIVE_STATUS_ERRORS` transient `sky jobs queue`
failures are tolerated (and recorded in the ledger) because a failed *query* says
nothing about the job, while the wave deadline still applies. A submit that reports
no job id is rejected outright, since polling would otherwise sit on `UNKNOWN` for
the whole deadline while the job ran.

### Failure, retry, cancellation

* A wave whose managed job reaches a terminal failure is retried up to
  `--retries` times with a backoff; each attempt is a separate ledger entry
  (`attempt: 1, 2, ...`) so a flaky node is visible rather than hidden.
* A wave that does not reach a terminal state within `--max-wait-seconds` is
  cancelled (job + cluster) unless `--no-cancel-on-timeout`, then fails the run —
  no leaked clusters.
* A failed batch of a parallel group stops the group; the remaining members are
  recorded as skipped, and the run fails with the **root cause** (the first
  failure), not the cascade.

### Local `--execute` vs runtime: one intentional difference

With the local executor a failed member of a `parallel:` group does **not** stop the
other members (they run in declared order, each recorded independently); with the
runtime executor a failed *batch* stops the group and the remaining members are
recorded as skipped. Both report the same root cause, and both fail the run. The
runtime behaviour is deliberate: continuing to launch cloud jobs whose barrier can
no longer be satisfied only spends money.

### Trigger / watch pattern

`trigger:` is polled **driver-side** (`s3_trigger_waiter`): list the prefix until
`minObjects` keys exist, record the watermark in the ledger, then submit the state's
wave. The wait is bounded twice — by `maxPolls` when the spec sets one, and *always*
by the run's `max_wait_seconds`, so the default `maxPolls: 0` cannot mean "wait
forever". No cloud job is burned to wait, and a
resumed run skips a watch it already satisfied. `sim_to_real_trigger`'s
SkyPilot-specific watcher is unchanged; this is the npa.workflow-native analogue.

### CLI surface

```
npa workbench workflow submit <spec> --runtime \
  [--resume] [--poll-seconds N] [--max-wait-seconds N] \
  [--retries N] [--max-concurrency N] [--no-cancel-on-timeout] \
  [--var k=v] [--secret-env NAME] [--registry ...] [--output-format json]

npa workbench workflow plan-spec <spec> --waves [--json]
```

`--runtime` runs the driver in the foreground and prints a JSON summary (status,
waves with job ids and timelines, decisions, run prefix, runtime-state URI);
without it, `submit` behaves exactly as before. A detached driver (submit and
poll from a supervisor process) is deliberately out of scope here — it needs a
process supervisor story of its own.

---

## 5. Backwards compatibility

| Guarantee | How it is held |
| --- | --- |
| Existing specs render identically | `build_plan` unchanged for specs without `parallel:`; `render_skypilot_yaml` untouched |
| Serial-only renderer guard preserved | Parallel uses a *different* function; `test_render_rejects_parallel_execution` passes verbatim |
| Plan-only guardrail ("`execution: serial`" in every rendered twin) | Parallel groups flatten for `--plan-only` |
| `--assume-decision` remains the offline path | Untouched in `build_plan` / `submit`; the runtime tier only uses it as a fallback when a gate artifact cannot be read |
| Local `--execute` behaviour | `step_executor=None` → module-level `_execute_step` (monkeypatch-compatible) |
| `RunManifest` schema | Unchanged; runtime state is a new, separate document |
| Scheduler seam | Both renderers build docs through `build_skypilot_task_doc` → `build_scheduler_task` |

### Deviations that were necessary (and why)

1. **`SubmitLiveCase` gained fields** (`runtime`, `config_vars`,
   `expected_parallel_tasks`, `image_tool`, `max_wait_seconds`) and the one-shot
   live test now skips `runtime=True` cases. An `expected_execution` field was
   considered and deliberately **not** added: because `--plan-only` always renders
   the flattened serial plan (D6), the plan-only guardrail's
   `"execution: serial"` assertion holds for parallel specs too, so the field would
   never carry a value other than `"serial"`. Reason: submitting a parallel sweep through the one-shot path would
   render the flattened serial plan — valid, but it proves nothing about
   concurrency and would burn four GPU-hours running a sweep serially. The
   plan-only matrix still covers every spec including the new ones.
2. **`workbench.insights.ingest_run` is not a universal barrier.** The first live
   fan-out run proved it: the insights ingester only recognises dataset,
   scenario and decision artifacts, so a caption fan-out made it fail with "no
   known manifest/report schemas found". The fan-out and sweep specs now use
   purpose-built join stages (`fanout_join.join_shards`, `rl_sweep.select_best`)
   that additionally *verify* every predecessor's artifact exists — a better
   demonstration of a barrier anyway. The gate-loop spec still uses
   `insights.ingest_run`, because `decision.json` *is* an ingestible artifact.
3. **The Isaac sweep uses `run.shell` + a real module, not
   `workbench.rl.policy_train`.** That toolRef's argv (`--learning-rate`,
   `--batch-size`, `--input-path`) does not match the actual
   `npa workbench isaac-lab train` CLI (`--override`, `--num-envs`, `--steps`,
   `--output-path`), and that CLI is a *launcher* (it provisions a VM/serverless
   job), so calling it inside a SkyPilot task would nest infrastructure. The
   in-pod contract is the upstream RSL-RL training script — exactly what the
   SkyPilot template this spec ports does. Logic lives in
   `npa/src/npa/workflows/rl_sweep.py` with unit tests, per the repo's
   "put testable logic in a real module" rule. Fixing the pre-existing
   `workbench.rl.policy_train` mismatch is out of scope for this change.
4. **Root-cause error reporting for parallel groups** — `_execute_state_machine`
   now raises the *first* failure of a group rather than the last, so the cascade
   of "skipped after batch N failed" records cannot mask the real error.
5. **The new fields are validated in Python, not by the JSON Schema.** The shipped
   `npa.workflow.v0.0.1.schema.json` gained entries for `parallel`,
   `maxConcurrency`, `params` and `trigger`, but the hand-rolled walker in
   `schema_validation.py` does **not** resolve `$ref` / `$defs` /
   `additionalProperties`, so `states.<name>.*` bodies have never actually been
   schema-enforced (pre-existing, unrelated to this change). Writing the tests for
   it exposed the consequence for the new fields — `parallel: shard-a` was iterated
   character by character and reported "duplicate parallel member" — so
   `_parse_state` now rejects a non-list `parallel` and non-string members with
   actionable messages, alongside the existing `params`/`trigger` mapping checks.
   Teaching the walker to resolve `$ref` would retroactively tighten validation for
   every shipped spec and is deliberately left out of this change.

---

## 6. Testing model

* **Unit (mocked, default suite).** `test_parallel_waves.py` (spec validation,
  params overlay, wave folding/batching, JobGroup rendering, serial guard),
  `test_runtime_orchestrator.py` (early exit, full budget, `goto` branch,
  plan/runtime equivalence, JobGroup batching, barrier stop on failure, retry,
  retry exhaustion, timeout→cancel, resume/replay, trigger poll and give-up,
  ledger contents), `test_rl_sweep.py`, `test_fanout_join.py`. Every dependency
  (submitter, status, timeline, canceller, sleeper, clock, S3) is injected — no
  infrastructure, no sleeping.
* **Live (separate tier, env-gated).** `NPA_E2E_NPA_WORKFLOW_RUNTIME=1` plus the
  existing `NPA_INTEGRATION_E2E` / `NPA_E2E_NPA_WORKFLOW_SUBMIT` gates. Two live
  tests: terminal success + concurrency + barrier for `runtime=True` matrix
  cases, and the gate-loop early-exit vs full-budget pair. Runner:
  `scripts/npa-workflow-runtime-live-e2e.sh`.

Live results, run ids, job timelines and decision artifacts are in
[`EVIDENCE.md`](EVIDENCE.md).

---

## 7. Deliberately out of scope

* Multi-step branches inside a `parallel:` group (each member is a leaf today;
  a branch would need one managed job per branch instead of one JobGroup).
* `gang` scheduling and `foreach` templating.
* A detached/daemonized runtime driver and a unified `workflow status` view for
  runtime runs (the ledger JSON is the source of truth today).

---

# DESIGN — retiring the raw SkyPilot task catalog

`npa/src/npa/workflows/skypilot/` shipped 36 raw SkyPilot task templates alongside
the `npa.workflow/v0.0.1` catalog. They are being retired so that a spec is the
**only** workflow authoring surface. This section records the decisions that work
depends on; the live runs backing each one are in [`EVIDENCE.md`](EVIDENCE.md).

SkyPilot itself is **not** being retired: it remains the execution engine, and
`detect_submit_format()` still accepts a customer's own SkyPilot YAML. What goes
away is the shipped *catalog* of templates.

## R1. What "equivalent twin" actually means (the finding that shaped everything)

13 specs declared `skypilotTwin:`, which reads as "this spec replaces that
template". Two of those twins turned out **not** to be equivalent, and only a live
run could show it:

| Twin | Looked fine | Failed live because |
| --- | --- | --- |
| `sonic-export` | validates, plans, renders | the tool only accepted **local** paths; the template did the S3 download/upload in ~60 lines of inline bash. `toolRef` argv has no such escape hatch, so the spec handed `s3://.../checkpoint.pt` to `Path.exists()` |
| `sonic-eval` / `sonic-export-eval` | same | `sonic eval` could *write* its result to S3 but only ever *read* local files |

The general rule this exposes: **a template that does I/O staging in bash cannot be
replaced by a `toolRef` unless the tool itself speaks object storage.** The fix is
therefore in the tool, not the spec: `export_onnx` and `evaluate_onnx_policy` now
stage `s3://` inputs and publish `s3://` outputs through the same `StorageClient`
the eval result path already used. Local behaviour is unchanged — the historic
export body moved to `_export_onnx_local` and is called directly when nothing is an
object URI, so a local run never constructs a storage client.

Two smaller traps fell out of the same run, both now covered offline:

* An ONNX is a **pair** of files when `torch.onnx.export` spills tensors to
  `<name>.onnx.data`, and onnxruntime resolves that path *relative to the model*.
  Staging must move the sibling too.
* The exporter resolves shapes from the policy when no `--obs-spec` is given, so a
  fixture policy must expose `obs_dim` / `action_dim`.

## R2. Three-tier contract: the third tier moves onto npa.workflow

`test_three_tier_contract.py` bound 11 CLI/SDK capabilities to a **SkyPilot YAML
`envs` block**. Deleting those YAMLs would delete the guardrail, which is not
allowed, so the tier was migrated rather than dropped:

| Tier | Before | After |
| --- | --- | --- |
| 1 | Typer callback parameter + flag | unchanged |
| 2 | SDK signature parameter | unchanged |
| 3 | `envs:` key declared **and referenced** in the YAML | the shipped spec loads, declares the `toolRef`, and the toolRef's `argv_template` passes the parameter's **real CLI flag** |

*Sharper*, because the old check could not tell whether the flag existed: a YAML
could declare `TRAIN_LEARNING_RATE` while the tool was invoked with a nonexistent
`--learning-rate`. *Narrower*, because a catalog argv exposes fewer knobs than a
YAML `envs` block did. Rather than hide that, every contract pins
`spec_gap` — the CLI parameters a spec author cannot set today — and
`SPEC_GAP_REASONS` classifies each one:

* `boolean` — the option is a paired/flag-only boolean (`--headless/--no-headless`).
  A v0.0.1 argv template is a fixed list with no conditional rendering, so it cannot
  express one. Closing these needs a spec-level conditional-argv feature.
* `infra` — the option selects infrastructure (`--image`, `--gpu-type`). The engine
  already owns image/accelerator selection via `resources.<profile>`; passing it
  again inside the pod would nest infrastructure choices.
* `knob` — a plain value the argv simply does not pass yet. These are the ones worth
  closing, tool by tool, with a live run each.

The test asserts the computed gap equals the declared gap **exactly**, so the gap is
visible in review, cannot grow silently, and shrinks only by a deliberate edit.
`workflow/trigger/{run,watch}` keep the legacy YAML tier until the sim-to-real port
gives them a spec twin; `LEGACY_YAML_TIER` pins that set so it can only shrink.

## R3. A catalog-wide argv guardrail replaces what the YAML tier used to imply

Nothing checked that a `toolRef`'s argv flags exist on the command they are handed
to, so a template could validate, plan and render and still die with `No such
option` in the pod. `guardrails/tool_catalog_argv.py` walks the `npa` Typer tree
along an argv template and reports flags the resolved command rejects;
`test_no_tool_ref_argv_passes_a_flag_its_cli_rejects` asserts **zero** drift across
the whole catalog.

It immediately found two: `workbench.rl.policy_train` (`--learning-rate`,
`--batch-size`, `--input-path` — the half `DESIGN §7` already knew about) and
`workbench.rl.evaluate_policy` (`--episodes`, previously unknown). Both are fixed
rather than allowlisted, so three specs that could never have run now can. Trainer
hyper-parameters go through Isaac Lab's repeatable Hydra `--override KEY=VALUE`,
which is what its CLI documents for exactly this.

## R4. Per-toolRef npa extras, instead of requiring a vendor image

A stage on SkyPilot's default image gets only the base `npa` install, so
`npa workbench sonic export` failed with its own advice: *"requires torch ... or use
the npa[sonic] extra"*. `TOOL_REF_PIP_EXTRAS` maps a toolRef prefix to an npa extra
and `render_setup_for_tool` installs it **from the same source tree it installed npa
from** (recorded in `/tmp/npa-src-root`). This is the pattern the renderer already
used to install vLLM for self-hosted `vlm_eval`.

Consequence: the SONIC specs need no vendor image at all. The alternative —
requiring every workbench image to be SkyPilot-hostable before its spec can run —
is still worth having (see R5) but is no longer a *blocker* for a spec to be live.

Rendering constraint worth knowing: the setup script must not contain `${var}`.
`assert_no_unresolved_placeholders` rejects any braced expansion, because SkyPilot
would leave it literal — the first live submit failed in 1.15 s on exactly that.
Compose paths with `printf` instead.

## R5. Image hostability is a repo deliverable, not operator lore

`EVIDENCE §8.5` listed "make the shipped Isaac image SkyPilot-hostable" as a
follow-up, and the SONIC image had the same defect. `SKYPILOT_HOSTED_IMAGES` in the
image guardrail now covers `sonic` as well as `isaac-lab`, the SONIC Dockerfile
carries the same four ingredients, and each has a `Dockerfile.k8s-prereqs` so an
operator can repair an already-published tag in-cluster. The list grows as the
catalog shrinks: **once a tool's only workflow surface is a spec, its image must be
able to host a SkyPilot task.**

## R6. Live fixtures that do not depend on gated weights

`sonic export` needs a loadable torch policy checkpoint, and the repo deliberately
does not vendor NVIDIA's gated `nvidia/GEAR-SONIC` weights, so those twins could
only ever be covered plan-only. `npa.workflows.sonic_fixture` builds a small
deterministic MLP policy the shipped exporter can trace, and
`scripts/stage-sonic-export-fixture.sh` runs it **in-cluster** through a ConfigMap —
no torch download touches the dev VM, which routinely sits above 90 % disk. The
harness seeds from `NPA_E2E_SONIC_CHECKPOINT_SRC` / `NPA_E2E_SONIC_ONNX_SRC`,
mirroring the existing `NPA_E2E_SONIC_MOTION_SRC` hook.

## R7. A machine-checked retirement tally

`test_skypilot_catalog_retirement.py` pins the exact remaining templates with a
reason each. A new raw template cannot appear without a deliberate edit, and each
retirement shows the count dropping in a one-line diff instead of a prose claim in a
PR body. Deleting an entry is the **last** step: the twin must already have a live
run id in `EVIDENCE.md`.

## R8. Harness parsing must not hide a live result

`json.loads(result.output)` in the live harness assumed the CLI writes nothing but
JSON. `CliRunner` merges stderr into `output` on this click version, and the submit
path legitimately writes `Hint: consider --secret-env NGC_API_KEY` first — so a
**successful** submit was reported as `JSONDecodeError: Expecting value: line 1
column 1`, and the real failure downstream stayed invisible for two runs. Both
parsers now slice the JSON document out of the stream, and a matrix guardrail
asserts every case declares the secrets its plan hints at so the advisory line is
not emitted at all.

## R9. `num_nodes`: multi-node stages on the resource profile

A spec could not ask for a multi-node block at all — that capability lived only in
`npa burst submit --nodes` (`burst.core.BurstSpec.num_nodes`), i.e. outside the workflow
surface. Closing it needed one decision: **where does the field go?**

SkyPilot places `num_nodes` at the **task** level, a sibling of `resources`
(`sky/utils/schemas.py`; `burst.core.build_task_spec` does the same). A spec has no
task-level surface — per-stage shape lives on the resource profile a state selects — so
the field is declared there and the renderer lifts it back out:

```yaml
resources:
  gang:
    cloud: kubernetes
    cpus: 2
    memory: 4Gi
    num_nodes: 2          # -> task-level `num_nodes: 2`, NOT inside `resources`
```

`normalize_resources()` deliberately does not pass it through; inside a `resources` block
it would be invalid. `build_scheduler_task` carries it so the portable seam stays
complete for a non-SkyPilot backend.

**Additive by construction.** A profile without `num_nodes` (or with `1`) emits no key,
so every previously rendered document is byte-identical — asserted, not assumed.
`validate_spec` rejects a non-integer, a bool, `< 1`, and `> MAX_PROFILE_NODES` (32): a
gang block that large would sit `PENDING` on a shared cluster rather than fail fast, which
is the worst failure mode to debug.

Rejected: a `gang:` state field (DESIGN §7 lists `gang` as out of scope, and a state that
selects a 2-node profile already *is* the gang); and a submit-time `--num-nodes` flag
(node count is a property of the stage, not of one invocation — unlike
`--max-concurrency`, which is a cost cap).

The live proof is `npa-workflows/multi-node-probe.yaml`: the gang stage writes one
`rank-<n>.json` per node and a single-node stage fails unless it finds one report per
expected rank **from distinct hostnames** — so a gang that collapsed onto one node cannot
pass. See `EVIDENCE.md` §R14.

## R10. BYOF resource profiles are not workflow templates

`isaac-lab-rl-train*.yaml` and `byof-*-smoke.yaml` describe a pod shape (accelerator,
cpu/memory floors, image placeholder, smoke command) that the BYOF runner substitutes
into — one task, no orchestration. The workflow surface for them is already the spec
`byof.yaml`, whose `workbench.byof.repo` toolRef passes one through
`--yaml {{config.resource_profile_yaml}}`.

So they were **relocated** to `npa/src/npa/workflows/byof/profiles/` rather than ported or
deleted, joining `byof-solution-smoke-rtxpro-gpu.yaml` and
`skypilot-kubernetes-rtxpro.yaml` which were already there — a move `byof/live.py`'s own
comment had anticipated ("so the SkyPilot reference catalog can be deprecated/removed
without breaking BYOF live runs").

Rejected: rewriting `run_isaac_lab_rl.py` / `run_byof_datagen.py` /
`run_byof_container_verify.py` onto `prepare_npa_workflow_for_submit`. They carry
render-only modes, output-root rewriting and BYOF image plumbing the engine does not
model, so the port would risk the BYOF onboarding live path for no gain in the *workflow*
surface.

`test_byof_profiles.py` keeps the boundary honest: the file set is pinned, each profile
must be a **single** task (chaining stages means it belongs in a spec), `live.py`'s
constants must resolve inside `profiles/`, and no runner may resolve a path under the
retiring catalog.

## R11. A per-`toolRef` **run** preamble, not just a setup hook

R4 added a per-`toolRef` `setup:` hook so a stage could install an npa extra. The
self-hosted VLM backend needed something the setup hook structurally cannot do: **start a
process that must still be alive while the stage's command runs.** SkyPilot executes `setup:`
and `run:` as separate shells, so anything backgrounded in setup is gone by the time the
command starts.

So `render_run_preamble_for_tool(tool_ref, config)` mirrors `render_setup_for_tool` and is
prepended to the `run:` script — after the interpreter shim, so the server uses the same
`python3` the command will.

Why this belongs in the renderer rather than in the tool: the tool's job is to *call* an
OpenAI-compatible endpoint, and it should not care whether that endpoint is hosted, remote or
in-pod. Where the endpoint comes from is a property of the **execution environment**, which is
exactly what the renderer owns. Rejected alternatives:

* *Bake vLLM serving into `vlm-eval run`.* Couples a scoring tool to a serving stack, and
  would start a server per invocation — the loop would pay it three times.
* *Require a prebuilt serving image.* That is what the template did (`NPA_VLM_IMAGE`), and it
  is precisely the operator lore R5 argues against: the live proof runs on SkyPilot's default
  image.
* *A generic `sidecar:` field in the spec schema.* Real, but much larger, and it would put
  serving mechanics into the authoring surface. Revisit if a second tool needs a companion
  process.

The preamble also fixes the environment's shape rather than demanding a shape: `ninja` and a
CUDA compiler both come from pip wheels vLLM already depends on, and the JIT-dependent sampler
falls back to pure PyTorch. A workflow engine must not require a build toolchain in a task
image.

## R12. Retiring a template may require *adding* a capability

The plan (D3) assumed `sim-to-real-loop.yaml` could retire because the staged sim2real engine
covers sim-to-real. Reading the template disproved that. It owns two things nothing else has:
per-rollout scoring of a rollout **set** (`vlm-eval run` scores one rollout — it discovers
frames recursively, so a prefix of many rollouts blends into a single score), and the
`task_success_report.json` aggregate that the cookbook documents as the gate.

The capability existed twice and was reachable from neither the CLI nor a spec: as `jq` inside
the template, and as Python inside a gated GPU test. That is the signature of a missing tool
feature, and it generalises: **a template's bash is either accidental glue or an unimplemented
capability, and only reading it tells you which.** The same reading is what produced the sonic
S3 staging work (R1) and the two `detection-training` gaps.

Consequence for the remaining work: a retirement is not a delete plus a reference sweep. The
honest order is (1) read the template, (2) name every behaviour it owns, (3) put the ones the
tool lacks into the tool, (4) verify live, (5) delete. Where step 3 is not finished, the
template stays and the tally records what is missing — which is why `bdd100k-pipeline.yaml`
survives with three named gaps rather than a half-ported runner.

## R13. Two failure classes worth encoding as guardrails, not fixes

Both were found by reading twins rather than by a test, and both are *classes*:

**A spec may not name a path inside the repo checkout.** `vlm-eval-benchmark.yaml` passed
`--dataset npa/src/npa/workbench/vlm_eval/fixtures/.../benchmark.json`. It resolves on a
laptop and never in a pod. The guardrail fires only when a resolved argv value is a *relative*
path that also *exists in this repo* — deliberately narrow, because such a value is always a
mistake and the check then cannot produce false positives. It immediately found five more
specs, which is the argument for a guardrail over five fixes.

**A same-named spec is not automatically a twin.** `tokenfactory-rollout-judge` has a spec and
a template that share a name and do different things (the template's GPU stage *produces* what
its judge scores; the spec's first stage is an unrelated reasoner). Equivalence cannot be
inferred from the file name, and the retirement tally now carries the *reason* a template
survives rather than a status word — so "twin live-verified" cannot stand in for "I compared
them".

## R14. Executing a *plan step* is a better validation than executing a template's bash

`bdd100k-pipeline.yaml` shipped with something none of the other templates had: a
`--mock-endpoints` mode in its runner that stood up in-process LanceDB and
detection-training servers, ran every task's `run:` bash against them, and asserted the exact
request sequence. It was the closest thing in the repo to an end-to-end test of a pipeline
without infrastructure.

Porting the runner onto the spec forced a choice about that mode, because a spec has no bash
to run. Executing **each plan step's resolved argv** turns out to be strictly better:

* it is *exactly* what the engine will run in a pod, so a passing mock run is evidence about
  the real execution path rather than about a parallel bash implementation;
* it exercises the toolRef → CLI boundary, which is where this change kept finding defects;
* it removes the drift risk of a second implementation of the pipeline.

It proved itself immediately: the first spec-driven mock run failed on
`No such option '--table'` from `lancedb create-mv` — a toolRef defect that had been
invisible because the flag audit skipped `bash -c` entries — and then on a declared artifact
URI missing a path separator. Neither is visible in a diff, and neither needed a GPU, a
cluster or a credential to find.

The generalisation for the remaining ports: **where a template's runner has a
no-infrastructure validation mode, port that mode to plan steps before porting the submit
path.** It is the cheapest place to learn that a spec is wrong, and for a pipeline whose live
run is blocked on a missing service it is the only honest offline proof available.

The check should assert *order*, not just counts. `--wait` and `--discover-checkpoint` are
both invisible to a POST-count assertion: what distinguishes them is that a `GET /status`
follows every `POST /train`, and a `GET /runs` precedes every `POST /eval`.
