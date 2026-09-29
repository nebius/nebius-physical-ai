# Matched Astra and routed specialist repair benchmark

This benchmark compares Astra alone with Astra coordinating concurrent Token
Factory specialists through Workbench's LangGraph runtime. It replays three
historical Workbench regressions: physical-evidence validation, transactional
dataset publication, and simulation-to-LeRobot input validation. Agents edit
real source and operate native simulation/export/verification tools.

These are known regressions from a public ancestor, not newly discovered bugs.
Current source provides the calibrated reference. Candidate changes stay in
disposable copies. No private repository or routing service is required.

The [September 27 measured report](../../../../docs/workbench/specialists-matched-repair-experiment.md)
retains all six arms, including recovery costs and mixed timing results. It also
records a delegation-lock fix made after the measured snapshot was frozen.
The [September 28 coordination report](../../../../docs/workbench/specialists-coordination-experiment.md)
adds two fresh cohorts, a real cross-process contention test and timing for
coordinator turns, model-free waits, overlapping workers and final verification.
It distinguishes the enabled native-failure handoff policy from actual usage:
no failure triggered a model handoff in those runs.

## Fixed comparison

[`protocol.json`](protocol.json) declares three counterbalanced pairs before
inference. Each arm receives identical requirements, starting source, immutable
checks, native inputs and operation grants. Both can run operations concurrently.
Astra uses the same model and effort in both arms. The hybrid delegates through
a real Astra call; Lightning selects GLM Flash or full GLM, and LangGraph manages
workers and their tool loops. Completion receipts allow the host to wait without
model polling. Astra recovery calls count if needed.

Each lane must pass all 128 focused checks and a real three-case MuJoCo Fetch
workflow. The host combines the three patches, repeats the checks, and runs a
six-case workflow. Independent replay verifies 1,110 physics transitions and
both camera streams. Native LeRobot loading must yield four accepted episodes,
740 timesteps and 1,480 aligned camera samples; two failed-grasp controls remain
outside the training dataset. Agents cannot edit tests or verifiers.

Timing starts before coordinator launch and ends after combined verification.
Every failed attempt, route, fallback and model response counts. The scorer
requires all six declared arms to pass with reconciled usage before supporting
equal quality, lower median latency and lower median total model cost. Three
pairs provide a small repeated benchmark, not a general ranking of models.

## Prepare on Linux

Use an isolated checkout with the historical commit available. Install
`npa[specialists]` and repository test requirements in its own `npa/.venv`.
Install `bubblewrap`, `ffmpeg` and `libosmesa6`. Prepare a simulation interpreter
with `npa[robot-sdg,adapter]`, and a separate CPU-only reader interpreter with
`lerobot==0.5.1`. See the [native prerequisites](../robot_workflow/README.md).
The sandbox must be permitted to create user and network namespaces.
Python mounts include the selected environment and its actual runtime prefix;
an interpreter layout that would expose the account home or filesystem root is
rejected. Temporary files stay on a sandbox-private scratch filesystem.

Authenticate Codex and configure Token Factory using Workbench's normal
credential mechanism. Confirm the exact model IDs in `protocol.json` are
available. Keep credentials and run evidence outside Git. The scripts add no
time, cost, or job budget. `--live` starts paid inference; preparation and
calibration make no model calls.

```bash
export BENCHMARK_ROOT="$HOME/.npa/operations/my-new-repair-benchmark"
npa/.venv/bin/python npa/examples/specialists/repair_benchmark/prepare.py \
  --output "$BENCHMARK_ROOT" \
  --native-python "$SIMULATION_PYTHON" --reader-python "$LEROBOT_READER_PYTHON"
npa/.venv/bin/python "$BENCHMARK_ROOT/harness/run.py" calibrate \
  --root "$BENCHMARK_ROOT"
```

Calibration requires the correct reference to pass, every historical lane to
fail with ordinary regression failures, and reference native verification to
pass. Interpreter paths preserve virtualenv symlinks. Preparation refuses
missing dependencies and freezes source, prompts, matrices and tests by hash.
Do not change these files or interpreter dependencies during a measured run.

## Capture usage and run

In another terminal, start the loopback-only telemetry receiver. Leave it
running until all measured coordinator processes finish.

```bash
npa/.venv/bin/python "$BENCHMARK_ROOT/harness/telemetry.py" \
  --directory "$BENCHMARK_ROOT/telemetry" --codex "$(command -v codex)"
```

Read the endpoint from `ready.json`, then run and score:

```bash
export NPA_CODEX_OTEL_ENDPOINT=$(npa/.venv/bin/python -c \
  'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["BENCHMARK_ROOT"])/"telemetry/ready.json").read_text())["endpoint"])')
npa/.venv/bin/python "$BENCHMARK_ROOT/harness/run.py" run \
  --root "$BENCHMARK_ROOT" \
  --codex-wrapper "$BENCHMARK_ROOT/telemetry/bin" \
  --otel-endpoint "$NPA_CODEX_OTEL_ENDPOINT" \
  --telemetry-records "$BENCHMARK_ROOT/telemetry/events.jsonl" \
  --prices "$BENCHMARK_ROOT/harness/prices.json" --live
npa/.venv/bin/python "$BENCHMARK_ROOT/harness/score.py" \
  --root "$BENCHMARK_ROOT" --output "$BENCHMARK_ROOT/results.json"
```

Stop the receiver with Ctrl-C after scoring. A failed arm remains in the
sequence. An uncertain native process stops further arms until an operator
reconciles its durable state; rerunning never silently repeats paid inference.
Start a new root for a new declared experiment, retaining the old result. Run
arms serially on the same host and avoid unrelated heavy work during timing.
Model workers within a hybrid arm execute concurrently.

Pass `--workspace-layout independent-checkout` during preparation to give every
lane in both arms its own complete detached Workbench checkout. Native operations
import that lane's source tree; they cannot see another lane's edits or outputs.
Each checkout's runtime source is copied from the same frozen working-tree
reference, including any operator changes present during preparation.
Only the assigned repair file is writable through the agent tools. Tests and
verifiers remain immutable, and the final integration check combines all patches
against the neutral frozen reference. The default `overlay` layout keeps a shared
read-only reference with separate candidate files.

The checkouts share a local Git object store and read-only Python dependencies,
but their working files and operation outputs are independent. Per-profile
ownership locks still prevent duplicate workers and conflicting takeovers of the
same agent. This tests filesystem isolation on one host, not separate VMs or cloud
job isolation. Checkout preparation is outside measured task latency; record its
time and disk use separately. Do not compare a new cohort with an older one as a
causal test of the layout itself.
Frozen-input audits before each arm also sit outside the task timer. Report the
between-arm gaps and whole-sequence elapsed time alongside per-task results;
full-checkout integrity checking is part of running the experiment.

## Cost and evidence

[`prices.json`](prices.json) records dated public API-equivalent rates. Refresh
it before preparing a future benchmark. The scorer reconciles all five Astra
CLI token counters with per-request telemetry, applies context tariffs per
request, and charges cached tokens once. An initial input-only response absent
from CLI totals remains a zero-to-full-price uncertainty interval. Unknown or
inconsistent usage cannot support a savings claim.

All Token Factory router, worker, rejected-response and fallback usage is
included at full input tariff; no prefix-cache discount is assumed. Prices are
standard API equivalents, not a verified Codex subscription charge or invoice.
Development conversation, installation, calibration and CPU hosting/storage
costs are excluded and explicitly identified. Savings compare recurring
measured model work, not total project expenditure.

The receiver retains permitted numeric counters, fixed enums and hashed
identifiers only. It does not persist raw telemetry bodies, prompts or headers.
Other private evidence includes model conversations, source snapshots, every
diagnosis/native attempt, stdout/stderr, artifacts and content hashes. The
scorer exports source diffs, numeric verification receipts and usage summaries.
Review and scan that export before publication; model-authored diffs are still
untrusted text. Keep exact host paths and credentials in private operator storage.

## Explain coordination time

The scorer also derives timing from the retained coordinator and worker receipts.
Coordinator-process time and the subsequent combined verification partition total
elapsed time. Nested measurements show Astra turns, host waiting without model
calls, worker shutdown, classifier requests, model intervals and tool activity.
Worker intervals overlap; their summed durations are work performed, not additional
elapsed time. Model intervals run from context-view journaling to response
journaling and include client preparation and validation, not just provider time.

Delegation counts distinguish a confirmed pre-submission lock refusal from an
uncertain or rejected call. A recovered refusal requires the same specialist,
task ID and goal in the later accepted request. Missing boundaries remain
incomplete, and missing receipts never imply zero overhead. This analysis can
be applied to older retained runs without changing their frozen inputs or costs.

Reproduce the separate-process lock check without provider calls:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/agent_eval/test_specialist_workflow_experiment.py::test_process_lock_refusal_retries_one_durable_assignment -q
```

It holds the real profile lock, verifies that refusal queues nothing, then
releases ownership and verifies that identical retries create one durable task.

For a separate routing-policy experiment, prepare a fresh root with
`--native-failure-handoff`. This records the variant in the frozen protocol and
enables the existing `handoff_on_failure` policy only on the native `wait`
operation. Initial diagnosis failures are expected and do not change models.
A failed native wait passes its receipts and current files to the next configured
endpoint; the runtime does not replay the operation. Unresolved native effects
still block resubmission. Include the first worker's failed attempt and the
backup's usage in the comparison. If no failure triggers this policy, the run
provides no evidence that escalation helped.
