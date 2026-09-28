# Matched Astra and routed specialist repair benchmark

This benchmark compares Astra alone with Astra coordinating concurrent Token
Factory specialists through Workbench's LangGraph runtime. It replays three
historical Workbench regressions: physical-evidence validation, transactional
dataset publication, and simulation-to-LeRobot input validation. Agents edit
real source and operate native simulation/export/verification tools.

These are known regressions from a public ancestor, not newly discovered bugs.
Current source provides the calibrated reference. Candidate changes stay in
disposable copies. No private repository or routing service is required.

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
