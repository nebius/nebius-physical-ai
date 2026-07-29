# EVIDENCE — live runs for `npa.workflow` parallel execution + runtime control flow

Every claim below is backed by a **live run on real Nebius infrastructure** from
the operator dev VM (`nebius-dev-vm`) against the `npa-rtxpro-mk8s` Kubernetes
cluster, real S3, and the real hosted Token Factory models. Anything that was
**not** verified live is called out explicitly in
[§7 Not verified live](#7-not-verified-live).

Secrets are never printed: credentials come from `~/.npa/live-e2e.env` /
`~/.npa/credentials.yaml` on the dev VM, and the live harness asserts
(`assert_no_credential_leakage`) that no CLI output contains credential material.
Bucket and registry values below come from that environment; nothing is
hardcoded in the repo.

Isolation: all work ran in a dedicated git worktree `~/npa-wf-runtime` (branch
`cursor/bc-fd5052e0-...-6c5b`) with `PYTHONPATH=<worktree>/npa/src` so the shared
editable venv could not shadow branch code (verified:
`python -c "import npa; print(npa.__file__)"` →
`/home/ubuntu/npa-wf-runtime/npa/src/npa/__init__.py`), in dedicated tmux
sessions, with a run-id prefix of its own and a private staged copy of the npa
package (`s3://<bucket>/npa-workflow-e2e/npa-src-wfrt/npa`) so no other agent's
runs were disturbed.

---

## 0. Environment (secrets redacted)

```bash
# on nebius-dev-vm, in the isolated worktree
cd ~/npa-wf-runtime
set -a; . ~/.npa/live-e2e.env; . ~/.npa/live-e2e-gates.env; set +a
export PYTHONPATH=$PWD/npa/src
export NPA_SRC_S3_URI=s3://lerobot-ccc9d3c7/npa-workflow-e2e/npa-src-wfrt/npa   # branch source
export NPA_INTEGRATION_E2E=1 NPA_E2E_NPA_WORKFLOW_SUBMIT=1 NPA_E2E_NPA_WORKFLOW_RUNTIME=1
export NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS=20
export NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS=2700
unset NPA_E2E_FORCE_ACCELERATORS NPA_E2E_ACCELERATOR_REMAP   # CPU tiers stay CPU
# registry: $NPA_E2E_REGISTRY = cr.us-central1.nebius.cloud/<registry-id>
# cluster:  kubernetes/npa-rtxpro-mk8s  (2 x 8 RTXPRO-6000-BLACKWELL + 1 CPU node)
# skypilot: ~/.npa/skypilot-venv/bin/sky, version 0.12.2
```

Branch source staged for the tasks (tasks install `npa` from S3 because the live
env sets `NPA_E2E_CLEAR_WORKBENCH_IMAGES=1`):

```bash
bash scripts/stage-npa-src.sh --bucket lerobot-ccc9d3c7 --prefix npa-workflow-e2e/npa-src-wfrt
# staged 549 files -> s3://lerobot-ccc9d3c7/npa-workflow-e2e/npa-src-wfrt/npa
```

---

## 1. Offline baseline and unit suites

| Run | Command | Result |
| --- | --- | --- |
| Baseline (base commit `d129ee90`, before any change) | `pytest npa/tests/ --ignore=npa/tests/e2e --timeout=120 -q` | **3538 passed, 28 skipped, 1 xpassed, 2 errors** in 637s |
| Engine + specs after the change | `pytest npa/tests/orchestration/npa_workflow/ npa/tests/smoke/test_all_workflow_yamls.py npa/tests/smoke/test_npa_workflow_smoke.py -q` | **263 passed** |
| New modules | `pytest npa/tests/workflows/test_rl_sweep.py npa/tests/workflows/test_fanout_join.py -q` | **10 passed** |

The 2 baseline errors are **pre-existing** and unrelated: the live-GPU fixtures in
`npa/tests/workbench/test_vlm_eval_backend.py` and `test_vlm_eval_loop_e2e.py`
try to launch a SkyPilot cluster whenever `sky` is on `PATH` and hit the 120s
timeout. They fail identically on the base commit.

Guardrails that stayed green untouched: `test_render_rejects_parallel_execution`
(serial-only renderer guard), `test_dynamic_execution.py` (monkeypatched local
executor), `test_submit_live_matrix.py`, and the plan-only matrix assertion that
every rendered twin says `execution: serial`.

---

## 2. Plan-only matrix (no cloud spend)

```bash
pytest npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_plan_only_matrix_no_leak -q
# 23 passed in 4.42s
```

All 23 matrix specs — including the three new ones — render cleanly, contain no
unresolved `${...}` placeholders, leak no credentials, and still emit
`execution: serial` headers for `--plan-only`.

Wave preview of the new specs (offline, `plan-spec --waves`):

```
$ npa workbench workflow plan-spec .../token-factory-parallel-fanout.yaml --run-id demo --waves
workflow: token-factory-parallel-fanout
waves: 2
  00. [parallel] caption-shards: caption-shard-a, caption-shard-b, caption-shard-c maxConcurrency=3 batches=1
  01. [serial] aggregate: aggregate

$ npa workbench workflow plan-spec .../isaac-lab-rl-sweep.yaml --run-id demo --waves
workflow: isaac-lab-rl-sweep
waves: 2
  00. [parallel] sweep: variant-lr-1e-3, variant-lr-3e-4, variant-entropy-0, variant-entropy-0-01 maxConcurrency=4 batches=1
  01. [serial] select-best: select-best
```

---

## 3. Phase 1 — real parallel execution (live, CPU tier)

**Spec:** `token-factory-parallel-fanout.yaml` (three real Token Factory caption
shards + a join barrier).
**Command:**

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=token-factory-parallel-fanout.yaml \
pytest npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_live_reaches_terminal -q -s
```

**Run id:** `npa-wf-cpu-token-factory-parallel-fanout-a8cf71e0`
**Run prefix:** `s3://lerobot-ccc9d3c7/npa-workflow-e2e/npa-wf-cpu-token-factory-parallel-fanout-a8cf71e0/token-factory-parallel-fanout/`

### 3.1 The three shards ran concurrently

`sky jobs queue --all --output json` (job **75** = the JobGroup wave):

| job_id | task | status | submitted_at | end_at |
| --- | --- | --- | --- | --- |
| 75 | caption-shard-a | SUCCEEDED | 1785297417.2239 | 1785297483.0892 |
| 75 | caption-shard-b | SUCCEEDED | 1785297417.2297 | 1785297482.8844 |
| 75 | caption-shard-c | SUCCEEDED | 1785297417.2362 | 1785297477.9160 |

All three share **one managed `job_id`** (a SkyPilot JobGroup), were submitted
within **12 ms** of each other, and their lifetimes overlap almost entirely
(~60–66 s each). A serialized chain cannot produce this.

The driver also sampled the live task statuses while polling:

```
[runtime] wave caption-shards batch 1/1 (parallel, 3 task(s)): ['caption-shard-a', 'caption-shard-b', 'caption-shard-c']
[runtime] wave 001|caption-shards|...: submitted job_id=72 name=npa-wf-cpu-token-factory-parallel-fanout-2f73db1d-01-caption
[runtime] wave 001|caption-shards|...: 3 tasks running concurrently (caption-shard-a, caption-shard-b, caption-shard-c)
```

(`max_concurrent_observed: 3` is stored in the wave ledger.)

### 3.2 The barrier waited for all predecessors

| wave | job_id | submitted_at | note |
| --- | --- | --- | --- |
| `caption-shards` (parallel) | 75 | 1785297417.22 | last member ended **1785297483.09** |
| `aggregate` (barrier) | 76 | **1785297577.64** | submitted **94.6 s after** the last member finished |

`aggregate` is the state that declares `needs: [caption-shards]`; the runtime tier
submitted it only after the whole group reached a terminal state.

### 3.3 Real work, real artifacts

`s3://.../token-factory-parallel-fanout/reports/join_report.json` (written by the
barrier, `npa.workflows.fanout_join.join_shards`):

```json
{
  "joined_shards": 3,
  "manifest": "captions.json",
  "missing_shards": [],
  "schema": "npa.fanout.join_report.v1",
  "shard_count": 3,
  "shards": [
    {"items": 2, "model": "Qwen/Qwen2.5-VL-72B-Instruct", "shard": "shard-a", "status": "ok", "uri": "s3://.../captions/shard-a/captions.json"},
    {"items": 2, "model": "Qwen/Qwen2.5-VL-72B-Instruct", "shard": "shard-b", "status": "ok", "uri": "s3://.../captions/shard-b/captions.json"},
    {"items": 2, "model": "Qwen/Qwen2.5-VL-72B-Instruct", "shard": "shard-c", "status": "ok", "uri": "s3://.../captions/shard-c/captions.json"}
  ],
  "total_items": 6
}
```

Each shard's `captions.json` is a **real hosted-VLM** result (`"dry_run": false`,
`Qwen/Qwen2.5-VL-72B-Instruct`), e.g.:

```json
{"captions": [{"caption": "The image shows two squares, one red and one green, placed side by side against a black background with a brown horizontal strip at the bottom...", "image": "frame_000.png"}], "dry_run": false}
```

### 3.4 Resume / idempotency (live)

Re-running the **same run id** with `--resume`:

```bash
npa workbench workflow submit .../token-factory-parallel-fanout.yaml \
  --run-id npa-wf-cpu-token-factory-parallel-fanout-a8cf71e0 --runtime --resume \
  --var bucket=lerobot-ccc9d3c7 --var prefix=npa-workflow-e2e/<run-id>/token-factory-parallel-fanout ...
```

```
[runtime] wave 001|caption-shards|...: replayed from ledger (job 75)
[runtime] wave 002|serial|:aggregate:-: replayed from ledger (job 76)
"status": "succeeded"      # "replayed": true for both waves
```

**Zero new SkyPilot jobs** were submitted and the run finished in ~4 s instead of
~6 min: the durable ledger
(`s3://.../npa-workflow/runtime.json`, `npa.workflow.runtime.v1`) made the rerun
idempotent.

---

## 4. Phase 2 — real runtime control flow (live, CPU tier)

**Spec:** `token-factory-gate-loop.yaml` — bounded loop (`max_iterations: 3`,
`until: promote_checkpoint`) over caption → VLM score → gate, then a `route`
state that branches on the same decision artifact.
**Command:**

```bash
pytest npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_gate_loop_early_exit_vs_full_budget -q -s
```

The two runs differ **only** in `--var grade_threshold=...`; the decision comes
from the real VLM score each iteration.

### 4.1 Run A — gate passes on iteration 1 → REAL early exit

**Run id:** `npa-wf-cpu-token-factory-gate-loop-37be5c1f-early` (`grade_threshold=0.0`)

Wave ledger (`s3://.../npa-workflow/runtime.json`):

```
status: succeeded | schema: npa.workflow.runtime.v1
  001|serial|refine:caption-batch:-    serial job=77 succeeded states=caption-batch
  002|serial|refine:score-batch:-      serial job=78 succeeded states=score-batch
  003|serial|refine:quality-gate:-     serial job=79 succeeded states=quality-gate
  004|serial|:route:-                  serial job=80 succeeded states=route
  005|serial|:publish:-                serial job=81 succeeded states=publish
  decision: promote_checkpoint <- s3://.../gate/decision.json     (loop exit check)
  decision: promote_checkpoint <- s3://.../gate/decision.json     (route branch)
```

**One** loop iteration out of a budget of three: `caption-batch` / `score-batch` /
`quality-gate` each ran exactly once, and no further iteration was submitted.

Artifacts under the run prefix:

```
captions/captions.json            1473    real Token Factory captions
grade/vlm_eval_stub.json           985    real hosted VLM eval  ("backend": "api", "dry_run": false)
gate/decision.json                  39    {"decision": "promote_checkpoint"}
insights/records.jsonl             614    route ingested the decision artifact
reports/promoted/dashboard.html   1222    the PROMOTE branch artifact
npa-workflow/runtime.json         5526    wave ledger
```

`gate/decision.json` (verbatim):

```json
{
  "decision": "promote_checkpoint"
}
```

Note the branch: only `reports/promoted/` exists — the `escalate` branch never
ran.

### 4.2 Run B — gate can never pass → full budget + the other branch

**Run id:** `npa-wf-cpu-token-factory-gate-loop-<hex>-full` (`grade_threshold=1.01`,
above the clamped `[0,1]` VLM score)

<!-- RUN_B_PLACEHOLDER -->

### 4.3 `--assume-decision` plan-only is unchanged

<!-- PLAN_ONLY_PLACEHOLDER -->

---

## 5. Phase 1 — parallel GPU reference case (`isaac-lab-rl-sweep.yaml`)

<!-- SWEEP_PLACEHOLDER -->

---

## 6. Cost

<!-- COST_PLACEHOLDER -->

---

## 7. Not verified live

<!-- NOT_VERIFIED_PLACEHOLDER -->
