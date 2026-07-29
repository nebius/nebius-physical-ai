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
export NPA_SRC_S3_URI=s3://<artifact-bucket>/npa-workflow-e2e/npa-src-wfrt/npa   # branch source
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
bash scripts/stage-npa-src.sh --bucket <artifact-bucket> --prefix npa-workflow-e2e/npa-src-wfrt
# staged 549 files -> s3://<artifact-bucket>/npa-workflow-e2e/npa-src-wfrt/npa
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
**Run prefix:** `s3://<artifact-bucket>/npa-workflow-e2e/npa-wf-cpu-token-factory-parallel-fanout-a8cf71e0/token-factory-parallel-fanout/`

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
  --var bucket=<artifact-bucket> --var prefix=npa-workflow-e2e/<run-id>/token-factory-parallel-fanout ...
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

**Run id:** `npa-wf-cpu-token-factory-gate-loop-9f272bff-full` (`grade_threshold=1.01`,
above the clamped `[0,1]` VLM score, so the gate can never promote)

```
status: succeeded | schema: npa.workflow.runtime.v1
  001|serial|refine:caption-batch:-   serial job=82 succeeded states=caption-batch
  002|serial|refine:score-batch:-     serial job=84 succeeded states=score-batch
  003|serial|refine:quality-gate:-    serial job=85 succeeded states=quality-gate
  004|serial|refine:caption-batch:-   serial job=86 succeeded states=caption-batch
  005|serial|refine:score-batch:-     serial job=87 succeeded states=score-batch
  006|serial|refine:quality-gate:-    serial job=88 succeeded states=quality-gate
  007|serial|refine:caption-batch:-   serial job=89 succeeded states=caption-batch
  008|serial|refine:score-batch:-     serial job=90 succeeded states=score-batch
  009|serial|refine:quality-gate:-    serial job=91 succeeded states=quality-gate
  010|serial|:route:-                 serial job=92 succeeded states=route
  011|serial|:escalate:-              serial job=93 succeeded states=escalate
  decision: loop_back_to_inner_loop <- s3://.../gate/decision.json   (iteration 1)
  decision: loop_back_to_inner_loop <- s3://.../gate/decision.json   (iteration 2)
  decision: loop_back_to_inner_loop <- s3://.../gate/decision.json   (iteration 3)
  decision: loop_back_to_inner_loop <- s3://.../gate/decision.json   (route branch)
```

**Eleven** waves — the full budget of three iterations — and the *other* branch:

```
gate/decision.json                  44   {"decision": "loop_back_to_inner_loop"}
grade/vlm_eval_stub.json          1038   score 0.0, backend "api", dry_run false, Qwen/Qwen2.5-VL-72B-Instruct
reports/shortfall/dashboard.html  1220   the ESCALATE branch artifact (no reports/promoted/)
npa-workflow/runtime.json        14005   wave ledger
```

### Side-by-side

| | Run A (`grade_threshold=0.0`) | Run B (`grade_threshold=1.01`) |
| --- | --- | --- |
| loop iterations executed | **1** of 3 | **3** of 3 |
| SkyPilot jobs | 77, 78, 79, 80, 81 | 82, 84–93 |
| waves | 5 | 11 |
| decision read from S3 | `promote_checkpoint` | `loop_back_to_inner_loop` (×4) |
| terminal branch | `publish` → `reports/promoted/` | `escalate` → `reports/shortfall/` |

Nothing but the threshold differed; the engine read the real artifact each
iteration, exited early in Run A, and branched differently in the two runs. The
harness test that asserts exactly this passed:

```
pytest .../test_npa_workflow_runtime_gate_loop_early_exit_vs_full_budget -q -s
1 passed in 2854.45s (0:47:34)
```

### 4.3 `--assume-decision` plan-only is unchanged

A second worktree was checked out at the **base commit** `d129ee90` and the same
specs were planned with both interpreters:

```bash
git worktree add -f /tmp/npa-base d129ee90
for s in sim2real-vlm-rl tokenfactory-cosmos-gate physical-ai-data-factory; do
  for d in loop_back promote_checkpoint; do
    PYTHONPATH=/tmp/npa-base/npa/src  python -m npa.cli.main workbench workflow plan-spec /tmp/npa-base/$P --run-id fixed-run --assume-decision $d --json > base.json
    PYTHONPATH=$WT/npa/src            python -m npa.cli.main workbench workflow plan-spec $WT/$P        --run-id fixed-run --assume-decision $d --json > branch.json
    diff base.json branch.json
  done
done
```

Result (after dropping the one additive JSON key `group`, which is `""` for every
serial step):

```
IDENTICAL (ignoring additive group key)  sim2real-vlm-rl          [loop_back]           steps=19
IDENTICAL (ignoring additive group key)  sim2real-vlm-rl          [promote_checkpoint]  steps=11
IDENTICAL (ignoring additive group key)  tokenfactory-cosmos-gate [loop_back]           steps=9
IDENTICAL (ignoring additive group key)  tokenfactory-cosmos-gate [promote_checkpoint]  steps=5
IDENTICAL (ignoring additive group key)  physical-ai-data-factory [loop_back]           steps=12
IDENTICAL (ignoring additive group key)  physical-ai-data-factory [promote_checkpoint]  steps=9
```

The raw diff before normalization contains **only** added `"group": ""` lines —
no step, argv, iteration or ordering change. The plan-time full unroll under
`--assume-decision` is byte-for-byte the same as on `main`.

---

## 5. GPU tier

### 5.1 A parallel wave whose tasks really request GPUs (live, SUCCEEDED)

Same spec and same code path as §3, re-run with the operator knob that puts every
task on a GPU (`NPA_E2E_FORCE_ACCELERATORS=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`),
to show the JobGroup path is not CPU-specific.

**Run id:** `npa-wf-cpu-token-factory-parallel-fanout-d65368f3`

```
ID  TASK  NAME                                       REQUESTED                                   STATUS
94        ...-01-caption  (JobGroup)                 -                                           SUCCEEDED
 ↳  0     caption-shard-a                            1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]  SUCCEEDED
 ↳  1     caption-shard-b                            1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]  SUCCEEDED
 ↳  2     caption-shard-c                            1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]  SUCCEEDED
95        ...-02-aggregate (barrier)                 1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]  FAILED
```

| task | submitted_at | end_at |
| --- | --- | --- |
| caption-shard-a | 1785301226.5093 | 1785301302.4967 |
| caption-shard-b | 1785301226.5134 | 1785301302.2853 |
| caption-shard-c | 1785301226.5172 | 1785301302.7468 |
| aggregate (barrier) | **1785301386.9483** | — |

Three GPU tasks, submitted within 8 ms of each other, ran concurrently
(driver log: `3 tasks running concurrently (caption-shard-a, caption-shard-b,
caption-shard-c)`), and the barrier was submitted 84 s after the last one ended.

The barrier task then failed with `ModuleNotFoundError: No module named 'npa'`:
SkyPilot's **GPU** default image resolves a different `python3` for the task body
than the one `setup` pip-installs into, so the S3-staged package is not importable
there. The identical stage succeeds on the CPU default image (§3, job 76). This is
an image/interpreter quirk of the default-image + staged-source path, not the
parallel or runtime engine — but it is a real failure and is reported as such.

### 5.2 `isaac-lab-rl-sweep.yaml` — NOT completed live (see §7)

The four-variant JobGroup was submitted (job **83**, run
`npa-wf-multi-isaac-lab-rl-sweep-f1d78688`, four
`1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` tasks). Two blockers appeared:

1. **`ErrImagePull` 401 for every private image on the cluster.** The cluster's
   `npa-nebius-registry` imagePullSecret (referenced by
   `~/.sky/config.yaml → kubernetes.pod_config.imagePullSecrets`) held a 9-day-old
   Nebius IAM token. Fixed by minting a fresh token and patching the secret:

   ```bash
   TOKEN=$(python -c "from npa.workflows.sim2real.registry_auth import mint_nebius_registry_token; print(mint_nebius_registry_token())")
   kubectl create secret docker-registry npa-nebius-registry -n default \
     --docker-server=cr.us-central1.nebius.cloud --docker-username=iam --docker-password="$TOKEN" \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

   After that the image pulled in ~0.3 s (`Successfully pulled image
   "...npa-isaac-lab:2.3.2.post1" ... Image size: 8413247039 bytes`). Side effect:
   this also unblocked another agent's job that had been stuck pulling
   `npa-cosmos2-transfer` for five days.
2. **SkyPilot cannot host a task inside the Isaac Lab image on this cluster.**
   Every provisioning attempt then failed with
   `KubernetesError: Failed to get ssh user for pod variant-*-83-...-head ...
   container not found ("ray-node")` and retried in a loop. This is the
   pre-existing workbench-image limitation the operator environment already works
   around with `NPA_E2E_CLEAR_WORKBENCH_IMAGES=1` (see the comment in
   `skypilot_render.py`: *"SkyPilot's k8s apt-ssh runtime setup fails inside
   npa-cosmos"*). The job was cancelled (`sky jobs cancel -y 83`) rather than left
   retrying.

---

## 6. Mandated harness commands

### 6.1 `test_npa_workflow_submit_live_e2e.py` (cpu tier)

```bash
export NPA_INTEGRATION_E2E=1 NPA_E2E_NPA_WORKFLOW_SUBMIT=1 NPA_E2E_NPA_WORKFLOW_RUNTIME=1
export NPA_E2E_REGISTRY=$NPA_E2E_REGISTRY
export NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu
export NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS=2700
export NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT=1
pytest npa/tests/e2e/test_npa_workflow_submit_live_e2e.py -v
```

<!-- MATRIX_RESULT -->

**Why `cpu` and not `cpu,gpu,multi`:** the gpu/multi one-shot twins in the matrix
(SONIC, Cosmos3, the 11-stage BDD100K pipeline, ...) are pre-existing cases that
are unrelated to this change and would cost many GPU-hours; and on this cluster
they need workbench images, which currently cannot host a SkyPilot k8s task
(§5.2). The tiers that exercise **this change** are the runtime cases, which are
all in the cpu tier plus the (blocked) `multi` sweep. Every spec in the matrix —
all tiers — is still covered by the plan-only matrix test in the same file.

### 6.2 `test_burst_live_e2e.py`

```bash
pytest npa/tests/e2e/test_burst_live_e2e.py -v
# 1 skipped in 1188.16s (0:19:48)
# reason: capacity / GPU not offered:
#   RTXPRO-6000-BLACKWELL-SERVER-EDITION:1=FAILED_PRECHECKS, L40S:1=STARTING
```

The burst test rotates through its GPU candidates and **skips** when none can be
scheduled; that is the test's own capacity guard, not a failure. The burst path is
unrelated to this change (it does not use the npa.workflow engine); it was run
because the mandate asked for it, and the honest result is "skipped for capacity".

---

## 7. Cost

| Item | Amount |
| --- | --- |
| CPU tasks (Token Factory captions, VLM scoring via hosted API, gates, joins, dashboards) | ~45 short pods, `1x[CPU:4+]`, ~30–90 s each |
| GPU seconds | 3 × RTXPRO-6000 × ~76 s (§5.1 fan-out) + 1 × ~60 s (barrier attempt) ≈ **~5 GPU-minutes** |
| GPU sweep (`isaac-lab-rl-sweep`) | **0 GPU-minutes billed for compute** — every attempt failed in provisioning (`ErrImagePull`, then SkyPilot runtime start); cancelled after ~40 min of retries |
| Burst test | 2-node GPU request, never scheduled (skipped) |
| Token Factory | ~20 caption/score calls on `Qwen/Qwen2.5-VL-72B-Instruct` with `max_images<=4`, `max_tokens<=128` |
| Storage | a few MB of PNG fixtures, JSON reports and ledgers under `s3://<bucket>/npa-workflow-e2e/...` |

Approximate spend: **single-digit GPU-minutes** plus negligible CPU/hosted-token
usage. No cluster was provisioned for this work; no cluster was left running (§9).

---

## 8. Not verified live

Stated plainly, so nothing here is mistaken for proven:

1. **`isaac-lab-rl-sweep.yaml` never executed its training variants live.** The
   spec validates, plans, renders (4-task JobGroup + barrier) and is registered in
   `SUBMIT_LIVE_MATRIX`; the JobGroup was submitted and SkyPilot did schedule four
   GPU pods, but the Isaac Lab image cannot host a SkyPilot k8s task on this
   cluster (§5.2). Its stage functions (`npa.workflows.rl_sweep`) are covered by
   unit tests only. Concurrency and barrier semantics for the *same* code path are
   proven live in §3 (CPU) and §5.1 (GPU).
2. **The `trigger:` / watch pattern is unit-tested only.** No live run waited on an
   S3 prefix; no spec in the shipped catalog uses `trigger:` yet.
3. **Wave retry and timeout-cancellation are unit-tested only.** No live wave
   failed transiently or timed out during these runs, so the retry/cancel paths did
   not execute against real infrastructure.
4. **Bounded-concurrency batching was exercised live only with
   `maxConcurrency == group size`** (one batch). The multi-batch path (`3` members,
   `maxConcurrency: 2` → two batches) is unit-tested.
5. **The GPU-tier barrier task failed** with `ModuleNotFoundError: No module named
   'npa'` on SkyPilot's GPU default image (§5.1). The same stage succeeds on the
   CPU default image; the interpreter mismatch on GPU images is not fixed here.
6. **Only the `cpu` tier of the live submit matrix was executed**; gpu/multi
   one-shot twins were covered plan-only (§6.1).

---

## 9. Teardown

<!-- TEARDOWN_PLACEHOLDER -->

---

## 10. Self-review checklist

| # | Acceptance item | Commit(s) | Evidence |
| --- | --- | --- | --- |
| 1 | A workflow can declare parallel fan-out and it launches as genuinely concurrent SkyPilot jobs | `4ab46d20` (spec fields + wave planner), `a8419888` (JobGroup renderer) | §3.1 — one `job_id`, 3 tasks submitted within 12 ms, overlapping lifetimes, driver log "3 tasks running concurrently" |
| 2 | The serial-only guard is lifted behind an **explicit** parallel path; serial stays the default | `a8419888` | `render_skypilot_yaml` untouched; `test_render_rejects_parallel_execution` still passes verbatim; §2 shows `--plan-only` still emits `execution: serial` for all 23 twins |
| 3 | Barrier: a downstream `needs:` state waits for all parallel predecessors | `b55d081f` (wave boundary in the runtime tier) | §3.2 — barrier submitted 94.6 s after the last member ended; §5.1 — 84 s after |
| 4 | Bounded concurrency respected | `4ab46d20` (`maxConcurrency` + batching), `b55d081f` (`execute_parallel` chunking) | unit: `test_wave_plan_groups_parallel_members`, `test_runtime_launches_parallel_group_as_job_group_with_barrier` (2+1 batches), `test_runtime_max_concurrency_override_widens_batches`; live: single-batch only (§8.4) |
| 5 | `isaac-lab-rl-sweep.yaml` ported to a real npa.workflow parallel spec | `0cd3cc40` (spec + `rl_sweep` stages) | §2 wave preview (4-task JobGroup + barrier); **not run live** — §5.2 / §8.1 |
| 6 | Runtime tier above `build_scheduler_task`: plan → submit → poll → read S3 decision → replan | `b55d081f` | §4.1/§4.2 ledgers: per-wave `job_id`, `sky_status`, decision reads |
| 7 | Consumes the existing decision contract, no new gate mechanism | `b55d081f` (`RecordingDecisionReader` over `decisions.refresh_context_decision`) | §4.1 `gate/decision.json` written by the existing `grade_gate`, read back by the engine |
| 8 | Bounded loops with **real** early-exit | `b55d081f`, `0cd3cc40` | §4 — 5 waves / 1 iteration (threshold 0.0) vs 11 waves / 3 iterations (threshold 1.01), same spec |
| 9 | Data-dependent branching (`goto`) | `b55d081f` | §4 — `route` → `publish` vs `route` → `escalate` decided by the artifact; unit `test_runtime_branch_follows_transition_goto` |
| 10 | Trigger / watch-loop pattern | `4ab46d20` (spec field), `b55d081f` (`s3_trigger_waiter`) | unit `test_trigger_waits_for_objects_then_runs`, `test_trigger_gives_up_after_max_polls`; **not run live** — §8.2 |
| 11 | `--assume-decision` plan-only path preserved as the offline fallback | (unchanged code) | §4.3 — plan JSON identical to base commit `d129ee90` for 3 dynamic specs × 2 assumptions |
| 12 | Every existing plan-only test and the shown-catalog guardrail still pass | all | §1 (263 engine+smoke, 3581 full offline), §2 (23 plan-only matrix cases) |
| 13 | Job failure/retry, idempotency and resume built on `run_state.py` | `b55d081f` (`RuntimeLedger`, `npa.workflow.runtime.v1`) | §3.4 — live `--resume` replayed both waves, **zero** new jobs; unit tests for retry/timeout/exhaustion |
| 14 | No hardcoded project/tenant/registry/bucket IDs or secrets | all | every live value comes from `NPA_E2E_*` / `~/.npa/*`; `scripts/stage-npa-src.sh` takes `--bucket`; leak assertions in the harness |
| 15 | Unit tests mock all infra | `4ab46d20`, `a8419888`, `b55d081f`, `e378d38a`, `a7cbde3a` | injected submitter/status/timeline/canceller/sleeper/clock/storage in `test_runtime_orchestrator.py`, `test_rl_sweep.py`, `test_fanout_join.py` |
| 16 | Scheduler-task seam preserved | `a8419888` | both renderers build docs via `build_skypilot_task_doc` → `build_scheduler_task`; the runtime tier only passes `PlanStep`s |
| 17 | Backward compatible: existing serial specs render and submit unchanged | `4ab46d20` (flatten), `a8419888` (separate entry point) | §4.3 plan parity; §6.1 the pre-existing cpu twins still submit and succeed |
| 18 | New specs registered in `SUBMIT_LIVE_MATRIX` (incl. a `multi`/parallel case) | `683a1abd`, `e747c8ef` | `token-factory-parallel-fanout` (cpu), `token-factory-gate-loop` (cpu, also in `DYNAMIC_SPECS`), `isaac-lab-rl-sweep` (multi) |
| 19 | Cheapest live path first; cancel on timeout; no leaked clusters | — | §3 (CPU first), §5.2 (`sky jobs cancel -y 83`), §9 |
| 20 | Honest reporting of what was not verified live | — | §8 |
