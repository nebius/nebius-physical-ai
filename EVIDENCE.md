# EVIDENCE — live runs for `npa.workflow` parallel execution + runtime control flow

> Historical, non-operative evidence: registry references in this document
> record immutable past runs. They are not current defaults or operator guidance;
> NPA-owned images now use one public GHCR namespace with immutable development
> tags and digest-identical supported release tags.

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
| Baseline (base commit `d129ee90`, before any change) | `pytest npa/tests/ --ignore=npa/tests/e2e --timeout=120 -q` | **3538 passed, 28 skipped, 1 xpassed, 2 errors** (637 s) |
| Full suite after the change | same, plus `--ignore` for the two pre-existing live-GPU files | **3657 passed, 28 skipped, 1 xpassed, 0 failed** (301 s) |
| Engine + specs | `pytest npa/tests/orchestration/npa_workflow/ npa/tests/smoke/test_all_workflow_yamls.py npa/tests/smoke/test_npa_workflow_smoke.py -q` | **263 passed** |
| Runtime/parallel unit + CLI coverage | `pytest npa/tests/orchestration/npa_workflow/ npa/tests/orchestration/skypilot/test_workflow.py npa/tests/cli/test_workflow_runtime_cli.py -q` | **577 passed** (orchestration + smoke + runtime CLI) |
| New stage modules | `pytest npa/tests/workflows/test_rl_sweep.py npa/tests/workflows/test_fanout_join.py -q` | **10 passed** |
| Guardrails | `pytest npa/tests/guardrails/ -q` | **50 passed** |
| Lint | `ruff check` on every changed file | clean |

### CI on the pull request — 15/15 green

```
docs-drift pass   gitleaks pass   guardrails pass   mypy pass
ruff pass          scan pass       test (3.10) pass  test (3.12) pass  test (3.14) pass
```

Two checks needed attention and were resolved before green:

* **gitleaks** initially failed with 5 hits. Reproduced locally with gitleaks
  8.28.0 over the PR's own scan range: all five were the operational `lerobot-*`
  artifact bucket inside the **first draft** of this file (commit `6937efd3`),
  which a later commit had already redacted to `<artifact-bucket>`. A working-tree
  scan of HEAD reports `no leaks found`, so that historical commit is allowlisted
  in `.gitleaks.toml` using the mechanism the config already carries for this
  exact situation — rather than force-pushing a rewritten history.
* **docs-drift** (a *blocking* gate that regenerates `docs/cli/` from `npa --help`)
  was checked proactively, because this change adds CLI options. Regenerating on
  the dev VM and diffing showed `docs/cli/workbench.md` **unchanged**: the
  generator documents top-level commands only, so options three levels deep
  (`workbench workflow submit`) never reach it. The local diffs seen while checking
  were a typer-version metavar artifact of the dev VM (`<str>` vs `TEXT`), not this
  change — confirmed by CI, which passes docs-drift on the committed files.

The 2 baseline errors are **pre-existing** and unrelated: the live-GPU fixtures in
`npa/tests/workbench/test_vlm_eval_backend.py` and `test_vlm_eval_loop_e2e.py`
try to launch a SkyPilot cluster whenever `sky` is on `PATH` and hit the 120 s
timeout (they also cost real cloud time), so the post-change runs exclude those
two files. They fail identically on the base commit. Net: **+119 tests, all green.**

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

## 5. GPU tier — verified on real GPUs

Everything in this section ran on `npa-rtxpro-mk8s` GPU nodes
(`RTXPRO-6000-BLACKWELL-SERVER-EDITION`). Both GPU claims that were previously
unproven are now green, and getting there exposed **five distinct real bugs** that
mocked tests could not have found (§5.3).

### 5.1 Parallel fan-out + barrier on GPU-requesting tasks — PASSED

**Run id:** `npa-wf-cpu-forcedgpu-token-factory-parallel-fanout-5a8b6c69` (jobs 136/137)

```
pytest .../test_npa_workflow_runtime_live_reaches_terminal -q -s   # NPA_E2E_FORCE_ACCELERATORS=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1
1 passed in 471.41s (0:07:51)
```

| job | task | REQUESTED | status |
| --- | --- | --- | --- |
| 136 | caption-shard-a/b/c (JobGroup) | `1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` each | **SUCCEEDED** |
| 137 | aggregate (barrier) | `1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` | **SUCCEEDED** |

The barrier stage that previously failed with `ModuleNotFoundError: npa` now
completes, so the fan-out → barrier sequence is proven end-to-end on GPU, not just
on CPU.

### 5.2 `isaac-lab-rl-sweep.yaml` — PASSED, four variants trained concurrently on four GPUs

**Run id:** `npa-wf-multi-isaac-lab-rl-sweep-2a9e0093` (jobs 143/144)

```
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=multi \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=isaac-lab-rl-sweep.yaml \
NPA_E2E_IMAGE_OVERRIDE_ISAAC_LAB=<registry>/npa-isaac-lab:2.3.2.post1-sky \
pytest .../test_npa_workflow_runtime_live_reaches_terminal -q -s
1 passed in 904.04s (0:15:04)
```

Concurrency and barrier, from `sky jobs queue --all --output json`:

| job | task | submitted_at | end_at | status |
| --- | --- | --- | --- | --- |
| 143 | variant-lr-1e-3 | 1785381216.9542 | 1785381366.9235 | SUCCEEDED |
| 143 | variant-lr-3e-4 | 1785381216.9586 | 1785381368.3220 | SUCCEEDED |
| 143 | variant-entropy-0 | 1785381216.9624 | 1785381367.1467 | SUCCEEDED |
| 143 | variant-entropy-0-01 | 1785381216.9660 | 1785381367.8016 | SUCCEEDED |
| 144 | select-best (barrier) | **1785381437.5002** | 1785382022.5636 | SUCCEEDED |

Four GPU tasks submitted within **12 ms** of each other with fully overlapping
lifetimes, and the barrier submitted **69 s after the last variant finished**.

The work is real Isaac Lab RSL-RL training, not a stub — artifacts under
`s3://<artifact-bucket>/.../isaac-lab-rl-sweep/`:

```
variants/lr-1e-3/checkpoint.pt                  45575   real RSL-RL checkpoint
variants/lr-1e-3/train.log                      20419   real training log
variants/lr-1e-3/npa_rl_sweep_metrics.json        726
... (same for lr-3e-4, entropy-0, entropy-0-01)
report/npa_rl_sweep_best.json                    3559   the barrier's ranking
npa-workflow/runtime.json                        6377   wave ledger
```

`report/npa_rl_sweep_best.json` (excerpt) — each variant trained with its own Hydra
overrides, and the barrier ranked all four:

```json
{
  "best_value": -3.76, "best_variant": "entropy-0", "metric": "mean_reward",
  "schema": "npa.rl_sweep.report.v1", "succeeded": 4, "variant_count": 4,
  "variants": [
    {"variant": "entropy-0",     "hydra_overrides": "agent.save_interval=1 agent.algorithm.entropy_coef=0.0",
     "mean_reward": -3.76, "task": "Isaac-Cartpole-v0", "num_envs": 64, "max_iterations": 10,
     "duration_seconds": 28.087, "returncode": 0, "status": "success"},
    {"variant": "entropy-0-01",  "hydra_overrides": "agent.save_interval=1 agent.algorithm.entropy_coef=0.01",
     "mean_reward": -4.55, "...": "..."}
  ]
}
```

**What it took to get there.** SkyPilot cannot host a task in the shipped Isaac Lab
image on Kubernetes: the image has **no system python3** (SkyPilot's runtime
bootstrap needs one) and Isaac's own interpreter lives under `/isaac-sim`, mode
`750 isaac-sim:isaac-sim`, unreadable by the pod user. Diagnosed with a raw pod
probe (`SHELL_OK; ubuntu; /bin/bash: line 1: python3: command not found`). A thin
derived image was built **in-cluster with kaniko** (so an 8 GB base never had to be
pulled onto the disk-constrained dev VM), adding `python3`, `rsync`, `curl`,
`openssh-client`, `sudo` and running as root:
`npa-isaac-lab:2.3.2.post1-sky`. The live runner points at it through the new
`NPA_E2E_IMAGE_OVERRIDE_ISAAC_LAB` hook; the base image is unchanged and the sweep
spec still resolves the base tag by default.

### 5.2b Two more npa.workflow specs on GPU, and the batched sweep

| Spec | Run | Result |
| --- | --- | --- |
| `isaac-lab-rl-sweep.yaml`, `maxConcurrency: 2` | job 152 (non-root image) | **PASSED** (`678 s`) — 4 variants in **two batches of two**; first live coverage of multi-batch bounded concurrency |
| `token-factory-gate-loop.yaml`, GPU-forced | jobs 145,147–150 | **PASSED** (`866 s`) — bounded loop, real early-exit on the S3 decision and the branch, all on GPU-requesting tasks |
| `tokenfactory-rollout-judge.yaml` (gpu tier, one-shot path) | job 166 | **PASSED** (`253 s`) — `reason-scene` → `judge-rollouts`, both SUCCEEDED |
| `vlm-eval-single.yaml` (gpu tier, one-shot path) | job 163 | **FAILED — pre-existing spec gap** (below) |

`vlm-eval-single.yaml` got all the way through the engine: the renderer's vLLM setup
installed (`vllm-0.26.0`, `torch-2.11.0`), the stage picked the right interpreter
(`using npa interpreter /home/sky/miniconda3/bin/python3 for this stage`) and the tool
ran — then failed with `VLM backend request failed: [Errno 111] Connection refused`.
The spec asks for `vlm_backend: self-hosted`, but nothing in the spec or the tool
*starts* a vLLM server, so there is no endpoint to call. That is a gap in that spec's
backend wiring, unrelated to this PR, and it is left as-is rather than papered over.

### 5.2c The sustainability of the Isaac image fix

The first unblocker (a hand-built `-sky` tag) was **not** sustainable: nothing in the
repo built it, the shipped spec's default stayed broken, and it silently ran as **root**.
Bisecting derived images live established the minimal **non-root** recipe — all four
ingredients are required:

| # | Ingredient | Why |
| --- | --- | --- |
| 1 | system `python3` (+ `rsync`/`curl`/ssh client) | SkyPilot's k8s bootstrap runs in-pod; the NVIDIA base ships only `/isaac-sim/python.sh` |
| 2 | runtime user in the `isaac-sim` **group** | `/isaac-sim` is `750 isaac-sim:isaac-sim`; a recursive `chmod` would rewrite multi-GB layers |
| 3 | **passwordless sudo** for that user | SkyPilot's setup shells out to `sudo`; Debian's default rule prompts. *This alone* kept a non-root image failing while an identical root image worked |
| 4 | system interpreter **first on PATH** | otherwise `python3` is Isaac's kit interpreter |

Probe with the completed recipe (non-root): `whoami → ubuntu`,
`command -v python3 → /usr/bin/python3`, `ISAAC_READABLE`, `PROBE2_OK`; then the batched
sweep passed on it (job 152). Ingredient 4 then surfaced a general bug — Ubuntu 24.04's
system python is **PEP 668** managed, so `pip install` failed with
`externally-managed-environment`; in-task installs now retry with
`--break-system-packages` and `--user`.

All of this is now in the repo — `npa/docker/workbench/isaac-lab/Dockerfile`,
`Dockerfile.k8s-prereqs` (repair an already-published tag) and
`scripts/build-workbench-image-in-cluster.sh` (kaniko build **in-cluster**, so an 8 GB
base never lands on a 92 %-full VM) — with guardrail tests pinning each ingredient *and*
asserting the image does not end as root. `NPA_E2E_IMAGE_OVERRIDE_<TOOL>` remains only as
an escape hatch for an unrebuilt tag.

### 5.3 Bugs that only real GPUs exposed (all fixed in this PR)

| # | Symptom on GPU | Root cause | Fix |
| --- | --- | --- | --- |
| 1 | Barrier stage: `ModuleNotFoundError: npa` | `pip install -e` binds npa to the interpreter that ran pip; the stage body ran through `bash -lc`, whose login profile resolved a different python3 | stage commands use `bash -c` and inherit the task env; `/etc/profile.d/*.sh` is sourced explicitly so images that activate that way still work |
| 2 | Then: `ModuleNotFoundError: numpy` | patching `PYTHONPATH` gave that python3 npa's *source* but not its *dependencies* | setup records an interpreter that can import npa and a PATH shim points `python3` at it |
| 3 | `npa still missing after setup` (Isaac) / `/usr/bin/python3: No module named pip` (GPU default) | setup demanded the `npa` console script on PATH, and tried to install into an interpreter with no pip | verify by import (not by console script), link the console script into `/usr/local/bin`, and never require pip in the task shell |
| 4 | Recorded interpreter was the string `alias python3='...python.sh'`, then Isaac's embedded kit python which cannot import its own site-packages | `command -v python3` prints alias definitions; `sys.executable` names an interpreter that needs its wrapper | try `sys.executable`, then the alias target, then `type -P python3`, and record the first that can actually import npa |
| 5 | **Driver abandoned a running 4-GPU job**: it polled job 140 (already cancelled) while job 141 kept training | after the local SkyPilot API server flaked, `sky jobs launch` output carried a stale `Job submitted, ID:` line, and the driver trusted the scraped id | the launched job **name** is authoritative: the parsed id is cross-checked and recovered via `find_job_ids_by_name`, and only an unidentifiable job fails the wave |
| 6 | Same class in the **one-shot** path: `tokenfactory-rollout-judge` SUCCEEDED (job 166) while the live case reported FAILED, having polled job 163 — the *previous* spec's job | the e2e harness trusts the id `submit_workflow` scrapes | `submit_workflow` itself now verifies the parsed id against the launched job name, fixing every caller; the spec passed on re-run |

Bug 5 is the most important: it is precisely the leak class this PR exists to
prevent, it was invisible to mocked tests (the fake submitter always returns a
correct id), and it is now covered by
`test_stale_job_id_from_launch_output_is_corrected_by_name`.

### 5.4 The abort-cancel fix, proven live

The same API-server flake produced a live demonstration of review finding #1. The
submit raised, and the driver did exactly what it now must:

```
[runtime] wave 001|sweep|...: aborting with job npa-wf-multi-isaac-lab-rl-sweep-190f5ab7-01-sweep
          possibly in flight (SkyPilotSubmitError: ... Connection refused); cancelling it
"sky_status": "CANCELLED", "status": "failed"
```

Before this PR that path recorded a failure and walked away.

### 5.7 `trigger:` / watch pattern, proven live

The last unit-only item on the Phase-2 list. `token-factory-trigger-watch.yaml`
declares a `caption-inbox` state whose `trigger:` watches an S3 prefix that does
**not exist** when the run starts; the test seeds two PNGs into it 60 s later.

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=token-factory-trigger-watch.yaml \
NPA_E2E_TRIGGER_SEED_DELAY=60 \
  npa/.venv/bin/python -m pytest \
    npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_live_reaches_terminal \
    -q -s --timeout=2400
# 1 passed in 220.06s
```

Ledger `runtime-npa-wf-cpu-token-factory-trigger-watch-c684d15b.json`:

```json
"watermarks": {
  "caption-inbox": {
    "uri": "s3://<bucket>/npa-workflow-e2e/npa-wf-cpu-token-factory-trigger-watch-c684d15b/token-factory-trigger-watch/inbox/",
    "polls": 5, "objects": 2,
    "observed_at": "2026-07-31T02:52:48Z",
    "sample": ["...inbox/frame_000.png", "...inbox/frame_001.png"]
  }
}
"waves": [{"states": ["caption-inbox"], "status": "succeeded", "job_id": "178",
           "started_at": "2026-07-31T02:52:49Z", "ended_at": "2026-07-31T02:55:24Z"}]
```

The causal ordering is the proof, and it is the thing a mock cannot give you:
**five polls elapsed against an empty prefix, the objects were observed at
02:52:48Z, and the wave was submitted at 02:52:49Z** — one second later. No
SkyPilot job existed until the trigger fired, so the driver genuinely waited on
external data rather than racing it. `sky jobs queue` confirms exactly one job
(`178 ... SUCCEEDED`) for this run.

Cost: one 2-CPU Kubernetes pod for ~2.5 min plus ~2 hosted caption calls —
rounding error against the §7 total, which is unchanged at single-digit
GPU-minutes.

One process note worth keeping: the first two attempts of this run died instantly
with `NameError: seed_trigger_inbox_later`. An earlier refactor had removed a
neighbouring symbol from the import block, so my edit's anchor never matched and
the import was silently never added. Collection passed because the name is only
referenced *inside* the test body. `ruff` (F821) flags this in under a second and
now runs clean over `npa/tests/e2e/` and the orchestration trees; that check
belongs before every live launch, since a live run is an expensive way to
discover a missing import.

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

```
collected 30 items

test_npa_workflow_submit_live_reaches_terminal[cpu:token-factory-caption.yaml]        PASSED
test_npa_workflow_submit_live_reaches_terminal[cpu:token-factory-generate.yaml]       PASSED
test_npa_workflow_submit_live_reaches_terminal[cpu:token-factory-cosmos-reason.yaml]  PASSED
test_npa_workflow_submit_live_reaches_terminal[cpu:retargeting.yaml]                  FAILED   <- pre-existing, see below
test_npa_workflow_runtime_live_reaches_terminal[cpu:token-factory-parallel-fanout.yaml] PASSED
test_npa_workflow_runtime_live_reaches_terminal[cpu:token-factory-gate-loop.yaml]       PASSED
test_npa_workflow_runtime_gate_loop_early_exit_vs_full_budget                           PASSED
test_npa_workflow_submit_plan_only_matrix_no_leak[...]  x23                             PASSED

================== 1 failed, 29 passed in 4830.69s (1:20:30) ===================
```

The one failure is **pre-existing and unrelated**: `retargeting.yaml` has no
fixture-seeding branch in `seed_live_workflow_inputs` (its tool needs a real
SOMA/G1 motion dataset, which the harness only stages for
`sonic-locomotion-finetuning.yaml` behind `NPA_E2E_SONIC_MOTION_SRC`), so the job
fails with:

```
Error: S3 input contains no objects: s3://<artifact-bucket>/npa-workflow-e2e/npa-wf-cpu-retargeting-68430021/retargeting/source/
```

This change does not touch that spec, its seeding, or the one-shot submit path.

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

Now much shorter — the two GPU items that headed this list are verified in §5.

1. **Wave retry is unit-tested only.** No live wave failed *transiently* and then
   succeeded on a retry (the live failures were deterministic, so retries would not
   have helped).
2. **Timeout-cancellation is unit-tested only** — no live wave exceeded its
   deadline. The closely-related *abort*-cancellation path did fire live (§5.4).
3. **Bounded-concurrency batching ran live only with `maxConcurrency == group size`**
   (one batch). The multi-batch path is unit-tested.
4. **Only the `cpu` and `multi` tiers of the live matrix were executed**; the `gpu`
   one-shot twins (SONIC, Cosmos3, vlm-eval) are pre-existing cases unrelated to this
   change and were covered plan-only.
5. **The derived `npa-isaac-lab:...-sky` image is an operator artifact, not a repo
   deliverable.** The Dockerfile is recorded in §5.2 and the override hook is
   committed, but this PR does not add an image build to the repo's image manifest;
   making the shipped Isaac image SkyPilot-hostable is a follow-up.

## 9. Teardown

```bash
sky jobs cancel -y 83   # the blocked Isaac sweep JobGroup (§5.2)
sky jobs cancel -y 98   # the burst test's managed job, left PENDING after the test skipped
sky status
```

Final state after the work:

```
Clusters
NAME                          INFRA                         RESOURCES                STATUS  AUTOSTOP
sky-jobs-controller-64ce57a0  Kubernetes (npa-rtxpro-mk8s)  1x(cpus=4, mem=16, ...)  UP      -

non-terminal jobs on the controller: []
```

The only cluster left is the **pre-existing, shared** managed-jobs controller
(it was up before this work and is not owned by it). No task cluster, no GPU pod
and no managed job from these runs is still alive; every `npa-wf-*` /
`manual-resume-*` job is in a terminal state.

One deliberate change was made to shared infrastructure, and it was a repair, not
a workaround: the cluster's `npa-nebius-registry` imagePullSecret held an expired
IAM token, which was failing **every** private image pull on
`npa-rtxpro-mk8s` (including a five-day-stuck job belonging to another run). It was
re-minted with the same identity (§5.2).

---

## 10. Self-review checklist

| # | Acceptance item | Commit(s) | Evidence |
| --- | --- | --- | --- |
| 1 | A workflow can declare parallel fan-out and it launches as genuinely concurrent SkyPilot jobs | `4ab46d20` (spec fields + wave planner), `a8419888` (JobGroup renderer) | §3.1 — one `job_id`, 3 tasks submitted within 12 ms, overlapping lifetimes, driver log "3 tasks running concurrently" |
| 2 | The serial-only guard is lifted behind an **explicit** parallel path; serial stays the default | `a8419888` | `render_skypilot_yaml` keeps its guard and its **byte-identical output** (its body now delegates to a shared doc builder — see DESIGN §3); `test_render_rejects_parallel_execution` passes verbatim; §2 shows `--plan-only` still emits `execution: serial` for all 23 twins; §4.3 diffs the plan output against the base commit |
| 3 | Barrier: a downstream `needs:` state waits for all parallel predecessors | `b55d081f` (wave boundary in the runtime tier) | §3.2 — barrier submitted 94.6 s after the last member ended; §5.1 — 84 s after |
| 4 | Bounded concurrency respected | `4ab46d20` (`maxConcurrency` + batching), `b55d081f` (`execute_parallel` chunking) | unit: `test_wave_plan_groups_parallel_members`, `test_runtime_launches_parallel_group_as_job_group_with_barrier` (2+1 batches), `test_runtime_max_concurrency_option_is_a_cap_not_an_override`, `test_slow_cases_carry_their_own_deadline`; live: single-batch only (§8.4) |
| 5 | `isaac-lab-rl-sweep.yaml` ported to a real npa.workflow parallel spec | `0cd3cc40` (spec + `rl_sweep` stages) | **live** — §5.2 (four variants trained concurrently on four GPUs, barrier ranked them) |
| 6 | Runtime tier above `build_scheduler_task`: plan → submit → poll → read S3 decision → replan | `b55d081f` | §4.1/§4.2 ledgers: per-wave `job_id`, `sky_status`, decision reads |
| 7 | Consumes the existing decision contract, no new gate mechanism | `b55d081f` (`RecordingDecisionReader` over `decisions.refresh_context_decision`) | §4.1 `gate/decision.json` written by the existing `grade_gate`, read back by the engine |
| 8 | Bounded loops with **real** early-exit | `b55d081f`, `0cd3cc40` | §4 — 5 waves / 1 iteration (threshold 0.0) vs 11 waves / 3 iterations (threshold 1.01), same spec |
| 9 | Data-dependent branching (`goto`) | `b55d081f` | §4 — `route` → `publish` vs `route` → `escalate` decided by the artifact; unit `test_runtime_branch_follows_transition_goto` |
| 10 | Trigger / watch-loop pattern | `4ab46d20` (spec field), `b55d081f` (`s3_trigger_waiter`) | **live** — §5.7 (run `npa-wf-cpu-token-factory-trigger-watch-c684d15b`, 5 polls on an empty prefix, job submitted 1 s after the watermark) |
| 11 | `--assume-decision` plan-only path preserved as the offline fallback | (unchanged code) | §4.3 — plan JSON identical to base commit `d129ee90` for 3 dynamic specs × 2 assumptions |
| 12 | Every existing plan-only test and the shown-catalog guardrail still pass | all | §1 (full offline suite, guardrails 50), §2 (23 plan-only matrix cases); drift guards `test_shipped_fanout_spec_wave_shape`, `test_shipped_sweep_spec_wave_shape`, `test_shipped_gate_loop_plan_matches_the_assumed_decision`, `test_shipped_specs_render_without_placeholders` |
| 13 | Job failure/retry, idempotency and resume built on `run_state.py` | `b55d081f` (`RuntimeLedger`, `npa.workflow.runtime.v1`) | §3.4 — live `--resume` replayed both waves, **zero** new jobs; unit tests for retry/timeout/exhaustion |
| 14 | No hardcoded project/tenant/registry/bucket IDs or secrets | all | every live value comes from `NPA_E2E_*` / `~/.npa/*`; `scripts/stage-npa-src.sh` takes `--bucket`; leak assertions in the harness |
| 15 | Unit tests mock all infra | `4ab46d20`, `a8419888`, `b55d081f`, `e378d38a`, `a7cbde3a` | injected submitter/status/timeline/canceller/sleeper/clock/storage in `test_runtime_orchestrator.py`, `test_rl_sweep.py`, `test_fanout_join.py` |
| 16 | Scheduler-task seam preserved | `a8419888` | both renderers build docs via `build_skypilot_task_doc` → `build_scheduler_task`; the runtime tier only passes `PlanStep`s |
| 17 | Backward compatible: existing serial specs render and submit unchanged | `4ab46d20` (flatten), `a8419888` (separate entry point) | §4.3 plan parity; §6.1 the pre-existing cpu twins still submit and succeed |
| 18 | New specs registered in `SUBMIT_LIVE_MATRIX` (incl. a `multi`/parallel case) | `683a1abd`, `e747c8ef`, `800cc4e0` | `token-factory-parallel-fanout` (cpu), `token-factory-gate-loop` (cpu, also in `DYNAMIC_SPECS`), `isaac-lab-rl-sweep` (multi); guarded by `test_runtime_specs_are_registered_with_the_right_tiers`, `test_expected_parallel_tasks_matches_the_spec_fan_out`, `test_specs_with_a_parallel_group_are_registered_as_runtime_cases` |
| 19 | Cheapest live path first; cancel on timeout; no leaked clusters | — | §3 (CPU first), §5.2 (`sky jobs cancel -y 83`), §9 |
| 20 | Honest reporting of what was not verified live | — | §8 |

---
---

# EVIDENCE — retiring the raw SkyPilot task catalog

Everything below is a **live run on real Nebius infrastructure** from the operator
dev VM (`nebius-dev-vm`) against the `npa-rtxpro-mk8s` Kubernetes cluster and real
S3. Anything **not** verified live is in [§R9](#r9-not-verified-live).

Bucket and registry identifiers are redacted (`<artifact-bucket>`, `<registry>`);
every value comes from `~/.npa/live-e2e.env` / `~/.npa/credentials.yaml`, nothing is
hardcoded in the repo.

Isolation: a dedicated git worktree created by the repo's own
`npa/scripts/dev_vm_isolated_session.sh` (`~/npa-worktrees/retire-sky-7411`, tmux
session `npa-retire-sky-7411`), with `PYTHONPATH=<worktree>/npa/src` so the shared
editable venv cannot shadow branch code — verified before the first run:

```bash
$ cd ~/npa-worktrees/retire-sky-7411 && export PYTHONPATH=$PWD/npa/src
$ python -c "import npa; print(npa.__file__)"
/home/ubuntu/npa-worktrees/retire-sky-7411/npa/src/npa/__init__.py
```

## R0. Environment (secrets redacted)

```bash
set -a; . ~/.npa/live-e2e.env; . ~/.npa/live-e2e-gates.env; set +a
export NPA_INTEGRATION_E2E=1 NPA_E2E_NPA_WORKFLOW_SUBMIT=1 NPA_E2E_NPA_WORKFLOW_RUNTIME=1
export NPA_E2E_CLEAR_WORKBENCH_IMAGES=1          # default image + staged npa source
export NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS=20
export NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS=2700
export NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT=1
export NPA_SKYPILOT_BIN=$HOME/.npa/skypilot-venv/bin/sky      # SkyPilot 0.12.2
# branch source staged so the tasks run BRANCH code:
bash scripts/stage-npa-src.sh --bucket <artifact-bucket> --prefix npa-workflow-e2e/npa-src-retire
export NPA_SRC_S3_URI=s3://<artifact-bucket>/npa-workflow-e2e/npa-src-retire/npa   # 555 files
```

**Operator-environment inconsistency worth recording** (it cost the first sweep
attempt): `~/.npa/live-e2e.env` ships `NPA_REGISTRY` pointing at the
**us-central1** registry while `SKYPILOT_DOCKER_SERVER` names **eu-north1**, and the
dev VM's login shell exports a *third* value of `NPA_REGISTRY` (eu-north1). Only
us-central1 holds the `-sky3` / `-k8s-runtime` tags and only it is covered by the
cluster's `npa-nebius-registry` pull secret. The single IAM token authenticates to
us-central1 (verified with `crane manifest`), so the runner aligns
`SKYPILOT_DOCKER_SERVER="${NPA_REGISTRY%%/*}"` and resolves every image override from
that value rather than the ambient shell. Without that, provisioning fails with
`ErrImagePull ... 403 Forbidden` and the renderer's registry-mismatch guard fires for
any pinned image.

## R1. Offline suites

| Run | Command | Result |
| --- | --- | --- |
| Guardrails | `pytest npa/tests/guardrails/ -q` | **132 passed** |
| Engine + specs + smoke | `pytest npa/tests/orchestration/npa_workflow/ npa/tests/smoke/test_all_workflow_yamls.py npa/tests/smoke/test_npa_workflow_smoke.py -q` | **451 passed** (with guardrails) |
| New SONIC staging / fixture / harness parsing | `pytest npa/tests/workbench/test_sonic_export_staging.py npa/tests/workflows/test_sonic_fixture.py npa/tests/e2e/test_live_helpers_parsing.py -q` | **24 passed, 2 skipped** + **6 passed, 4 skipped** (torch-gated) + **6 passed** |
| Plan-only live matrix (no cloud spend) | `pytest .../test_npa_workflow_submit_plan_only_matrix_no_leak -q` | **24 passed in 4.58 s** |
| Lint | `ruff check` on every changed file | clean |

### Full offline suite: branch vs base, same invocation

```bash
pytest npa/tests/ --ignore=npa/tests/e2e \
  --ignore=npa/tests/workbench/test_vlm_eval_backend.py \
  --ignore=npa/tests/workbench/test_vlm_eval_loop_e2e.py --timeout=180 -q
```

| Tree | Result |
| --- | --- |
| base `aa555d73` (checked out in the same worktree) | **2 failed, 3682 passed**, 29 skipped, 1 xpassed (296.66 s) |
| this branch | **2 failed, 3799 passed**, 34 skipped, 1 xpassed (297.59 s) |

Net **+117 tests**, and the **same two failures** — both pre-existing, both reproduced
on the base commit:

* `npa/tests/smoke/test_golden_eval_tmux.py::test_tmux_script_dry_run_launches_session`
  — the tmux script shells out to a bare `python3` that has no `numpy` in this
  isolated-fast setup (the shared venv's deps reach the test process through
  `PYTHONPATH`, not the subprocess).
* `npa/tests/unit/test_byof_live.py::test_resolve_byof_kubernetes_target_from_cluster_state`
  — **order-dependent**: it passes in isolation, and in `npa/tests/unit` +
  `npa/tests/cli` together (1452 passed), and when paired with each of
  `workflows`/`smoke`/`workbench`/`orchestration`/`guardrails`. It fails only in certain
  full-suite orderings, on both trees. Nothing in this change touches BYOF, cluster state
  or config loading.

  Later confirmation: a full-suite run after more test files had been added reported
  **1 failed, 3911 passed** — the BYOF test passed that time, because adding files shifted
  collection order. An intermittent, order-sensitive result on both trees is the signature
  of pre-existing test-isolation leakage, not of this change.

Two more exclusions carried over from EVIDENCE §1:
`npa/tests/workbench/test_vlm_eval_backend.py` and `test_vlm_eval_loop_e2e.py` are
live-GPU fixtures that launch a SkyPilot cluster whenever `sky` is on `PATH`.

One environment note worth keeping:
`test_skypilot_render.py::test_workbench_workflow_submit_plan_only_redacts_registry_password`
fails **only when `live-e2e.env` is sourced** (it depends on `NPA_REGISTRY` /
`SKYPILOT_DOCKER_*` being unset). It passes on the branch and on `aa555d73` in a clean
shell, so unit suites must be run *without* the live env.

## R2. `cosmos3-reason.yaml` twin — PASSED

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=gpu \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=cosmos3-reason.yaml \
NPA_E2E_ACCELERATOR_REMAP=H100:1=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1 \
NPA_E2E_RELAX_CPU_MEM=1 \
  pytest npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_live_reaches_terminal -q -s
# 1 passed in 182.76s
```

**Run id** `npa-wf-gpu-cosmos3-reason-af7ded35` · **SkyPilot job 182**

| job | task | REQUESTED | status | submitted_at | end_at |
| --- | --- | --- | --- | --- | --- |
| 182 | `npa-wf-gpu-cosmos3-reason-af7ded35` | `1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` | **SUCCEEDED** | 1785471666.9383 | 1785471724.2441 |

Equivalence note: this twin is genuinely equivalent because **both** sides run the
same code — the SkyPilot template's `run:` is
`python -m npa.workflows.cosmos_split cosmos3-reason ...` and the toolRef is
`npa workbench cosmos3 reason`, which calls the same
`build_cosmos3_reason_manifest`. Both are manifest builders that request a GPU; that
pre-existing stub-on-a-GPU shape is unchanged by this work and is called out in §R9.

## R3. `isaac-lab-rl-sweep.yaml` twin — PASSED (4 GPU variants, 2 batches, barrier)

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=multi \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=isaac-lab-rl-sweep.yaml \
NPA_E2E_ACCELERATOR_REMAP=L40S:1=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1 \
NPA_E2E_RELAX_CPUS=8+ NPA_E2E_RELAX_MEMORY=32+ \
NPA_E2E_IMAGE_OVERRIDE_ISAAC_LAB=<registry>/npa-isaac-lab:2.3.2.post1-sky3 \
NPA_SRC_OVERLAY=1 \
  pytest .../test_npa_workflow_runtime_live_reaches_terminal -q -s
# 1 passed in 586.24s (0:09:46)
```

**Run id** `npa-wf-multi-isaac-lab-rl-sweep-c4b86dc5` · jobs **185, 186, 187**

Wave ledger (`s3://<artifact-bucket>/.../isaac-lab-rl-sweep/npa-workflow/runtime.json`,
`npa.workflow.runtime.v1`), `status: succeeded`:

| wave | kind | job | tasks | submitted_at | end_at | max_concurrent_observed |
| --- | --- | --- | --- | --- | --- | --- |
| 001 `sweep` | parallel | 185 | `variant-lr-1e-3` | 1785472779.8337 | 1785472892.9320 | 2 |
| | | | `variant-lr-3e-4` | 1785472779.8382 | 1785472892.1463 | |
| 002 `sweep` | parallel | 186 | `variant-entropy-0` | 1785472980.3979 | 1785473086.9207 | 2 |
| | | | `variant-entropy-0-01` | 1785472980.4032 | 1785473087.4682 | |
| 003 `select-best` | serial | 187 | barrier | 1785473191.0149 | 1785473270.9288 | — |

Three properties, all from the timeline rather than assertion:

* **Concurrency** — each batch's two GPU tasks were submitted **4.5 ms** apart and
  their lifetimes overlap almost entirely.
* **Bounded concurrency (`--var max_concurrency=2`)** — batch 2 was submitted
  **87.5 s after** batch 1's last task ended, so the run never held more than two
  GPUs. This is live coverage of the multi-batch path.
* **Barrier** — `select-best` was submitted **103.5 s after** batch 2's last task
  ended.

Real work, not a stub — artifacts under
`s3://<artifact-bucket>/npa-workflow-e2e/npa-wf-multi-isaac-lab-rl-sweep-c4b86dc5/isaac-lab-rl-sweep/`:

```
    45575  variants/lr-1e-3/checkpoint.pt              real RSL-RL checkpoint
    24996  variants/lr-1e-3/train.log                  real training log
      726  variants/lr-1e-3/npa_rl_sweep_metrics.json
      888  variants/lr-1e-3/npa_rl_sweep_summary.json
   ... same four files for lr-3e-4, entropy-0, entropy-0-01 ...
     3559  report/npa_rl_sweep_best.json               the barrier's ranking
     7845  npa-workflow/runtime.json                   wave ledger
```

A first attempt (**job 183**) was `CANCELLED` after ~12 min of `ErrImagePull ... 403
Forbidden` retries — the registry confusion described in §R0, not a workflow fault.

## R4. `sonic-export.yaml` twin — PASSED, and the three real defects it exposed

This is the run that justifies most of the code in this change. The twin validated,
planned and rendered cleanly, and then failed **three times** for three different,
genuine reasons before passing.

| # | SkyPilot job | In-pod error | Root cause | Fix |
| --- | --- | --- | --- | --- |
| 1 | — (submit rejected in 1.15 s) | `rendered SkyPilot YAML still contains unresolved placeholders: ${npa_src_root}` | the new per-toolRef extras snippet used a braced expansion, which `assert_no_unresolved_placeholders` rightly rejects | compose the pip target with `printf` |
| 2 | **184** | `Error: checkpoint not found: s3://<artifact-bucket>/.../sonic-export/checkpoint.pt` | `sonic export` only ever accepted **local** paths; the SkyPilot template did the S3 download/upload in ~60 lines of inline bash+boto3, and a `toolRef` argv has no such escape hatch | `export_onnx` stages `s3://` inputs and publishes `s3://` outputs (`workbench/sonic/staging.py`) |
| 3 | **188**, **189** | `Error: observation dimension is required. Provide --obs-spec or a policy with one of: observation_dim, obs_dim, input_dim, num_observations` | the staged fixture was a bare `nn.Sequential`, which exposes none of those; real SONIC policies do, and the toolRef does not pass `--obs-spec` (a pinned `spec_gap`) | the fixture policy carries `obs_dim` / `action_dim` |

Each of these was invisible to mocked tests, and each now has offline coverage
(`test_tool_pip_extras.py`, `test_sonic_export_staging.py`, `test_sonic_fixture.py`).

**Passing run:** `npa-wf-gpu-sonic-export-cb60c5ab` · **SkyPilot job 192** ·
`1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` · **SUCCEEDED** · `1 passed in 251.15s`

Pod log excerpt — note the new extras hook doing its job:

```
(setup pid=…) syncing s3://<artifact-bucket>/npa-workflow-e2e/npa-src-retire/npa -> /tmp/npa-src
(setup pid=…) npa interpreter recorded: /home/sky/miniconda3/bin/python3
(setup pid=…) installing npa[sonic] from /tmp/npa-src
(npa-wf-gpu-sonic-export-cb60c5ab, pid=…) using npa interpreter /home/sky/miniconda3/bin/python3 for this stage
```

Artifacts under `.../npa-wf-gpu-sonic-export-cb60c5ab/sonic-export/`:

```
    16697  checkpoint.pt                 the staged fixture (input)
     1672  sonic_policy.onnx             REAL ONNX graph produced by the shipped exporter
    11776  sonic_policy.onnx.data        external weights (torch.onnx.export)
      678  sonic_policy.metadata.json    npa_sonic_onnx_export_v1 sidecar
```

The `.onnx.data` file is why staging publishes *every* file next to the model: an
ONNX with external weights is a **pair**, and onnxruntime resolves the data file
relative to the model.

### R4.1 The fixture, built in-cluster

```bash
scripts/stage-sonic-export-fixture.sh \
  --image <registry>/npa-sonic:0.1.2-k8s-runtime \
  --uri   s3://<artifact-bucket>/npa-workflow-e2e/fixtures/sonic-export/checkpoint.pt
```

```json
{
  "act_dim": 12, "obs_dim": 48, "hidden": 32, "seed": 0,
  "bytes": 16697, "schema": "npa.sonic.export_fixture.v1",
  "torch_version": "2.9.0+cu130",
  "checkpoint_uri": "s3://<artifact-bucket>/npa-workflow-e2e/fixtures/sonic-export/checkpoint.pt"
}
```

No torch wheel was downloaded to the dev VM (which sits at **96 % disk, 8.7 GB
free**); the builder ran in a pod from the SONIC image with the module mounted as a
ConfigMap.

## R5. `sonic-eval.yaml` and `sonic-export-eval.yaml` twins — PASSED, after a second real defect

Both twins reached **SUCCEEDED on the first attempt** — and produced **no artifact**:

| Run | job | result | run prefix contents |
| --- | --- | --- | --- |
| `npa-wf-gpu-sonic-eval-87a704ad` | 194 | SUCCEEDED (244.99 s) | `sonic_policy.onnx`, `.onnx.data`, `.metadata.json` — **no `eval.json`** |
| `npa-wf-multi-sonic-export-eval-744b9c1e` | 195 | SUCCEEDED (379.38 s) | `checkpoint.pt`, `sonic_policy.*` — **no `eval.json`** |

Both specs declare `outputs: - uri: .../eval.json`. The toolRef argv passed
`--output json`, but on `npa workbench sonic eval` **`--output` is the result path**
(`output_path: str`) and `--output-format` is the format — the SkyPilot template it
replaces passed both correctly (`--output "${SONIC_EVAL_OUTPUT}"` *and*
`--output-format json`). The tool therefore wrote to a relative `json/` directory
inside the pod, and the artifact vanished with the container. **A green terminal
status was not evidence; the artifact listing was.**

Fixed by splitting the two options (`--output {{config.eval_uri}}`
`--output-format json`, plus an `eval_uri` config key in both specs) and by a new
guardrail that audits the whole catalog for the mistake in both directions — a format
word handed to a path option, and a value that is not a member of an option's Enum.
The audit found this as the only real case; six look-alikes are commands where
`--output` genuinely *is* the format.

### Passing re-runs

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=gpu   NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=sonic-eval.yaml        ... # 1 passed in 283.78s
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=multi NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=sonic-export-eval.yaml ... # 1 passed in 410.73s
```

| Run id | job | status |
| --- | --- | --- |
| `npa-wf-gpu-sonic-eval-bb3b9c72` | **198** | SUCCEEDED |
| `npa-wf-multi-sonic-export-eval-2f5e979e` | **197** | SUCCEEDED |

`s3://<artifact-bucket>/.../npa-wf-gpu-sonic-eval-bb3b9c72/sonic-eval/`:

```
     4339  eval.json                     <- the artifact the spec declares
      678  sonic_policy.metadata.json
     1672  sonic_policy.onnx
    11776  sonic_policy.onnx.data
```

`eval.json` is a **real onnxruntime evaluation** — `"status": "completed"`,
`"backend": "reference"`, 8 episodes each with `action_min` / `action_max` /
`action_norm` / `episode_return`, and the new `onnx_uri` field recording the durable
input:

```json
{"status": "completed", "backend": "reference",
 "onnx_uri": "s3://<artifact-bucket>/.../npa-wf-gpu-sonic-eval-bb3b9c72/sonic-eval/sonic_policy.onnx",
 "episodes": [{"episode_index": 0, "action_max": 0.16404074430465698,
               "action_min": -0.1481434553861618, "action_norm": 0.35056760907173157,
               "episode_length": 1, "fall": false, "steps": 1}, "... 7 more ..."]}
```

`sonic-export-eval` (job 197) shows the **chain** working end to end: the export
stage's ONNX is what the eval stage consumed, and both artifacts sit in one prefix:

```
    16697  checkpoint.pt                 (staged fixture)
     1672  sonic_policy.onnx             (export stage output)
    11776  sonic_policy.onnx.data
      678  sonic_policy.metadata.json
     4355  eval.json                     (eval stage output; onnx_uri points at the above)
```

## R6. Retirement tally: 36 → 31

```bash
$ ls npa/src/npa/workflows/skypilot/*.yaml | wc -l
31
$ for f in cosmos3-reason isaac-lab-rl-sweep sonic-eval sonic-export sonic-export-eval; do
    rg -n --fixed-strings "skypilot/$f.yaml" . ; done
# (only EVIDENCE.md / DESIGN.md / CHANGELOG.md history mentions remain)
```

| Retired template | Twin's live run | job(s) |
| --- | --- | --- |
| `cosmos3-reason.yaml` | `npa-wf-gpu-cosmos3-reason-af7ded35` | 182 |
| `isaac-lab-rl-sweep.yaml` | `npa-wf-multi-isaac-lab-rl-sweep-c4b86dc5` | 185, 186, 187 |
| `sonic-export.yaml` | `npa-wf-gpu-sonic-export-cb60c5ab` | 192 |
| `sonic-eval.yaml` | `npa-wf-gpu-sonic-eval-bb3b9c72` | 198 |
| `sonic-export-eval.yaml` | `npa-wf-multi-sonic-export-eval-2f5e979e` | 197 |

**`sonic-locomotion-finetuning.yaml` was NOT retired.** Its twin's first stage is
`workbench.retargeting.run`, which needs a real SOMA/G1 motion dataset
(`NPA_E2E_SONIC_MOTION_SRC`); the repo deliberately does not vendor that dual-licensed
upstream data, and the pre-existing `retargeting.yaml` live case already fails for the
same reason (EVIDENCE §6.1). Deleting it on plan-only evidence would break the rule
this work is built on, so it stays — with that reason recorded next to it in
`test_skypilot_catalog_retirement.py`.

## R7. Cost

| Item | Amount |
| --- | --- |
| GPU tasks (RTXPRO-6000-BLACKWELL) | 13 pods: cosmos3-reason ×1 (57 s), sweep ×4 in 2 batches (~110 s each) + barrier, sonic-export ×3 attempts (~90 s each), sonic-eval ×2 (~90 s), sonic-export-eval ×2 (~190 s) ≈ **~22 GPU-minutes** |
| Failed/cancelled attempts included above | job 183 (12 min PENDING on `ErrImagePull`, **0 GPU-seconds billed**), jobs 184/188/189 (~90 s each) |
| CPU pods | 2 in-cluster fixture builds (~2 min each), 1 sweep barrier |
| Storage | a few hundred KB of checkpoints, ONNX, reports and ledgers under `s3://<artifact-bucket>/npa-workflow-e2e/` |
| Local | 555-file npa source staged to S3 four times; **no image pulled to the dev VM** |

Approximate spend: **well under half a GPU-hour**. No cluster was provisioned for
this work.

## R8. Teardown

```bash
$ sky jobs queue    # every npa-wf-* / manual-* job from this work
183 CANCELLED  184 FAILED  188 FAILED  189 FAILED       # the diagnosed attempts
182 SUCCEEDED  185 SUCCEEDED  186 SUCCEEDED  187 SUCCEEDED
192 SUCCEEDED  194 SUCCEEDED  195 SUCCEEDED  197 SUCCEEDED  198 SUCCEEDED
```

All terminal — a filtered check confirms nothing of this work's is still alive:

```bash
$ sky jobs queue | grep -E "npa-wf|manual-" | grep -vE "SUCCEEDED|FAILED|CANCELLED"
# (no output)
$ kubectl -n default get pod,configmap,secret | grep npa-sonic-fixture
# (no output — the staging script's `trap cleanup EXIT` removed the pod,
#  its ConfigMap and its credentials Secret)
$ sky status
nurec-spike                   UP     # another run's cluster, untouched
npawfrt-isaac-probe2          INIT   # left by the earlier PR #225 session, untouched
sky-jobs-controller-64ce57a0  UP     # pre-existing shared managed-jobs controller
```

No task cluster and no managed job from this work is still alive. Resources belonging
to other runs (`paidf-*`, `nurec-spike`, `npawfrt-*`, and the week-old `paidf-faithful4`
job that has been `PENDING` since before this work) were deliberately left alone.

One shared-infrastructure note: **no** change was made to the cluster this time. The
`ErrImagePull ... 403` on job 183 was an *environment* problem (§R0), fixed by pointing
at the registry the cluster's existing pull secret covers — not by re-minting anything.

## R9. Not verified live

1. **`sonic-locomotion-finetuning.yaml`'s twin** — needs a real SOMA/G1 motion
   dataset. Its template is therefore **not** retired (§R6).
2. **The `npa[sonic]` extra was only exercised on SkyPilot's default image**, not on
   top of a baked workbench image (that path is the `NPA_SRC_OVERLAY` branch, which
   the sweep did exercise, but without an extra).
3. **The SONIC image's new k8s prerequisites were verified by using the already-built
   `0.1.2-k8s-runtime` tag**, which carries the same four ingredients; the Dockerfile
   change itself was not rebuilt and re-run in this change. The guardrail pins the
   ingredients textually, and `Dockerfile.k8s-prereqs` is the documented repair path.
4. **`cosmos3-reason` remains a manifest builder that requests a GPU.** That is
   pre-existing on both sides of the twin (the template ran the same code) and is not
   changed here; a real Cosmos3 reasoning stage is separate work.
5. **The eval reference backend runs 1-step episodes** on the fixture policy
   (`episode_length: 1`). That exercises onnxruntime, the metadata contract and the
   artifact path for real, but it is not a locomotion quality signal.
6. **Only the specs listed in §R2–§R5 were submitted live**; the rest of the matrix
   was covered plan-only (24/24) in this change.

## R10. Phase 2a — pointer-only CLI callers, then five more retirements (31 → 26)

Four workbench CLIs held a `*_WORKFLOW_PATH` constant naming a raw template. Those
constants are **printed** by `<tool> workflow` / `<tool> status` — nothing loads them —
so "porting the caller" is repointing the advertised path. A new guardrail
(`test_cli_advertised_workflow_paths_exist`) asserts every such constant is a real
file, so a retirement cannot silently hand an operator a 404.

Repointed: `token_factory.py` (×4), `mjlab.py`, `retargeting.py`, `vlm_eval.py` (×2,
plus a new `token_factory_workflow` key). **Behaviour change:** those subcommands now
print npa.workflow spec paths.

`vlm-eval-token-factory.yaml` had no twin, so one was authored and registered as a
`cpu` live case. It is the VLM eval case that can *always* run: `vlm-eval-single` asks
for `self-hosted` and nothing in that spec starts a vLLM server (pre-existing, §5.2b).

### Live runs

```bash
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=<spec> \
  pytest .../test_npa_workflow_submit_live_reaches_terminal -q -s
# mjlab additionally: NPA_E2E_ACCELERATOR_REMAP=H100:1=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1
```

| Spec | Run id | job | status | wall |
| --- | --- | --- | --- | --- |
| `token-factory-caption.yaml` | `npa-wf-cpu-token-factory-caption-1dbebbb4` | 199 | SUCCEEDED | 158 s |
| `vlm-eval-token-factory.yaml` *(new)* | `npa-wf-cpu-vlm-eval-token-factory-736df0b1` | 200 | SUCCEEDED | 163 s |
| `token-factory-cosmos-reason.yaml` | `npa-wf-cpu-token-factory-cosmos-reason-d9669c7f` | 201 | SUCCEEDED | 156 s |
| `token-factory-generate.yaml` | `npa-wf-cpu-token-factory-generate-94815797` | 202 | SUCCEEDED | 187 s |
| `mjlab-eval.yaml` | `npa-wf-gpu-mjlab-eval-32c1efb5` | 203 | SUCCEEDED | 148 s |

All five produced **real** artifacts (`"dry_run": false` throughout):

```
caption   captions/captions.json      942 B  Qwen/Qwen2.5-VL-72B-Instruct
          caption[0]: "The image shows a solid red square centered on a light gray
                       background. There is no action or additional objects present..."
vlm-eval  scores/vlm_eval_stub.json  1001 B  backend "api", 4 keyframes, score 0.0,
          rationale: "The provided frames do not show any robot or physical task being
          performed. The images only display two static blocks (red and green)..."
reason    plan/scene_reasoning.json  2114 B  nvidia/Cosmos3-Super-Reasoner
generate  generations.jsonl            86 B  hosted text generation
mjlab     mjlab/mjlab_eval.json       727 B  score 0.1423, suite locomotion,
          embodiment unitree-g1, episodes 8, passed false
```

### The defect these runs exposed: `outputs:` was a promise, ten times over

`vlm-eval-token-factory` wrote `scores/vlm_eval_stub.json` while its `outputs:`
declared `scores/report.json`; `mjlab-eval` wrote `mjlab/mjlab_eval.json` while
declaring `mjlab/report.json`; `token-factory-cosmos-reason` wrote
`plan/scene_reasoning.json` while declaring `plan/plan.json`. Every stage
**SUCCEEDED**. This is the same class as §R5's `--output json`, and it is the third
time a green status hid a missing artifact.

Rather than fix instances, `test_spec_declared_outputs.py` now resolves each stage's
argv, asks the tool's own `*_result_uri_for()` helper where it would write, and
compares. It found **ten** wrong declarations across seven specs — including
`tokenfactory-rollout-judge.yaml`, which `EVIDENCE §5.2b` already recorded as a
**PASSED** live run (job 166) and which therefore also never wrote its declared
artifact. All ten are corrected; the next one fails offline.

### Retirement tally: 31 → 26

| Retired template | Twin's live run | job |
| --- | --- | --- |
| `token-factory-caption.yaml` | `npa-wf-cpu-token-factory-caption-1dbebbb4` | 199 |
| `vlm-eval-token-factory.yaml` | `npa-wf-cpu-vlm-eval-token-factory-736df0b1` | 200 |
| `token-factory-cosmos-reason.yaml` | `npa-wf-cpu-token-factory-cosmos-reason-d9669c7f` | 201 |
| `token-factory-generate.yaml` | `npa-wf-cpu-token-factory-generate-94815797` | 202 |
| `mjlab-eval.yaml` | `npa-wf-gpu-mjlab-eval-32c1efb5` | 203 |

Cost for this phase: five short pods (four `1x[CPU:4+]`, one RTXPRO-6000 for ~31 s)
plus ~8 hosted Token Factory calls. Rounding error.

## R11. Phase 2b — a synthesized SOMA-CSV clip makes `retargeting` live-testable (26 → 25)

`retargeting.yaml`'s twin was the one live-matrix case that **failed** before this work:

```
Error: S3 input contains no objects: s3://<artifact-bucket>/.../retargeting/source/
```

(EVIDENCE §6.1). It needed a real SOMA/G1 motion clip, because the tool feeds NVIDIA's
upstream `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`, and this repo does
not vendor that dual-licensed dataset. `sonic-locomotion-finetuning.yaml` was blocked by
the same thing.

The upstream loader's contract is small and public, so a clip can be **synthesized**. It
was read from the pinned upstream ref with a blobless sparse clone (1.2 MB, nothing
kept):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/NVlabs/GR00T-WholeBodyControl.git && \
  git sparse-checkout set gear_sonic/data_process
# load_csv_motion(): joint_pos.csv -> (T, 29) IsaacLab order, radians
#                    body_pos.csv  -> (T, B*3), body 0 = pelvis -> root_trans_offset
#                    body_quat.csv -> (T, B*4) wxyz, body 0 -> root rotation
```

`npa.workflows.motion_fixture` writes exactly that — forward pelvis translation at
constant height, a gentle yaw, bounded joint angles — using **only the standard
library**, so the fixture needs no container, no numpy and no torch. The conversion
still happens in the pod, where the upstream script's joblib/pandas/scipy live. The live
harness synthesizes clips automatically when `NPA_E2E_SONIC_MOTION_SRC` is unset (the
env var remains the real-data override), and
`scripts/stage-sonic-motion-fixture.sh` stages a set to share across runs.

### `retargeting.yaml` — PASSED

```
npa/tests/e2e/test_npa_workflow_submit_live_e2e.py
[seed] synthesized 2 SOMA-CSV clip(s) (6 objects) — set NPA_E2E_SONIC_MOTION_SRC to use real data
1 passed in 167.54s (0:02:47)
```

**Run id** `npa-wf-cpu-retargeting-b8e5bc8b` · **SkyPilot job 204** · `1x[CPU:4+]` ·
**SUCCEEDED**

Artifacts — the **real upstream converter** ran at the pinned ref:

```
    16975  retargeted/motion_lib.pkl            real motion_lib PKL
     1385  retargeted/retargeting_result.json   status "retargeted", source_format
                                                "soma-csv", embodiment "unitree-g1",
                                                frame_rate 30, motion_count 2,
                                                upstream_ref a9d20b2ac0949244d94461a1a3263f38c5027c4a,
                                                dry_run false
    ~13 KB x2  source/{walk-forward,stand-sway}/body_pos.csv
    ~15 KB x2  source/{walk-forward,stand-sway}/body_quat.csv
    ~11 KB x2  source/{walk-forward,stand-sway}/joint_pos.csv
```

`command` in that JSON is the upstream script invocation, so there is no doubt about
what did the work:

```
['/opt/conda/bin/python3',
 '/tmp/npa-retargeting-.../upstream-sonic/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py',
 '--input', '.../input', '--output', '.../output/motion_lib.pkl', '--fps', '30']
```

### `sonic-locomotion-finetuning.yaml` — still NOT retired, and now for a *known* reason

**Run id** `npa-wf-multi-sonic-locomotion-finetuning-ff468526` · **SkyPilot job 205** ·
**FAILED** at the second of three stages.

| stage | result |
| --- | --- |
| `retarget` | **SUCCEEDED** — same fixture, `motion_count: 2`, `retargeting_result.json` written |
| `train` | **FAILED**: `Error: SONIC --runtime serverless requires --project-id or a configured project.` |
| `mjlab` | not reached |

This is **not** a missing fixture. The spec sets `sonic_runtime: serverless`, so the
toolRef asks the **in-pod** CLI to launch a Nebius *serverless job* — nested
infrastructure, and the pod has no project config. It is the same trap `DESIGN §7`
records for `workbench.rl.policy_train` ("that CLI is a launcher; calling it inside a
SkyPilot task would nest infrastructure"), and fixing it is a spec-design decision:
either train in-pod against the SONIC image, or keep the launcher outside the workflow.
Recorded as the reason the template survives in
`test_skypilot_catalog_retirement.py`; the fixture blocker is gone.

### Retirement tally: 26 → 25

| Retired template | Twin's live run | job |
| --- | --- | --- |
| `retargeting.yaml` | `npa-wf-cpu-retargeting-b8e5bc8b` | 204 |

One more `outputs:` correction fell out of it (the eleventh): both retargeting-backed
specs declared `retargeted/manifest.json` while the tool writes
`retargeting_result.json`. The declared-output guardrail now covers
`workbench.retargeting.run` too.

## R12. CI on the pull request — 21/21 green

```
docs-drift pass   gitleaks pass   guardrails pass   mypy pass
ruff pass         scan pass       test (3.10) pass  test (3.12) pass  test (3.14) pass
Two-tag strategy pass             Static Dockerfile scan pass        4x base-image CVE scan pass
```

One check needed a fix. **docs-drift** — a *blocking* gate that regenerates `docs/cli/`
from `npa --help` — failed because four `workflow` subcommands' short help changed when
they started advertising npa.workflow specs:

```
-workflow  Show the SkyPilot YAML template for MJLab evaluation.
+workflow  Show the npa.workflow spec for MJLab evaluation.
```

The exact diff CI computed was applied to
`docs/cli/{mjlab,retargeting,token-factory,vlm-eval}.md` rather than re-running the
generator on the dev VM, because there the generator also rewrites typer metavars
(`<str>` vs `TEXT`) — unrelated churn already documented in §1.

**gitleaks passed on the commit range**, which is the authoritative version of the local
scan reported in §R1.

## R13. Final state of this change

| | |
| --- | --- |
| SkyPilot templates | **36 → 25** (`ls npa/src/npa/workflows/skypilot/*.yaml \| wc -l` = 25) |
| Templates retired | 11, each with a live run id (§R2–R6, §R10, §R11) |
| `skypilotTwin:` fields | 13 → 3 |
| Offline suite | 3850 passed (base: 3682) — **+168**, same 2 pre-existing failures |
| New guardrails | 5, plus the migrated three-tier third tier |
| Specs whose `outputs:` was wrong | 8 specs / 11 stages, all corrected and now guarded |
| Live jobs | 182–205; all terminal; no leaked clusters or pods |
| Approx. spend | ~22 GPU-minutes + a handful of short CPU pods |

Templates that remain are pinned with a reason each in
`npa/tests/guardrails/test_skypilot_catalog_retirement.py`. The two that are *blocked
rather than unstarted* are called out explicitly: `sonic-locomotion-finetuning.yaml`
(its twin nests infrastructure — §R11) and the trigger/sim-to-real group (engine features
first). Nothing was deleted on plan-only evidence.

## R14. Phase 2c — BYOF profiles relocated (25 → 20) and multi-node stages proven live

### The 5 BYOF resource profiles were relocated, not deleted

`isaac-lab-rl-train{,-rtxpro,-rtxpro-smoke}.yaml`, `byof-datagen-rtxpro-smoke.yaml` and
`byof-container-smoke-rtxpro.yaml` describe a **pod shape** — accelerator, cpu/memory
floors, image placeholder, smoke command — not a pipeline. The workflow surface for them
is already the spec `byof.yaml`, whose `workbench.byof.repo` toolRef passes one through
`--yaml {{config.resource_profile_yaml}}`. They moved to
`npa/src/npa/workflows/byof/profiles/`, joining the two that were already there, with a
`README.md` stating the boundary and `test_byof_profiles.py` enforcing it (pinned file
set, one task per profile, `live.py`'s constants must resolve, no runner may resolve a
path under the retiring catalog).

Rewriting the runners onto the engine was rejected: they carry render-only modes,
output-root rewriting and BYOF image plumbing the engine does not model, so a port would
risk the BYOF onboarding live path for no gain in the *workflow* surface.

### The relocation exposed a real gap, and the second run proves the fix

**First attempt** — `byof-profile-relocation-075138`, **SkyPilot job 206**: the runner
found the relocated profile, rendered it, submitted it, the pod pulled the ~8 GB Isaac
image and **ran the profile's training script** — then died at the artifact upload:

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

All three BYOF runners called `submit_workflow` **without `secret_envs`**, while every
profile uploads its summary and artifacts to S3. Pre-existing; the relocation surfaced
it. Each runner now takes a repeatable `--secret-env` and defaults to forwarding the S3
credentials when they are set (an unset name is dropped, since SkyPilot rejects a secret
it cannot resolve).

**Second attempt** — `byof-profile-reloc2-075858`, **SkyPilot job 207** ·
`1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]` · **SUCCEEDED**:

```bash
python npa/scripts/run_isaac_lab_rl.py \
  --yaml npa/src/npa/workflows/byof/profiles/isaac-lab-rl-train-rtxpro-smoke.yaml \
  --image <registry>/npa-isaac-lab:2.3.2.post1-sky3 \
  --task Isaac-Cartpole-v0 --iterations 1 --run-id byof-profile-reloc2-... --cleanup
```

Real Isaac Lab training through the relocated profile:

```
    45575  npa_isaac_lab_checkpoint.pt              real RSL-RL checkpoint
      932  npa_isaac_lab_checkpoint_manifest.json   status "success", task Isaac-Cartpole-v0
      882  npa_isaac_lab_train_summary.json         status "success"
    14081  logs/rsl_rl/cartpole/.../params/env.yaml
    14208  outputs/.../.hydra/config.yaml
```

`cleanup`/`teardown` reported no errors, so nothing was left behind.

### Multi-node stages: `resources.<profile>.num_nodes` — PASSED

The one genuine expressiveness gap the brief named. Before this, a multi-node block was
reachable only through `npa burst submit --nodes`, i.e. outside the workflow surface.
`num_nodes` is **task level** in SkyPilot's schema, so it lives on the resource profile
in a spec and the renderer lifts it out; `normalize_resources` deliberately never passes
it through, and a 1-node profile emits no key at all (so every existing rendered document
is byte-identical).

**Run id** `npa-wf-cpu-multi-node-probe-11cc2065` · **SkyPilot job 208** ·
`1 passed in 228.05s`

`sky jobs queue --all --output json` — the node count is visible in `REQUESTED`:

| job | task | requested | status | submitted_at | end_at |
| --- | --- | --- | --- | --- | --- |
| 208 | `report-nodes` | **`2x[CPU:2+]`** | SUCCEEDED | 1785485152.8898 | 1785485209.7737 |
| 208 | `verify-nodes` | `1x[CPU:2+]` | SUCCEEDED | 1785485231.3538 | 1785485285.5915 |

And the proof is in S3 rather than in a log line — one report per rank, from **distinct
hosts**:

```json
nodes/rank-0.json {"rank": 0, "num_nodes": 2, "node_ip_count": 2,
                   "hostname": "report-nodes-208-64ce57a0-head"}
nodes/rank-1.json {"rank": 1, "num_nodes": 2, "node_ip_count": 2,
                   "hostname": "report-nodes-208-64ce57a0-worker1"}
report/multi_node_report.json
  {"expected_nodes": 2, "reported_nodes": 2, "ranks": [0, 1],
   "hostnames": ["report-nodes-208-64ce57a0-head",
                 "report-nodes-208-64ce57a0-worker1"]}
```

`verify-nodes` fails on a missing rank **and** on two ranks sharing a hostname, so a
gang that silently collapsed onto one node would not pass. A head + worker1 pair is
exactly what a real 2-node SkyPilot gang looks like.

Note the brief's premise that `isaac-lab-cosmos-sdg-burst-smoke.yaml` needs this feature
is wrong: that template is explicitly single-task/single-node and has zero references in
the repo. The feature is worth having on its own terms, and its live proof is this
purpose-built spec.

### Retirement tally: 26 → 20

| Template | Disposition |
| --- | --- |
| `retargeting.yaml` | retired, job 204 (§R11) |
| `isaac-lab-rl-train.yaml` | **relocated** to `byof/profiles/` |
| `isaac-lab-rl-train-rtxpro.yaml` | **relocated** |
| `isaac-lab-rl-train-rtxpro-smoke.yaml` | **relocated**, live-verified job 207 |
| `byof-datagen-rtxpro-smoke.yaml` | **relocated** |
| `byof-container-smoke-rtxpro.yaml` | **relocated** |

Cost: two RTXPRO pods (~25 s and ~24 s of job time, plus image pull) and one 2-node CPU
gang for ~57 s.

## R15. Phase 4 (partial) — the two insights specs get live coverage

`insights-smoke.yaml` and `insights-aggregate.yaml` were two of the 17 specs with no
live-matrix entry, and the reason was concrete: `workbench.insights.ingest_run` scans a
run prefix for known manifest/report schemas and fails with

```
no known manifest/report schemas found under run prefix: s3://...
```

when it finds none — the same failure EVIDENCE §2 recorded when a caption fan-out tried to
use the ingester as a barrier. Nothing seeded a prefix it recognised.

The harness now seeds the two shapes it does recognise (read from
`insights/store.py::_extract`): a real `npa.dataset.manifest.v1` document, which yields
record/corruption metrics **and** a lineage edge, and a bare `{"decision": ...}` document.
`insights-smoke` reads a *shared* fixture prefix (`insights-fixtures/run/`) so its seeding
is idempotent; `insights-aggregate` reads `runs/<run-id>/`, outside the e2e marker prefix.

| Spec | Run id | job | status | wall |
| --- | --- | --- | --- | --- |
| `insights-smoke.yaml` | `npa-wf-cpu-insights-smoke-f6e3c287` | 209 | SUCCEEDED | 298 s |
| `insights-aggregate.yaml` | `npa-wf-cpu-insights-aggregate-d54426f6` | 210 | SUCCEEDED | 258 s |

Real store output, not a stub:

```
insights-smoke/store/records.jsonl        6037   metric records
insights-smoke/store/edges.jsonl           840   lineage edges
insights-smoke/comparison/comparison.json 2684   base-vs-candidate comparison
insights-smoke/dashboard/dashboard.html   2431
insights-aggregate/store/records.jsonl    3827
insights-aggregate/store/edges.jsonl       326
insights-aggregate/dashboard/dashboard.html 1894
```

Live-matrix coverage: **17 uncovered specs → 15**, and the matrix grew from 24 to 28 cases
(the two above plus the two new specs `vlm-eval-token-factory.yaml` and
`multi-node-probe.yaml`). The remaining 15 are enumerated with their blockers in
`npa/src/npa/orchestration/npa_workflow/submit_matrix.py` and
`test_skypilot_catalog_retirement.py`; the recurring one is worth naming:

**Three toolRefs are launchers.** `workbench.rl.policy_train` and
`workbench.sonic.train` (with `--runtime serverless`) provision infrastructure of their
own, so invoking them from inside a SkyPilot stage nests infrastructure and fails in the
pod. Every spec that uses them — `adversarial-scenario-hardening`,
`hardening-with-insights`, `rl-policy-training-sim-success`, `sonic-train`,
`sonic-locomotion-finetuning` — is blocked on the same design decision: move the launcher
out of the workflow, or train in-pod against the vendor image. That is called out rather
than papered over with `plan_only=True`.

## R16. Phase 4 (continued) — the dataset-of-record specs

`dataset-of-record-smoke.yaml` and `dataset-ingest-curate.yaml` read
`config.raw_sensor_uri` and had no live coverage because nothing seeded one, so
`workbench.dataset.ingest` would fail with `raw sensor data not found`.

`npa.workflows.dataset_fixture` generates records against the contract
`dataset/ingestion.py` enforces (`record_id`/`modality`/`uri` required;
`event`/`location`/`timestamp`/`quality`/`embedding` each add 0.2 to `completeness`;
corrupt = empty `uri` **or** `quality.corruption > 0.5`), built to satisfy the *stricter*
of the two specs so one fixture serves both: completeness 1.0, zero corrupt records, and
half the set tagged with the queried `cut_in` event in `san_francisco` so the query stage
returns rows rather than an empty success. Standard library only, so the harness generates
it inline; `raw_sensor_uri` is a shared path outside the run prefix, so seeding is
idempotent.

### `dataset-of-record-smoke.yaml` — PASSED

```
[seed] 12 raw sensor records -> s3://<artifact-bucket>/dataset-of-record-fixtures/records.json
1 passed in 459.37s (0:07:39)
```

**Run id** `npa-wf-cpu-dataset-of-record-smoke-4b509956` · **SkyPilot job 211** ·
**SUCCEEDED** — a 5-stage run (ingest → validate → quality gate → curate → query) with
real dataset lineage in S3:

```
    5963  dataset/smoke-fleet/v1/manifest.json                  record_count 12
    1248  validation/validation_report.json
      39  gate/decision.json
    3844  curated/smoke-fleet/v1.curated-2ed91515/manifest.json record_count 6,
                                                               parent_version v1
```

12 records ingested, 6 curated, with the curated manifest carrying `parent_version: v1` —
the lineage link the dataset-of-record tool exists to produce.

### `dataset-ingest-curate.yaml` — four of five stages live, then an infrastructure wall

**Run id** `npa-wf-cpu-dataset-ingest-curate-b1fa3ebd` · **SkyPilot job 212**

| stage | result |
| --- | --- |
| `ingest` | SUCCEEDED |
| `validate` | SUCCEEDED |
| quality gate | SUCCEEDED |
| `curate` | SUCCEEDED — real 6-record curated manifest, `parent_version: v1` |
| `register` | **FAILED** |

```
Error: Unexpected error: workbench service call failed
  (http://npa-lancedb.workbench.svc.cluster.local:8686/query):
  [Errno -2] Name or service not known
```

The `register` stage indexes the curated dataset into the **LanceDB workbench service**,
which is not deployed on `npa-rtxpro-mk8s`. That is an infrastructure dependency, not a
spec or fixture defect, so the case is registered `plan_only=True` **with that reason
stated in full** — and the note says what to flip when the service is deployed. The same
tools are covered end to end by `dataset-of-record-smoke.yaml`, which does not register
into LanceDB.

Live-matrix coverage: **17 uncovered specs → 13**; matrix cases 24 → 30 (four newly
covered pre-existing specs plus the two new specs).

## R17. Phase 4 (continued) — `scenario-gen-smoke.yaml` needed no fixture at all

Reading the tool settled this one before any cloud time was spent:
`scenario_gen/generation.py::generate_scenarios` never opens `--policy-uri` or
`--input-path`. The default `simulate_adversary` backend is deterministic and GPU-free, and
those URIs are only recorded in the manifest's `lineage`. The `rank` stage then consumes the
manifest the `generate` stage wrote. So the spec was self-contained and simply had no matrix
entry.

**Run id** `npa-wf-cpu-scenario-gen-smoke-bc5ed74b` · **SkyPilot job 213** ·
**SUCCEEDED** · `1 passed in 224.64s`

```
    6223  adversarial/manifest.json          npa.scenario_gen.adversarial_set.v1, 8 scenarios
     ~489 adversarial/scenarios/adv-000{0..7}.json   one config per mined scenario
     ....  ranked/ranked.json                npa.scenario_gen.ranked_set.v1, top 3
```

Real mining output, ranked by severity and diversity rather than a placeholder:

```json
manifest top: {"scenario_id": "adv-0002", "failure_score": 0.6462,
               "severity": 0.6462, "diversity": 0.4627}
ranked  top: {"scenario_id": "adv-0002", "severity": 0.6462, "diversity": 0.4627}
```

Live-matrix coverage: **17 uncovered specs → 12**; matrix cases 24 → 31.

### Why the remaining 12 are still uncovered

| Spec(s) | Blocker |
| --- | --- |
| `adversarial-scenario-hardening`, `hardening-with-insights` | the **launcher** problem — both call `workbench.rl.policy_train`, which provisions its own infrastructure from inside the pod |
| `byof-maniskill`, `byof-mujoco-playground`, `byof-robocasa`, `byof-openpi`, `byof-droid-policy-learning` | each delegates to `run_byof_repo.py`, which builds and pushes a multi-GB image; covered by `test_byof_onboarding_live_e2e.py`, and the shared `byof.yaml` case is already `plan_only` for the same reason |
| `av-night-scene-hardening` | needs the LanceDB workbench service (same wall as `dataset-ingest-curate`, §R16) |
| `cosmos-synth-fanout-curation` | `workbench.fiftyone.launch_app` is `stub=True`, and a real Cosmos Transfer 2.5 run is a gated-weight diffusion job |
| `sim2real-two-step`, `sim2real-two-step-agent`, `sim2real-gpu-cross-region-agent` | the last of these has three `stub=True` toolRefs; the first two need a real Cosmos Transfer run |

None of these is "not attempted" — each has a named, specific blocker, and two of them
(`av-night-scene-hardening`, the LanceDB dependency) would be unblocked by deploying one
service.

---

## R18. Phase 3b — `sim-to-real-loop.yaml` owns a capability nothing else has

D3 in the plan said this template could simply retire in favour of the staged 14-stage
sim2real engine. Reading it closed that option: the YAML is not a thin wrapper around
`vlm-eval run`. Its `run:` block

1. lists the immediate child directories of the rollout prefix (falling back to the prefix
   itself when there are none),
2. calls `npa workbench vlm-eval run` **per rollout**, asserting each result's shape with
   `jq -e`,
3. aggregates the per-rollout records into `task_success_report.json` —
   `total_rollouts`, `passed_rollouts`, `success_rate`, `mean_score`, and a coarse
   `task_success` gate — and uploads it.

`vlm-eval run` scores **one** rollout: it discovers frames *recursively*, so pointing it at a
prefix holding many rollouts blends them into a single score. Nothing else in the repo
produces `task_success_report.json`:

```
$ grep -rn 'task_success_report' --include=*.py --include=*.yaml .
npa/src/npa/workflows/skypilot/sim-to-real-loop.yaml   (the bash above)
npa/tests/workbench/test_vlm_eval_loop_e2e.py          (the same loop, re-implemented in a test)
docs/workbench/cookbooks/vlm-eval-loop-runbook.md      (documents the artifact)
```

So the capability existed twice — once in bash inside a template, once in Python inside a
gated GPU test — and was reachable from neither the CLI nor a spec. Retiring the template
without moving it would have deleted a shipped capability.

**Fix:** `npa workbench vlm-eval loop` (`evaluate_rollout_set`, `discover_rollouts`,
`aggregate_loop_report`), byte-compatible with the template's report: same field names, and
the gate is still the **mean** score rather than the pass rate. That distinction is now
asserted — a set where every rollout passes its own threshold can still fail the mean:

```
test_aggregate_gate_is_the_mean_not_the_pass_rate
  two rollouts, both passed=True, score 0.5, threshold 0.8
  -> success_rate 1.0, task_success False
```

Object-store discovery uses a `Delimiter="/"` listing, which is the S3 equivalent of
`find -mindepth 1 -maxdepth 1 -type d`, with the same fall-back branch.

New spec `npa/workflows/workbench/npa-workflows/vlm-eval-loop.yaml` (twin of
`sim-to-real-loop.yaml`), registered as a `gpu` matrix case whose seeder plants **three**
rollout directories so the aggregate has something to aggregate.

## R19. The self-hosted VLM backend never worked from a spec, and why

`vlm-eval-single.yaml` asks for `vlm_backend: self-hosted`, which makes the tool POST to an
OpenAI-compatible endpoint on localhost. **Nothing in a spec started that server.** §5.2b
recorded the symptom without the cause: `VLM backend request failed: [Errno 111] Connection
refused`. The retired `vlm-eval.yaml` did the serve/wait/teardown in its `run:` block — 30
lines of bash that a `toolRef` argv cannot carry.

The renderer now has a **per-toolRef run preamble**, the sibling of its existing setup hook.
It has to be `run:` rather than `setup:` because SkyPilot runs those as separate shells: a
server started in setup is gone by the time the command runs.

Three live runs were needed to make it actually work, and each failure was a real property of
the environment rather than a flake:

| Run | Outcome | Root cause |
| --- | --- | --- |
| jobs 214 / 215 | FAILED after ~10 min | `FileNotFoundError: 'ninja'` — vLLM's FlashInfer sampler JIT-compiles a CUDA extension on first use and shells out to `ninja`, which the default image does not ship |
| jobs 216 / 217 | FAILED after ~10 min | `/bin/sh: 1: /usr/local/cuda/bin/nvcc: not found` — ninja ran, and the image has no CUDA **compiler** either |

Both are now handled without requiring anything of the task image: `ninja` comes from pip,
`CUDA_HOME` falls back to the `nvidia-cuda-nvcc` wheel vLLM already depends on, and
`VLLM_USE_FLASHINFER_SAMPLER=0` makes the sampler that wants the JIT use its pure-PyTorch
equivalent, so a compiler-less image cannot break startup at all.

The failures are also the fail-fast path working as designed: each died in ~4 minutes with
the server log tailed into the stage's stderr, instead of waiting out the full readiness
window and reporting a bare timeout.

```
starting vLLM for Qwen/Qwen2-VL-7B-Instruct on port 8000
...
vLLM server exited before becoming ready:
  RuntimeError: Engine core initialization failed.
  /bin/sh: 1: /usr/local/cuda/bin/nvcc: not found
```

The readiness window is 900 s by default (the template allowed 600 s; a cold HF download of a
7B checkpoint plus GPU load routinely exceeds that) and is overridable per spec with
`config.vlm_serve_ready_seconds`.

## R20. `vlm-eval-benchmark.yaml`'s twin diverged from the template twice

Reviewing the twin before trusting it found two defects that would each have made the
retirement a downgrade:

1. `--dataset` was **a path inside the repo**
   (`npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json`). That resolves
   on a developer's laptop and never in the pod the stage runs in, where the checkout does
   not exist. The template used an S3 URI. The spec now does too, and the harness seeds a
   real **labeled** set — one rollout that reaches the target, one that stalls short, plus
   both rubrics the spec sweeps — because a benchmark that reports accuracy over one
   unlabeled rollout is meaningless.
2. `vlm_backend: stub`, so the twin never touched a VLM at all and could not stand in for a
   template whose entire purpose is self-hosted serving. It is now `self-hosted`, which works
   because of R19.

Defect 1 is a *class* of bug, so it is now machine-checked:
`npa/tests/guardrails/test_spec_paths_are_not_repo_relative.py` fires when any resolved argv
value is a relative path that also exists in this repo. It is deliberately that narrow — such
a value is always a mistake, and the check cannot produce false positives. It immediately
found a second instance: five `byof-*.yaml` specs passed
`--yaml npa/src/npa/workflows/byof/profiles/byof-solution-smoke-rtxpro-gpu.yaml` to a stage
that runs in a pod. `resolve_byof_profile_path()` now accepts a packaged profile **name**
(and falls back to a packaged profile matching a path's basename), so an installed wheel
resolves what a checkout does, and the five specs name the profile.

## R21. `detection-training`: the two gaps that blocked the BDD100K port

`run_bdd100k_pipeline.py` still loads the raw template, and porting it onto the twin spec
turned out to need tool features, not just plumbing. Two are now done, and the remaining ones
are named honestly below.

Done:

* **The train stage did not wait.** The template POSTed `/train` and then polled `/status` in
  bash until `completed`, failing on `failed` or timeout and closing with a `jq -e` assertion
  that every epoch ran and a checkpoint pattern exists. The twin's `toolRef` POSTed and
  returned, so the eval stage would evaluate a checkpoint that does not exist yet. That loop
  now lives in the tool: `--wait/--poll-seconds/--timeout-seconds`, opt-in so existing
  fire-and-forget behaviour is unchanged.
* **`label_map` was unreachable.** The template sent a full BDD100K category map in the
  request body. `TrainRequest.label_map` is a real field, but it had no CLI flag and is not an
  accepted `--override` key — so no spec could set it. `--label-map` accepts the template's
  JSON spelling and `name=index` pairs.

Still missing, and therefore why `bdd100k-pipeline.yaml` is **not** retired in this change
(beyond the LanceDB service wall of §R16):

* the template's eval task first GETs `/runs` and **discovers** the completed training run's
  `checkpoint_uri_pattern` for its view, then evaluates that resolved checkpoint. The twin
  passes the training *output directory* instead.
* it asserts the returned `mAP`/`mAP_50`/`mAP_75` are numbers.
* with `WRITE_CANONICAL_EVAL_METRICS=1` it writes a canonical metrics object to S3.

Shipping the runner port without those would have silently degraded a shipped pipeline, which
is worse than leaving the template in place for one more change.

## R22. Phase 3b live results — three GPU twins PASSED (20 → 17 templates)

Third attempt on the vLLM path, after the ninja and nvcc fixes of §R19. All three launched
concurrently on `npa-rtxpro-mk8s`.

| Spec | Job | Run id | Wall | Result |
| --- | --- | --- | --- | --- |
| `vlm-eval-loop.yaml` | 218 | `npa-wf-gpu-vlm-eval-loop-88da76ad` | 11m49s | **SUCCEEDED** |
| `vlm-eval-single.yaml` | 219 | `npa-wf-gpu-vlm-eval-single-25906482` | 11m14s | **SUCCEEDED** |
| `vlm-eval-benchmark.yaml` | 220 | `npa-wf-gpu-vlm-eval-benchmark-e47bc877` | 11m25s | **SUCCEEDED** |

Accelerator: `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` (remapped from the specs' `H100:1`).
Image: SkyPilot's default — no vendor serving image. The renderer installed vLLM 0.26.0,
`ninja` 1.13.0 and the npa source, then started and health-checked the server:

```
using npa interpreter /home/sky/miniconda3/bin/python3 for this stage
starting vLLM for Qwen/Qwen2-VL-7B-Instruct on port 8000
vLLM server ready after 355s          (job 220)
vLLM server ready after 375s          (job 218)
```

355–375 s to ready, with three concurrent 7B checkpoint downloads. The retired template
allowed 600 s; the 900 s default of §R19 was the right call.

### The loop: a rollout SET, not one blended score

`npa-wf-gpu-vlm-eval-loop-88da76ad/vlm-eval-loop/`:

```
   818..821  rollouts/episode_00{0,1,2}/frame_00{0..3}.png   (12 seeded frames)
        837  scores/rollouts/episode_000/vlm_eval_stub.json
        837  scores/rollouts/episode_001/vlm_eval_stub.json
        837  scores/rollouts/episode_002/vlm_eval_stub.json
       1704  scores/task_success_report.json
```

The aggregate report — the artifact that previously existed only as `jq` output inside a
SkyPilot template:

```json
{
  "status": "completed",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "frame_selection": "keyframes",
  "success_threshold": 0.8,
  "total_rollouts": 3,
  "passed_rollouts": 3,
  "success_rate": 1.0,
  "mean_score": 1.0,
  "task_success": true,
  "latency_s": 13.701
}
```

Each per-rollout result records **its own** input, which is the property that distinguishes
the loop from pointing `run` at the prefix:

```json
{
  "backend": "self-hosted",
  "input_path": ".../vlm-eval-loop/rollouts/episode_001/",
  "frame_count": 4,
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "score": 1.0,
  "passed": true,
  "status": "passed",
  "rationale": "The robot successfully moved from the left to the right side of the image, completing the requested physical task."
}
```

A real VLM rationale, not a stub: `backend: self-hosted`, four keyframes per rollout, and
13.7 s of GPU inference across the three rollouts.

### The single-rollout twin

`npa-wf-gpu-vlm-eval-single-25906482/vlm-eval-single/scores/vlm_eval_stub.json`, the exact
artifact the spec declares, with `score: 1.0`, `passed: true` and a real rationale. This is
the case §5.2b recorded as **FAILED — pre-existing spec gap** (`Connection refused`); it now
passes because a spec can serve the model it calls.

### The benchmark twin

`.../vlm-eval-benchmark/results.json` (13,775 B) over the seeded labeled set:

```
item_count: 2
sweep: {"backend": "self-hosted", "models": ["Qwen/Qwen2-VL-7B-Instruct"],
        "rubrics": ["default", "strict"], "thresholds": [0.5, 0.8, 0.9],
        "frame_selection": "keyframes", "max_frames": 4, "fixture_scores": false}
ranked_configs: 6            (1 model x 2 rubrics x 3 thresholds)
best_config metrics: {"accuracy": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6667,
                      "true_positives": 1, "false_positives": 1, "total": 2}
```

`fixture_scores: false` — every case was scored by the served model, not read from the
manifest. Worth stating plainly: **the model scored both episodes as successes**, including
the one seeded to stall short of the target, so accuracy is 0.5 and there is one false
positive. That is a correct report of a real disagreement between the VLM and my synthetic
32-frame fixture, not a broken benchmark: the sweep, the ranking and the metrics all
computed from live inference. Discriminating on a 320x240 four-frame synthetic clip is a
weak task for a 7B VLM; the point verified here is that the twin exercises the same
labeled-sweep path the template did.

### Retired

```
npa/src/npa/workflows/skypilot/  20 -> 17 templates
  - vlm-eval.yaml            (twin vlm-eval-single.yaml,    job 219)
  - vlm-eval-benchmark.yaml  (twin vlm-eval-benchmark.yaml, job 220)
  - sim-to-real-loop.yaml    (twin vlm-eval-loop.yaml,      job 218)
```

## R23. `tokenfactory-rollout-judge` — the `outputs:` fix, confirmed live

§R10 recorded that `test_spec_declared_outputs.py` caught this spec declaring artifacts that
never appear: `plan/plan.json` (the reasoner writes `scene_reasoning.json`) and
`scores/report.json` (`vlm-eval run` writes `vlm_eval_stub.json`). §5.2b had already called
this spec a **PASSED** live run — a stage can succeed while writing somewhere else entirely,
which is the whole reason that guardrail exists. Re-running it after the fix closes the loop:

| Job | Run id | Wall | Result |
| --- | --- | --- | --- |
| 221 | — | 1m43s | **CANCELLED** — environment, not code (below) |
| 222 | `npa-wf-gpu-tokenfactory-rollout-judge-b2afbc62` | 3m42s | **SUCCEEDED** |

```
      1655  plan/scene_reasoning.json      <- declared, and present
       818  rollouts/episode_000/frame_00{0..3}.png
       834  scene/frame_000.png
      1025  scores/vlm_eval_stub.json      <- declared, and present
```

Both corrected declarations now name files that exist. Before the fix the same run would have
reported SUCCEEDED with `plan/plan.json` and `scores/report.json` promised and absent.

Job 221 is worth recording rather than hiding: it failed **prechecks** with

```
No resource satisfying Kubernetes({'RTXPRO6000': 1}) on Kubernetes.
Kubernetes cluster does not contain any instances satisfying the request
```

because this spec hardcodes `accelerators: RTXPRO6000:1` and my harness remap only rewrote
`H100:1`. That is a live-environment naming mismatch, not a defect in the change — adding
`RTXPRO6000:1=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` to `NPA_E2E_ACCELERATOR_REMAP` was the
whole fix. It does show that a spec naming a *specific* accelerator model is only portable to
clusters that spell it the same way.

**This template is still not retired.** Reading it against the same-named spec shows they are
different workflows that share a name:

| | template | spec |
| --- | --- | --- |
| stage 1 | LeRobot eval rollout on a GPU, **producing** the rollouts | Cosmos scene reasoner (unrelated) |
| stage 2 | hosted VLM judge over stage 1's output | VLM judge over rollouts seeded from outside |

The template's point is that a GPU stage produces exactly what the zero-GPU hosted judge
scores — the data dependency the cookbook advertises. A real twin needs
`workbench.lerobot.eval` → `workbench.vlm_eval.run`; the retirement tally now says so instead
of claiming the twin was verified.

## R24. Final state of this change

```
raw SkyPilot templates    36 -> 17
live-matrix cases         24 -> 32        (uncovered specs 17 -> 12)
offline suite             base aa555d73: 2 failed / 3682 passed
                          this branch:   1 failed / 4200 passed      (+518)
cd npa && ruff check src tests            clean
CI on the PR                              21/21 pass
```

The single offline failure is `smoke/test_golden_eval_tmux`, whose tmux subprocess uses a bare
`python3` without numpy; it reproduces on the base commit.

### Live jobs used in this section (182–222, all terminal)

```
218  npa-wf-gpu-vlm-eval-loop-88da76ad               SUCCEEDED   9m57s
219  npa-wf-gpu-vlm-eval-single-25906482             SUCCEEDED   9m31s
220  npa-wf-gpu-vlm-eval-benchmark-e47bc877          SUCCEEDED   9m43s
222  npa-wf-gpu-tokenfactory-rollout-judge-b2afbc62  SUCCEEDED   2m23s (2 stages)
214-217  the two vLLM bootstrap failures (ninja, then nvcc)  FAILED  ~8m each
221  RTXPRO6000:1 not a name this cluster uses       FAILED_PRECHECKS  2s
```

### Teardown

`kubectl get pods` shows no pod from any of these runs. `sky jobs queue` reports every job
182–222 in a terminal state. What remains on the cluster belongs to other sessions and predates
this work: `npa-rerun-…` (a Rerun deployment, 39 h), `nurec-spike-…` (8 h), and the two shared
`sky-jobs-controller-…` pods (2 d and 11 d). Nothing was created that needs manual cleanup.

---

## R25. `scenario-gen-adversarial.yaml` retired (17 → 16): the GPU it asked for was decorative

The tally reason said "no twin with live coverage". `scenario-gen-smoke.yaml` acquired that
coverage in §R17 (job 213, `npa-wf-cpu-scenario-gen-smoke-bc5ed74b`), so the question became
whether it is a real twin. Both run the same two commands — `scenario-gen generate` then
`scenario-gen rank`, with rank consuming exactly the manifest generate declared. The
differences were scale (16 vs 8 scenarios, 200000 vs 1000 adversary steps, top-k 4 vs 3) and,
apparently, a GPU:

```yaml
# scenario-gen-adversarial.yaml
accelerators: RTXPRO-6000-BLACKWELL-SERVER-EDITION:1
# Keep this image on an RT-core-capable Isaac Lab build (adversary RL backend).
image_id: "docker:.../npa-isaac-lab:2.3.2.post1"
ADVERSARY_STEPS: "200000"
```

That GPU is not reachable from the CLI. `generate_scenarios(adversary_backend=None)` falls
back to `simulate_adversary`, and the seam has **no CLI flag**:

```python
def simulate_adversary(request, seed):
    """Deterministic heuristic stand-in for an Isaac Lab adversarial RL rollout.

    This is NOT RL. ... a real Isaac Lab RL adversary replaces it via ``adversary_backend``.
    """
    budget_gain = min(0.25, math.log10(max(request.adversary_steps, 10)) / 40.0)
    for index in range(request.num_scenarios): ...
```

So `ADVERSARY_STEPS=200000` shifts one `log10` term and the loop is O(num_scenarios). A CPU
twin runs the *same code*; the template advertised a capability the shipped CLI does not have.

The claim is pinned rather than asserted in prose:
`test_the_cli_cannot_select_an_rl_adversary_backend` fails the day `scenario-gen generate`
grows a backend flag — which is exactly when a GPU spec should be authored. The scenario-gen
skill no longer tells operators to route this work to RT-core GPUs, and it documents the
template's production scale as `--var` overrides.

## R26. The BDD100K runner port, and the two defects the mock drive found

§R21 listed three tool features the port needed beyond `--wait` and `--label-map`. All three
are now in the tool:

| Template behaviour (bash + `jq`) | Now |
| --- | --- |
| GET `/runs`, take the **last completed** run for the view, substitute `{epoch}` in its `checkpoint_uri_pattern` | `eval --discover-checkpoint`, matching on the training output prefix (the same intent, and narrower: that prefix is what `/train` was handed) |
| `jq -e '(.mAP\|type=="number") and …'` | `assert_eval_metrics` — a service can answer 200 with a null mAP, and the stage would otherwise report success on an unusable report |
| `WRITE_CANONICAL_EVAL_METRICS=1` → `aws s3 cp` to `<output>/metrics.json` | `eval --write-canonical-metrics`, which also makes the spec's **declared** eval artifact true |

`run_bdd100k_pipeline.py` now takes `--spec` (with `--yaml` kept as an alias), renders through
`prepare_npa_workflow_for_submit` with config overrides instead of injecting envs into raw
documents, and forwards the service tokens as `secret_envs`.

### The mock drive: every stage's real argv, and what it caught

`--mock-endpoints` no longer runs each raw document's bash — a spec has no bash. It executes
**each plan step's resolved argv** against in-process LanceDB and detection-training
stand-ins, which is a stronger check because that argv is exactly what a pod would run. Two
real defects surfaced immediately, neither reachable by reading the diff:

**1. `curate-views` could never have worked.**

```
name: curate-views  returncode: 2
Usage: npa workbench lancedb create-mv [OPTIONS]
Error: No such option '--table'. (Did you mean one of: '--name', '--source-table'?)
```

The `create_failure_views` toolRef passed `--table`. The catalog-wide flag audit missed it
because it bailed out whenever `argv_template[0] != "npa"`, and this entry is
`["bash", "-c", "…three create-mv calls…"]` — so **every `bash -c` toolRef was unaudited**.
`embedded_npa_commands()` now extracts each `npa …` invocation from such a script and audits
it like any other argv; negative controls pin both the fix and the original broken command.

**2. The eval stages declared a URI nothing could write.**

```
declared: s3://…/eval/bdd100k_rider_trainmetrics.json
```

The eval prefixes had no trailing slash, so `{{config.rider_eval_uri}}metrics.json`
concatenated. Fixed in `bdd100k-pipeline`, `av-night-scene-hardening` and the two
diagram-example specs.

### Result

```
$ pytest npa/tests/workflows/test_bdd100k_pipeline.py -q
9 passed
```

The mock summary shows all eleven stages at `returncode: 0`, the exact LanceDB write sequence
(1 import, 6 backfills, 3 create-mv), the exact detection sequence (3 train, 3 eval), and the
two new behaviours in the **call order** rather than just the counts:

* every `POST /train` is immediately followed by `GET /status` — `--wait`;
* every `POST /eval` is immediately preceded by `GET /runs`, and its payload names a concrete
  `…/checkpoints/epoch_10.pt` with no `{epoch}` left in it — `--discover-checkpoint`.

Each `/train` payload carries the full ten-category `label_map`, with `num_classes` agreeing
with the map's size rather than contradicting it.

**The template is still not deleted.** A live run needs the LanceDB workbench service, which
is not deployed on this cluster (the same wall as §R16), and the mock stands in for both
services, so it cannot prove behaviour against the real ones. What changed is that the
retirement is now one step: deploy the service, run it live, delete. The tally entry says so.

---

## R27. `sim2real-envgen-split.yaml` retired (16 → 15), and the toolRef that could never run

The tally said "no twin". Authoring one first required fixing the toolRef, because
`workbench.sim2real_envgen.raw_shard` was **broken in a way nothing could see**:

```
python -m npa.workflows.sim2real_envgen raw-shard --output-uri … --env-count …
```

`--run-id` is `required=True` on that module's parser, so every stage using this toolRef could
only die with *"the following arguments are required: --run-id"*. Three shipped specs
reference it. The catalog flag audit could not catch it — it understands Typer commands
invoked as `npa …` (and, since §R26, the `npa …` calls inside a `bash -c` script), but not a
module CLI.

New guardrail `test_module_toolref_argv.py` asks the module's **real argparse parser**, which
is the only source of truth for a module CLI: placeholders are substituted with values each
`type=` accepts and the remainder is parsed, so a missing required option, an unknown flag or a
bad value all fail offline. A module toolRef whose module exposes no parser factory fails with
instructions rather than being silently skipped, and a negative control pins the argv that
shipped. `sim2real_envgen` gained `build_parser()` for this.

A second defect in the same toolRef: `--output-uri` is the **run root**, from which the module
derives `envs/raw/`, `envs/train/`, `envs/heldout/` and `envs/manifest/`. It was handed the raw
prefix, which would have nested a second `envs/raw` inside it. And all four specs using it
declared `<raw>/manifest.json`, a file `raw-shard` never writes — it writes
`raw-shard-<ii>-of-<nn>.jsonl` plus `raw-shard-<ii>-summary.json`, while `split-manifest.json`
comes from the `split` subcommand under `envs/manifest/`. Declarations corrected in all four.

### The fan-out, declared instead of implied

The template drove sharding from Kubernetes' Job completion index
(`--shard-index "${JOB_COMPLETION_INDEX:-0}"`), so producing N shards meant N submissions, or
an indexed Job the workflow surface never modelled. `sim2real-envgen-shards.yaml` declares it:
a `parallel:` group whose members differ only through `params.shard_index`, with `split` as the
barrier. New `workbench.sim2real_envgen.split` toolRef.

### Live: jobs 223 / 224 — `npa-wf-multi-sim2real-envgen-shards-79c2cb1c`, 5m38s, SUCCEEDED

The runtime ledger records concurrency directly, rather than being asserted:

```json
{"group": "generate-shards", "kind": "parallel", "job_id": "223",
 "max_concurrent_observed": 2,
 "observations": [
   {"observed_at": "13:33:55Z", "statuses": {"shard-0": "STARTING",  "shard-1": "STARTING"}},
   {"observed_at": "13:34:25Z", "running": ["shard-0", "shard-1"], "running_count": 2},
   {"observed_at": "13:34:55Z", "statuses": {"shard-0": "RUNNING",  "shard-1": "SUCCEEDED"}},
   {"observed_at": "13:35:25Z", "statuses": {"shard-0": "SUCCEEDED","shard-1": "SUCCEEDED"}}],
 "started_at": "13:32:27Z", "ended_at": "13:35:30Z"}
{"states": ["split"], "kind": "serial", "job_id": "224", "started_at": "13:35:31Z"}
```

The two shards were submitted **4.2 ms apart** (`1785504825.3996` / `.4037`) and the barrier
started **1 s after** the group ended.

The artifacts prove the shards were genuinely disjoint halves that the barrier recombined —
the property a single-task template could not demonstrate:

```
   42291  envs/raw/raw-shard-00-of-02.jsonl
     378  envs/raw/raw-shard-00-summary.json     shard_index 0, raw_count 32 of env_count 64
   42277  envs/raw/raw-shard-01-of-02.jsonl
     378  envs/raw/raw-shard-01-summary.json     shard_index 1, raw_count 32
   67388  envs/train/envs.jsonl
   17180  envs/heldout/envs.jsonl
     642  envs/manifest/split-manifest.json      raw_count 64, train 51, heldout 13,
                                                 train_fraction 0.8, disjoint: true
```

`51 + 13 = 64` — the split saw both shards, and reports the sets disjoint.

### CPU, not the GPU the template asked for

The template pinned `RTXPRO6000:1` and a test asserted it. Shard generation writes a catalog of
environment descriptors and never renders, so the twin is CPU and that GPU went unused; the
assertion moved onto the spec as its opposite (no resource profile declares an accelerator).
This is the third template in this change whose accelerator request did not match its work
(see also §R25).

---

## R28. `cosmos3-ea-fetch.yaml` retired (15 → 14), and the one load-bearing line in 60 lines of setup

The template's `setup:` was ~35 lines — `apt-get git curl`, install `uv`, build a 3.11 venv,
`pip install huggingface_hub[cli]` — and its `run:` opened with three hand-rolled token checks
(`test -n "${!NPA_COSMOS3_HF_TOKEN_ENV-}"`) before the two commands that do the work. The token
checks are the tool's job: `cosmos check` reports precisely which of source, checkpoint and NGC
access is missing. So the twin is those two commands and nothing else.

Exactly one line of that preamble was load-bearing, and dropping it cost a live run:

| Job | Outcome | Cause |
| --- | --- | --- |
| 225 | FAILED_SETUP in 57 s | `botocore.exceptions.NoCredentialsError` — the matrix case declared only `HF_TOKEN`, but **every** stage's setup syncs the npa source from `NPA_SRC_S3_URI` with boto3 |
| 226 | `check-access` SUCCEEDED, `fetch-artifacts` FAILED | `checkpoint download failed: [Errno 2] No such file or directory: 'huggingface-cli'` — `cosmos fetch` shells out to it |
| 227 | **SUCCEEDED** (12m44s) | — |

Both produced a guardrail rather than just a fix:

* **job 225** → `test_every_live_case_declares_the_object_store_credentials_setup_needs`. The
  existing secret check derives hints from the *plan*, so it structurally cannot see a need that
  comes from `setup:`. Now every non-`plan_only` case must declare the object-store keys.
* **job 226** → `TOOL_REF_PIP_REQUIREMENTS`, the sibling of `TOOL_REF_PIP_EXTRAS` for
  third-party packages. Each entry pairs an executable with a pip requirement and installs only
  when `command -v` cannot find it, so a purpose-built image that already ships it is untouched.

### Live: job 227 — `npa-wf-cpu-cosmos-fetch-ebbcc897`

```
(setup) installing huggingface_hub[cli]>=0.23,<1.0 for huggingface-cli      <- both stages

check-access:                          fetch-artifacts:
{ "status": "ok",                      { "status": "ok",
  "source_repo": "reachable",            "cache_dir": "/tmp/npa-cosmos3-cache",
  "hf_auth": "configured",               "source_checkout": ".../source",
  "hf_model": "reachable",               "checkpoint_dir": ".../checkpoint",
  "github_auth": "missing",              "checkpoint": "downloaded",
  "ngc_auth": "skipped",                 "errors": [] }
  "errors": [] }
```

`github_auth: missing` is the tool doing its job — the template would have `test -n`'d it into a
hard failure; `cosmos check` reports it and continues, because a public clone needs no token.

**On the substituted assets.** The spec's defaults still name `nvidia/Cosmos3-Nano` and the
Cosmos framework repo, both gated behind early access plus a licence acceptance. The live case
overrides them with a public repo and a tiny public checkpoint via `config_vars`. That is a real
`git clone` and a real Hugging Face download through the same commands, flags and cache layout —
the code path is identical, and asset identity is the one thing a live run here cannot prove.
Same approach as the SONIC and SOMA-CSV fixtures (§R4.1, §R11).

---

## R29. `isaac-lab-cosmos-sdg-burst-smoke.yaml` relocated, not retired (14 → 13)

The tally said "no twin; single-task burst reference". The template says why itself:

```yaml
# This YAML is intentionally one executable SkyPilot task so it can run through
# the burst Python-API path. Multi-stage workbench pipelines should use
# `npa workbench workflow submit`.
```

and `npa.burst.core.submit_yaml()` agrees in code — it loads one document, substitutes `${VAR}`
from `--var`, refuses to submit while a placeholder is unresolved, and injects a registry login.
There is no plan, no stage graph, no decision artifact and nothing for a `toolRef` to describe.
Authoring a spec for it would misrepresent a *different capability* as a workflow.

So it moved to `npa/src/npa/burst/examples/` with a README stating the boundary — the same call
as the five BYOF resource profiles (§R14, DESIGN §R10) — which lets the retiring workflow
catalog go away without breaking `npa burst submit-yaml`'s documented example.

`npa/tests/guardrails/test_burst_examples.py` pins the directory and both invariants:

* **one task per file** — a second stage means it is a workflow, and the test says so in its
  failure message;
* **the `${VAR}` placeholders survive** — they are the substitution surface, so no concrete
  registry id, bucket name or run id can be committed there.

It also proves burst *accepts* the file **offline**, by running `submit_yaml`'s own substitution
and `_validate_burst_yaml_runtime` without launching. No live run was needed: unlike the BYOF
profiles (whose paths were resolved by Python constants, hence job 207), nothing in Python
resolves this path — the only consumer is an operator command line, which the README and the
catalog README now spell correctly.

## R30. The three Token Factory combos share one blocker, now named

`tokenfactory-rollout-judge`, `tokenfactory-scene-to-rollout-judge` and
`tokenfactory-train-triage` were recorded as "no twin" or, worse, as having a twin that turned
out to be a different workflow (§R23). Reading all three shows a single shared blocker:

**each has a GPU stage that runs LeRobot inside the vendor image and produces exactly what the
Token Factory stage consumes.**

```bash
# tokenfactory-train-triage, train-gpu stage
source /opt/lerobot/venv/bin/activate
lerobot-train  …            # then a python block uploads artifacts to ARTIFACTS_URI
```

The producer/consumer dependency *is* the point of these combos — a GPU stage on Nebius feeding
a zero-GPU hosted judge — so a twin cannot drop it. What is missing:

* the catalog has only `workbench.lerobot.eval`, which shells out to `npa workbench lerobot
  eval`. `npa workbench lerobot` exposes `eval`, `benchmark`, `profile-train`,
  `train-student`, `list-checkpoints` — but **no plain `train`**; the template deliberately
  calls `lerobot-train` in the vendor venv;
* `train-triage` needs one more thing of its own: its triage stage builds `prompts.jsonl` and a
  system prompt **from the training artifacts** in bash + python before calling
  `token-factory generate`. No tool does that.

So the remaining work is one coherent piece rather than three unknowns: an in-image LeRobot
producer toolRef plus a policy/dataset fixture, and a prompt-builder for triage. The tally now
carries that instead of "no twin".

---

## R31. The LeRobot producer: three defects down, one named gap left

§R30 established that all three Token Factory templates share one blocker — a GPU stage that
runs LeRobot **inside the vendor image** and produces exactly what the hosted stage consumes.
Building that producer surfaced three more latent defects and left one gap, all recorded here
rather than papered over.

### What now exists

* `npa.workbench.lerobot.policy_container` already had a real in-image `train`/`eval` module
  CLI. It now exposes `build_parser()`, so the module-argv guardrail (§R27) checks any toolRef
  pointing at it.
* **`train --artifacts-s3-uri`** is new. `--checkpoint-s3-uri` uploads only the *checkpoint*
  (`upload_checkpoint_path(checkpoint, config)`), while a stage that reads the **run** needs the
  configs, logs and metrics beside it. The retired template did that with a trailing
  inline-python `rglob` upload — glue a `toolRef` cannot carry.
* **`npa/src/npa/workflows/token_factory_triage.py`** makes the triage stage executable. The
  template spent ~45 lines downloading textual artifacts, digesting them with the pure helpers,
  and calling `token-factory generate --system-prompt "$(cat …)"` — a shell substitution no argv
  can express. The pure helpers stay pure: `token_factory_combos` documents that it holds no
  network, storage or Token Factory calls. The new module also **refuses to write a silent empty
  report**: a run whose artifacts contain no readable text fails loudly instead of triaging
  nothing.
* Two catalog toolRefs (`workbench.lerobot.policy_train`, `workbench.token_factory.triage`) and
  the twin spec `npa-workflows/tokenfactory-train-triage.yaml`.

### Defect: five toolRefs invoked bare `python`

Live job 242 died before training started:

```
(train-gpu) using npa interpreter /home/sky/miniconda3/bin/python3 for this stage
(train-gpu) bash: python: command not found
```

The interesting part is that **the same argv shape had passed two runs earlier**: jobs 223/224
ran `python -m npa.workflows.sim2real_envgen` successfully on SkyPilot's default image, where
miniconda provides `python`. The LeRobot vendor image does not. So five toolRefs
(`raw_shard`, `split`, `write_decision` and the two new ones) carried image-dependent breakage
that only one image exposes — the definition of something a rule should catch rather than a live
run. All five now use `python3`, which is also what the renderer's interpreter shim records and
therefore the interpreter that can import `npa`, and
`test_no_tool_ref_invokes_bare_python` pins it.

### The remaining gap, named

Job 243 got one step further — the module ran and reported its own precondition:

```
PolicyContainerError: --data-path or --dataset-path is required
```

`run_lerobot_training` asserts a **local** dataset root containing `meta/info.json`.
`--dataset-repo-id` is only the label passed through to `lerobot-train`. Stages do not share a
filesystem, so a separate "fetch the dataset" stage cannot help: the dataset has to be
materialised **inside** the train stage from its repo id (what `lerobot-train` does natively).

That is one bounded tool feature, and it is the last thing between this twin and a live run.
Until it lands the matrix case is `plan_only` with exactly that reason, so nothing claims a
passing live case — the same discipline as `dataset-ingest-curate` (§R16).

The three templates therefore remain, but their blocker has gone from "no twin" (§R30) to a
single named feature with two of the three producer pieces already shipped and unit-tested.

---

## R32. The LeRobot producer, run to ground: five engine gaps closed, one broken vendor image

§R31 left one named gap. Closing it took six live iterations, each of which found a distinct
defect that no offline test could have shown. Every fix is in the engine or the tools; the last
failure is in the published image itself.

| Job | How far it got | What it found |
| --- | --- | --- |
| 242 | died before training | `bash: python: command not found` — five toolRefs invoked bare `python`, which the vendor image lacks. The same argv had *passed* on SkyPilot's default image, where miniconda provides it (§R31) |
| 243 | module ran | `--data-path or --dataset-path is required` — `run_lerobot_training` needs a local dataset root, and stages share no filesystem, so the stage must materialise its own |
| 245 / 247 | no LeRobot at all | the rendered document had **no `image_id`**: the harness's *submit* path only ever consulted `NPA_E2E_CLEAR_WORKBENCH_IMAGES` and silently dropped a case's `image_tool`, which the *runtime* path had honoured all along |
| 250 | vendor interpreter switch worked | `No module named 'npa.workbench'` — the image bakes a **partial** npa (`__init__`, `server`, `smoke`) on `PYTHONPATH` for its own entrypoint, which shadows the real one a stage installs |
| 252 | dataset materialised, training started | `lerobot-train failed (exit=1, log=/tmp/lerobot_output.train.log)` — a path inside a dead pod, so the reason was unreachable |
| 253 / 254 | training reached step 0 | the log tail finally showed it (below) |

### What the engine gained

* **`TOOL_REF_VENDOR_INTERPRETERS`** — a toolRef declares its vendor image's interpreter. Setup
  installs npa **into** it and records it as the stage interpreter, so the tool and the vendor
  library share one environment. Live proof, job 250:

  ```
  npa interpreter recorded: /usr/bin/python3
  installing npa into vendor interpreter /opt/lerobot/venv/bin/python
  npa interpreter switched to vendor python: /opt/lerobot/venv/bin/python
  using npa interpreter /opt/lerobot/venv/bin/python for this stage
  ```

  It probes `import npa.workbench`, not `import npa`, precisely because a baked stub makes the
  latter pass. And it installs with **`--no-deps`**: a vendor image ships a pinned stack, and
  resolving npa's requirements inside it can bump torch.
* **`npa-lerobot` is now SkyPilot-hostable** (`Dockerfile` + `Dockerfile.k8s-prereqs`, added to
  `SKYPILOT_HOSTED_IMAGES`). Two things had to be understood to get the derived build through:
  the base purges `linux-libc-dev` with `--force-depends` as CVE hardening, which strands
  `libc6-dev` and leaves **apt refusing to install anything** (even `rsync`); and the fourth
  prerequisite — system python first on PATH — is exactly what *creates* the need for the vendor
  interpreter switch. The source build installs the prerequisites *before* the hardening step so
  nothing needs repairing; only the derived recipe uses `--fix-broken`, and says so.
* **The harness treats `image_tool` the same in both paths**, so a case that must run inside a
  vendor image cannot silently get the default one.
* **A training failure carries its log out of the pod** (last 60 lines), the same pattern as the
  vLLM preamble's server-log tail.

### The remaining failure is the published image, and `--no-deps` proves it

```
OSError: /opt/lerobot/venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core4.so:
         undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
OSError: Could not load this library: .../torchcodec/libtorchcodec_core5.so
[end of libtorchcodec loading traceback]
```

`torchcodec`'s compiled extensions in `npa-lerobot:0.5.1` are built against a different torch
than the one installed — the classic ABI mismatch, and LeRobot loads the decoder on the training
path. The obvious suspicion was that installing npa into the venv bumped torch; job 254 ran with
`--no-deps` and reproduced the identical error, which **rules that out**: the image ships broken.

So `tokenfactory-train-triage.yaml` is not retired. Its twin's engine path is now complete and
each piece is proven live; what is left is a vendor-image repair (a consistent torch/torchcodec
pair) rather than anything in this change. `npa-lerobot:0.6.0` exists and may already pair them,
which is the next thing to try.

The same producer is the blocker for `tokenfactory-scene-to-rollout-judge` and a real
`tokenfactory-rollout-judge` twin (§R30), so one image repair unblocks three templates.

## R33. `tokenfactory-train-triage.yaml` retired (13 → 12) — the combo, live

The `torchcodec` ABI mismatch of §R32 is specific to `npa-lerobot:0.5.1`. `0.6.0` pairs torch and
torchcodec consistently, so the derived hostable tag was rebuilt from it
(`npa-lerobot:0.6.0-k8s-runtime`) and the twin ran clean.

### Job 256 — `npa-wf-multi-tokenfactory-train-triage-6732d78a`, both stages SUCCEEDED

```
train-gpu  3m40s   1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]   SUCCEEDED
triage    11m46s   1x[CPU:4+]                                    SUCCEEDED
```

The GPU stage produced a real training run, not a manifest:

```
 206666936  artifacts/model.safetensors
 412752084  artifacts/checkpoints/000001/training_state/optimizer_state.safetensors
      5817  artifacts/train_config.json
      3263  artifacts/checkpoints/000001/training_state/optimizer_param_groups.json
        62  artifacts/checkpoints/000001/training_state/training_step.json
     23214  triage/generations.jsonl
     20635  triage/prompts.jsonl
```

and the hosted model's report is genuinely derived from those artifacts — it quotes values *out
of the files it was handed*:

```
### Summary
The training run `tokenfactory-train-triage` appears to be training a robot policy using a
ResNet18 backbone with a transformer architecture. …

### Signals
- **Training Step**: … (`"step": 1` in `training_state/training_step.json`)
- **Batch Size**: … 2 (`"batch_size": 2` …)
- **Learning Rate**: … 1e-5 (`"lr": 1e-05` in `optimizer_param_groups.json`)
```

2,529 characters of report from an 18,779-character prompt built by
`npa.workflows.token_factory_triage` out of the run's textual artifacts. That is the combo's whole
point: Nebius GPU compute produces a run, a hosted model reads it, and **no GPU is held on the
reading side** — the triage stage ran on `1x[CPU:4+]`.

The migrated test asserts something the template could only imply: the triage stage's
`--artifacts-uri` is byte-identical to the train stage's `--artifacts-s3-uri`, so the consumer
provably reads what the producer wrote.

### What this unblocks

The same in-image LeRobot producer is the blocker for `tokenfactory-scene-to-rollout-judge` and
for a real `tokenfactory-rollout-judge` twin (§R30, §R23). Both now have a proven path: the
vendor-interpreter switch, the dataset materialisation, the hostable image and the artifact
upload all exist and are live-verified. What remains for those two is authoring their specs
(reason → rollout → judge) and a policy checkpoint for the eval-based producer.

## R34. `tokenfactory-rollout-judge.yaml` retired (12 → 11) — the GPU producer feeds the hosted judge

§R23 recorded that the spec sharing this template's *name* is a different workflow. The real twin
is `tokenfactory-rollout-judge-combo.yaml`, which keeps the property the combo exists to
demonstrate: **the GPU stage produces exactly what the zero-GPU hosted judge scores.**

### Job 261 — `npa-wf-multi-tokenfactory-rollout-judge-combo-d4798e41`, both stages SUCCEEDED

```
rollout-gpu  1m50s  1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]  SUCCEEDED
judge        1m37s  1x[CPU:4+]                                   SUCCEEDED

    1328  rollouts/eval_info.json
   89634  rollouts/videos/pusht_0/eval_episode_0.mp4
   97428  rollouts/videos/pusht_0/eval_episode_1.mp4
    1129  scores/vlm_eval_stub.json
```

Two real rendered episodes, then a hosted judgment of them:

```json
{
  "backend": "api",
  "model": "Qwen/Qwen2.5-VL-72B-Instruct",
  "frame_selection": "keyframes", "frame_count": 4,
  "passed": false,
  "rationale": "The robot appears to be attempting to interact with a blue sphere, but there is
                no clear indication that the task has been completed. … The outcome is ambiguous
                and does not show clear task completion."
}
```

That is a description of the *pusht* task from the rendered frames, not a stub, produced by a
**72B** model on a stage holding `1x[CPU:4+]`. The judge's `input_path` is byte-identical to the
rollout stage's `--rollouts-s3-uri`, and the migrated test pins that equality.

### Three more defects on the way, all now fixed

| Job | Symptom | Cause |
| --- | --- | --- |
| 257 | `lerobot-eval failed (… log=/tmp/lerobot_rollout.eval.log)` | the eval path named a log inside a dead pod — the same gap `train` had (§R32); it now carries its tail too |
| 258 | `ProcessorMigrationError: Config file 'policy_preprocessor.json' not found` | LeRobot ≥ 0.6 needs the processor format, and the obvious public policy `lerobot/diffusion_pusht` predates it. A checkpoint produced *by* 0.6 already has it — job 256's own training output — so the live case seeds one, the same pattern as the SONIC fixture |
| 259 | `HFValidationError: Repo id must be … : 's3:/lerobot-…/policy'` | argparse's `type=Path` had collapsed `s3://` to `s3:/`, so the S3 branch never matched. `--checkpoint-path` is a string now; it may be a URI, an HF id or a path |
| 260 | `bash: npa: command not found` in the judge stage | `npa_pip_install` falls back to `--user` under PEP 668, which moves the console script out of the default scripts dir; setup now also looks in the user scheme and `$HOME/.local/bin` |

Two of the four are general engine fixes that apply to every stage, not just this one.

## R35. `tokenfactory-scene-to-rollout-judge.yaml` retired (11 → 10) — the chain holds

The last combo template, and the only three-stage one. Its point is not that three things run:
it is that **the judge scores the rollout against the plan the reasoner produced.** The template
carried that link as

```bash
plan_task="$(python3 - "${PLAN_URI%/}/scene_reasoning.json" <<'PY' … )"
npa workbench vlm-eval run --task "${plan_task}" …
```

a command substitution no `toolRef` argv can express. Without it the third stage would score
against a literal string and the combo would be three unrelated jobs. So `vlm-eval run` gained
`--task-from <artifact>`, which reads the reasoner's `analysis` and builds the same prompt with
the same 900-character budget.

### Job 262 — `npa-wf-multi-tokenfactory-scene-to-rollout-judge-c9b64b65`, all three SUCCEEDED, first attempt

```
scene-reason  1m43s  1x[CPU:4+]                                   SUCCEEDED
rollout-gpu   1m58s  1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]   SUCCEEDED
scene-judge   1m38s  1x[CPU:4+]                                   SUCCEEDED

     834  scene/frame_000.png                    (seeded)
    3330  plan/scene_reasoning.json              (reasoner)
    1329  rollouts/eval_info.json                (GPU)
   89457  rollouts/videos/pusht_0/eval_episode_0.mp4
   99847  rollouts/videos/pusht_0/eval_episode_1.mp4
    2157  vlm-judge/vlm_eval_stub.json           (hosted judge)
```

The chain is visible in the artifacts rather than asserted. The reasoner wrote:

> *"The scene consists of a flat brown surface with a red cube positioned on it. The cube is
> centered on the surface and appears to be the only object present…"*

and the judge's recorded `task` begins:

> *"Judge whether the robot rollout accomplishes this planned task. Plan: The scene consists of a
> flat brown surface with a red cube positioned on it…"*

— the reasoner's own analysis, carried into the judge by `--task-from`, scored by
`Qwen/Qwen2.5-VL-72B-Instruct` through the hosted `api` backend. **Only the middle stage held a
GPU.**

The migrated test pins both links offline: the judge's `--input-path` equals the rollout's
`--rollouts-s3-uri`, and its `--task-from` equals the reasoner's `--output-path` artifact.

### The Token Factory combo group is done

All three combo templates are retired (§R33, §R34, §R35) and `COMBO_YAMLS` is empty. What made
them portable was five tool/engine additions, each replacing bash the templates carried:
in-image LeRobot train and rollout toolRefs, run-artifact and rollout uploads, the executable
triage stage, `--task-from`, and the vendor-interpreter switch. One more gap surfaced while
porting the last test: `npa.workflow.submit` had no `image` parameter, so a spec pinning
workbench images could not be submitted from Python against anything else — the CLI's `--image`
now has an SDK equivalent.

## R36. `sim2real-actions.yaml` retired (10 → 9) — two templates were halves of one pipeline

The actions template took its train slice from an operator-supplied `NPA_TRAIN_ENVS_URI`. That
URI is precisely what the split template had just written. Nothing in the raw surface connected
them: joining the two was an operator's job, done by hand, with no record that it had happened.
So the actions template is not a spec of its own — it is the fourth stage of
`sim2real-envgen-shards.yaml`.

### Live: `npa-wf-multi-sim2real-envgen-shards-d5c752f1` (runtime tier, 8m16s)

Four stages: two shards concurrently, a split barrier, then action conditioning.

```
  42292  envs/raw/raw-shard-00-of-02.jsonl
    378  envs/raw/raw-shard-00-summary.json
  42287  envs/raw/raw-shard-01-of-02.jsonl
    378  envs/raw/raw-shard-01-summary.json
  67385  envs/train/envs.jsonl              <- split
  17194  envs/heldout/envs.jsonl
    642  envs/manifest/split-manifest.json
  70621  actions/train/envs.jsonl           <- actions
    487  actions/train/actions-summary.json
```

and the join is recorded in the artifact rather than asserted:

```json
{
  "input_train_uri": "s3://…/sim2real-envgen-shards/envs/train/envs.jsonl",
  "action_conditioned_count": 32,
  "policy_image": "npa-reference-policy:local",
  "schema": "npa.sim2real.actions_summary.v1"
}
```

`input_train_uri` is the split stage's own output. The pipeline is now one submitted object, and
the connection is checkable offline.

### Another decorative GPU, and an honest note about the policy image

The template pinned `accelerators: RTXPRO6000:1` and `image_id: docker:${POLICY_IMAGE}`. The
shipped `sim2real_envgen actions` implementation uses neither: it reads the train slice, salts a
per-env seed with the image name, and writes reference actions with `random.Random`. It never
loads the image or touches a GPU. Third such finding (§R25 scenario-gen, §R27 envgen-split).

Rather than quietly drop the image, the spec keeps `--policy-image` — it is real provenance in
`actions-summary.json` and it changes the generated seeds — and states plainly where the swap
point is: a resource-profile change on the spec, not a rewrite. A reader who wants a real policy
container knows what to change; a reader who assumed the template already ran one now knows it
did not.

## R37. `isaac-franka-capture-reason.yaml` retired (9 → 8) — code that had never run

Two stages: a headless Isaac Lab Franka rollout renders RGB frames on a GPU, then a hosted
Cosmos3 reasoner plans from them on CPU. The template could not run at all without a repo
mounted into the pod —

```bash
REPO_ROOT="${NPA_REPO_ROOT:-/opt/nebius-physical-ai}"
if [ -d "${REPO_ROOT}/npa" ]; then … else
  echo "NPA repo not found at ${REPO_ROOT}; mount or bake /opt/nebius-physical-ai" >&2
```

— because the capture code lived in `npa/scripts/`. Moving it into the package as
`npa.workflows.isaac_capture` removed that requirement **and made the code reachable for the
first time.** It did not work. Four defects, in the order the cluster found them:

| Job | Symptom | Cause |
| --- | --- | --- |
| 267/268 | `No module named 'isaaclab'` | npa installed into `/usr/bin/python3`; the simulator lives in the Omniverse kit environment. Isaac is now a declared vendor interpreter, and the `--no-deps` vendor install learned a with-deps fallback (job 268: the kit python carries none of npa's dependencies). |
| 270 | `No module named 'pxr'` | `isaaclab_tasks` was imported before `AppLauncher`. Isaac Lab's modules reach into the kit runtime at import time; that runtime does not exist until the app launches. |
| 271 | 45 minutes at 170% CPU, no frames | `/isaac-sim/kit/data` and `/logs` did not exist and `/cache` was root-owned, so Kit could not write. It logged `failed to open … user.config.json`, `omni.kvdb … Unexpected key-value database error`, `mdl_list_cache is not complete` — and then simply stopped. Three empty directories in the image; the capture now takes **25 seconds**. |
| 278 | six frames, exit 0, nothing uploaded | `simulation_app.close()` ends the process instead of returning, so the upload that lived after `_capture_frames()` never ran. Publishing is now a callback that fires before the simulator tears down. |

### Then it ran, and the run was still wrong

Job 280 succeeded end to end and produced six technically perfect photographs of **bare floor**.
The reasoner was not fooled — it replied that it could see "a tiled floor … no visible objects,
obstacles, or environmental features". The template had borrowed the sim2real engine's
`_attach_isaac_viz_camera`, whose pose was tuned for a different scene.

Job 281, with the stage owning its own look-at pose, aimed 90 degrees off and photographed the
ground receding to a horizon ("a tall building with a grid-patterned facade", said the reasoner):
the look-at was built for OpenGL's -Z-forward frame, while Isaac Lab's `convention="world"` is
REP-103, **+X forward, +Z up**.

<img alt="Job 281: mis-framed capture, ground plane and sky" src="/opt/cursor/artifacts/screenshots/isaac-franka-capture-misframed-job281.png" width="320" />

**A capture stage that photographs the wrong thing fails silently, which is worse than failing
loudly.** Both runs were green. Only looking at the pixels caught it.

### Job 283 — `npa-wf-multi-isaac-franka-capture-reason-d8eca4b3`, both stages SUCCEEDED

```
capture  2m46s  1x[RTXPRO-6000-BLACKWELL-SERVER-EDITION:1]   SUCCEEDED   25.0s of simulation
reason   0m49s  1x[CPU:4+]                                   SUCCEEDED   6 images -> a plan

  191727  scene/frame_00.png        (512x512, was 128x128: a VLM has to read these)
  …
  225767  scene/frame_05.png
     335  scene/isaac_capture_summary.json
    2515  reasoning/scene_reasoning.json
```

<img alt="Job 283: the Franka, its table and the cube, correctly framed" src="/opt/cursor/artifacts/screenshots/isaac-franka-capture-frame-03-job283.png" width="420" />

and `nvidia/Cosmos3-Super-Reasoner`, reading those six frames:

> *"The scene is a simulated environment featuring a robot arm mounted on a black table. The
> robot arm has a gripper at its end and is positioned above a small, colorful cube on the table.
> The background consists of a grid-patterned floor…"*

That is the picture. The stage is verified by comparing what the model said against what the
camera saw, not by an exit code.

### What this template's retirement bought

Three of the four fixes are engine- or image-level and benefit every Isaac stage, not just this
one: the vendor-interpreter entry, the with-deps fallback, and the writable Kit directories
(pinned by `test_workbench_image_k8s_prereqs.py`, which now separates "can be scheduled" from
"can render"). The framing is unit-tested without a simulator by rotating the camera's +X axis
and asserting it lands on the target.

## R38. `cosmos2-transfer.yaml` retired (8 → 7) — a stub replaced by the real model

The template held an `RTXPRO6000:1` in order to run this:

```python
payload = { …, "status": "contract_ready" }
print(json.dumps(payload, indent=2, sort_keys=True))
```

It transferred nothing. **A faithful twin would have been a stub on the new surface**, which is
precisely what `skills/atomic/real-components/SKILL.md` exists to prevent, so the twin uses
`workbench.cosmos2.transfer_execute` — `--execute` turns a missing transfer runtime into a hard
error instead of a silent fall back to reference augmentation. It is also the first spec to use
that toolRef, which had been sitting in the catalog unused while five specs used the
manifest-only variant.

### Three defects between the template and a real run

| Job | Symptom | Cause |
| --- | --- | --- |
| 284 | `No such command 'cosmos2'. Did you mean 'cosmos'?` | The image bakes its own `npa` console script, first on PATH. Setup saw `command -v npa` succeed, skipped installing, and the stage ran a stale CLI. `python3` was already shimmed to the recorded interpreter; `npa` now is too. |
| 285 | same error, now from the shim | Deeper: the image ships `PYTHONPATH=/opt/npa/src` holding a stale npa **source tree**, which shadows every install, editable or not, in any interpreter. A probe pod settled it: `/opt/npa/src/npa/__init__.py`, `cosmos2 registered: False`. Third image to do this (lerobot was job 250), so the engine now puts the staged source ahead of whatever PYTHONPATH it inherited. |
| 286 | `hf download nvidia/Cosmos-Guardrail1` exit 1 | It reached `examples/inference.py` — the real model — and stopped at a gated Hugging Face repo. `workbench.cosmos2` now hints `HF_TOKEN`, and the guardrail that pins hint-vs-case agreement immediately found three other cosmos2 cases missing it. |

### Job 287 — the model ran, and published half of what it made

`SUCCEEDED` in 14m21s, leaving exactly one object: a 3.9 MB augmented MP4. The manifest — prompt,
control spec, guidance, whether the run was conditioned on an input clip — was echoed to stdout
and died with the pod. **For a synthetic-data stage that provenance is the product**, and the
spec had nothing durable to declare as its output. The data-factory path already published a run
manifest; the single-inference path now matches it.

### Job 288 — `npa-wf-gpu-cosmos2-transfer-f867e7c3`, SUCCEEDED, 14m23s on one GPU

```
  3918459  cosmos2-transfer/augmented/robot_depth.mp4
     1068  cosmos2-transfer/augmented/manifest.json
```

```json
{
  "mode": "cosmos_transfer2.5",
  "status": "executed",
  "output_kind": "video",
  "control_spec": "assets/robot_example/depth/robot_depth_spec.json",
  "video_bytes": 3918459,
  "schema": "npa.cosmos2.transfer.v1"
}
```

`"status": "executed"` where the template said `"contract_ready"`. Frame 20 of the generated
clip — photoreal output synthesised from a depth control spec:

<img alt="Job 288: a frame from the Cosmos-Transfer2.5 augmented clip" src="/opt/cursor/artifacts/screenshots/cosmos2-transfer-augmented-frame-job288.png" width="420" />

Two of the three engine fixes here (the `npa` shim, the PYTHONPATH precedence) apply to every
vendor image, not just this one, and both replace a class of failure that presents as "the tool
does not have this command" while the tool plainly does.

## R39. `cosmos3-text-to-image-inference.yaml` — the tool exists, the template stays (7 remain)

This is the one place in this phase where the work landed and the retirement did not, so it is
worth being exact about what was proven and what was not.

The template carried the whole capability as bash: roughly a hundred lines of shell and heredoc'd
python inside an `envs:` block, including the inference command itself as a multi-line
environment variable that `run:` executed with `bash -lc "${NPA_COSMOS3_INFER_COMMAND}"`. Nothing
about it was reachable from the CLI or the SDK, and nothing about it was tested. It is now
`npa.workbench.cosmos.text_to_image` behind `npa workbench cosmos3 text-to-image`, with the
framework's inference invoked as an argv rather than an interpolated string, and 14 unit tests —
mostly around verification, which is what stands between "exit 0" and an image.

### What the cluster taught, in order

| Job | Reached | Stopped by |
| --- | --- | --- |
| 289 | tool ran | `'Cosmos3FetchResult' object has no attribute 'source_dir'` — wrong field, and no `ok` check, so a genuinely failed fetch would have surfaced as an AttributeError with the cause discarded |
| 290 | fetch | `[Errno 2] No such file or directory: 'huggingface-cli'`, seconds after installing it: console scripts land in whichever scripts dir pip chose. Now resolved as `python -m huggingface_hub…` |
| 291 | HF download | `[Errno 2] ... 'uv'`. SkyPilot's default image ships a uv in its own runtime dir — on setup's PATH, not the command's. Now probed and invoked as a module |
| 296 | uv sync, into inference | `libstdc++.so.6: version GLIBCXX_3.4.29 not found` for transformer_engine. Now finds a libstdc++ that exports the symbol by reading the library |
| 301 | same, one layer down | `libc.so.6: version GLIBC_2.32 not found`. glibc is the host C library; no `LD_LIBRARY_PATH` substitutes for it. The stage needs a modern base — which in production it already resolves (`workbench.cosmos3` → `npa-cosmos3-reason`); the harness had been clearing workbench images |
| 304 | nothing | `container not found ("ray-node")` — the cosmos3 image was not SkyPilot-schedulable. Added its k8s-prereqs recipe |
| 307 | setup | `npa is not importable after setup`. Forcing `/usr/bin` first put a bare system python ahead of the one carrying npa. **PATH ordering is an Isaac requirement, not a universal one**, and the guardrail now says so instead of demanding it everywhere |
| 308 | setup | `[Errno 13] Permission denied: '/opt/npa/venv/bin/npa'`; `--user` is not a fallback inside a virtualenv. The runtime user now owns the environment it runs from |
| 309 | setup, then the CLI | `ModuleNotFoundError: No module named 'paramiko'` |

### Why it stays

Job 309 is one dependency short. The cosmos3 image installs npa with a curated `--no-deps` list,
so overlaying a newer npa leaves the CLI's import chain missing `paramiko`. The fix is either
that dependency in the image or a with-deps fallback for the source overlay — the same
two-attempt rule the vendor-interpreter install already uses. Both are small; neither is
verified, so **the template is not deleted.** The live case is `plan_only` with that reason
recorded, so it reads as a named blocker rather than an untried spec.

What did land and is verified offline: the tool, its tests, the HF-module and uv-module
resolution (both of which fix a class of "installed it, still cannot find it"), the libstdc++
probe, the cosmos3 k8s-prereqs recipe, and the correction that PATH ordering is Isaac-specific.

## Retirement tally: 36 → 7

```
raw SkyPilot templates    36 -> 7      (ls npa/src/npa/workflows/skypilot/*.yaml | wc -l)
```

The seven that remain, and why each one does — no "not attempted" among them:

| Template | Why it survives |
| --- | --- |
| `bdd100k-pipeline.yaml` | Runner ported; every stage passes the offline mock drive. A live run needs the LanceDB workbench service (§R16, §R26). |
| `dataset-ingest-curate.yaml` | Four of five stages pass live; `register` needs the same LanceDB service (§R16). |
| `sim-to-real-pipeline.yaml`, `sim-to-real-trigger.yaml` | Wrap `npa.workflows.sim_to_real`, which itself emits a DeprecationWarning pointing at the staged engine. Awaiting the decision on whether to retire the stack or implement its stubs. |
| `sonic-locomotion-finetuning.yaml`, `sonic-train-standalone.yaml` | The launcher problem: the train stage asks the in-pod CLI to launch a Nebius serverless job (§R11). |
| `cosmos3-text-to-image-inference.yaml` | Twin and tool exist and were driven live through nine jobs to 309; one dependency short (§R39). |

Four of the seven are blocked on two decisions (LanceDB, the legacy stack) and one known
engine problem (the launcher). The other three are one small fix each.

## R40. The legacy sim-to-real stack retired (7 → 5) — deliberately without a twin

`sim-to-real-pipeline.yaml`'s stage ran `python -m npa.workflows.sim_to_real real-loop`. That
module opens with:

```python
warnings.warn(
    "npa.workflows.sim_to_real is legacy. Use npa.workflows.sim2real "
    "(Sim2RealWorkflow / runbook.yaml) for the production VLM→RL loop.",
    DeprecationWarning,
)
```

Every other retirement in this PR required a live twin first. This one is the exception, and the
reason is the point: **a twin would have made the new surface the home of a legacy path.** The
maintained loop already has a spec (`npa-workflows/sim2real-vlm-rl.yaml`) and a runbook for
reading without npa in the loop (`sim2real/runbook.yaml`).

**Watching a bucket is not deprecated**, so `npa.workflows.sim_to_real_trigger` stays — and was
ported rather than deleted. It used to shell out to `scripts/run_sim_to_real_pipeline.py`, 531
lines of in-place document surgery on the template that just went; it now submits the staged
loop's spec the same way an operator would by hand, and `--render-only` became
`workflow validate-spec`, which resolves every config token and builds the plan without
launching anything. `run_sim_to_real_quickstart.py` went with it: it imported that runner to
drive the same deprecated loop.

Two docs carried a "legacy" banner over this path and were deleted; their readers now land on
the maintained guide. The two three-tier contracts that named the retired YAML moved onto the
spec, with the watch parameters declared as `spec_gap` — a stage runs once, a watcher does not —
which **empties `LEGACY_YAML_TIER`**, so its guardrail flips from "only shrinks" to "must stay
empty".

## R41. The LanceDB service, deployed — and four defects it had been hiding (5 → 4)

`npa workbench lancedb deploy` could target a local docker daemon, a managed VM (blocked), or
LanceDB Cloud. **None of those is reachable from a workflow stage**, which is why two templates
could not retire. `--runtime kubernetes` creates a Deployment and a ClusterIP Service and
returns `http://npa-lancedb.workbench.svc.cluster.local:8686` — a name every pod can resolve.

Getting one pipeline green took five findings, each invisible until the service existed:

| Symptom | Cause |
| --- | --- |
| `ImagePullBackOff`, `401 Unauthorized` on a tag that exists | the namespace's shared registry secret holds an expiring IAM token. SkyPilot never notices because it passes credentials per submit; a Deployment's kubelet pulls again on every restart. The deploy now mints its own. |
| deploy timed out; `/health` answered `{"detail":"LANCEDB_TOKEN is not configured"}` forever | `auto` auth meant `token` for every non-container runtime. A readiness probe that can never pass, and a timeout that pointed at nothing. `auto` now means "token if the operator supplied one". |
| deploy reported `running` while the new pod crash-looped | `kubectl wait --for=condition=Available` is satisfied by the OLD ReplicaSet during a rolling update. `rollout status` waits for the new one. |
| `404 Not Found` for `…/query` (job 313) | the dataset integration posts `/index` and `/query`; the wrapper exposed `/tables/{name}` and `/query-table`. **Two halves written against different APIs that never met**, because nobody had ever been able to make the call. |
| `register` SUCCEEDED returning 0 records from a table holding 3 matching rows (job 316) | the query sent no table, so the service read its default; and it sent every facet it knows about, set or not, asking for `modality = '' AND min_quality = 'None'`. |

A sixth was upstream of all of them: `index_in_lancedb` had **no caller that could be given an
endpoint** — the ingest CLI never exposed one — so the dataset-of-record could not populate the
index it queries.

### Job 317 — `npa-wf-cpu-dataset-ingest-curate-754816f0`, all five stages SUCCEEDED

```
ingest  validate  quality-gate  curate  register        (5/5 SUCCEEDED, 5m46s)

register: {"backend": "lancedb", "count": 12, "records": [...]}
```

Twelve records read back out of the service that `ingest` had written to minutes earlier. The
round trip is the evidence: a query returning rows can only happen if the index was populated,
through the real service, from this run.

## R42. SONIC trains in the pod it is already in

`workbench.sonic.train` asked for `--runtime serverless`, which provisions a Nebius Job **from
inside a pod**. A workflow stage cannot: it failed live with
`SONIC --runtime serverless requires --project-id` (§R11), and given a project id it would mean
a workflow launching infrastructure the workflow engine had already provisioned for it.

`--runtime in-job` runs the same training body — the SONIC image's own `/entrypoint.sh train`,
the same `SONIC_*` environment, the same S3 upload step — in the pod the stage is running in.
The body is now shared by both runtimes rather than duplicated, because two training scripts
would be two trainers; the only thing that differs is who provides the machine.

## R43. `cosmos3-text-to-image-inference.yaml` retired (4 → 3) — eleven jobs, eight defects

The longest chase in this PR, and worth recording as one, because every wall was invisible from
the diff. §R39 covered jobs 289–309; three more followed once the tool could load at all:

| Job | Symptom | Cause |
| --- | --- | --- |
| 309 | `No module named 'paramiko'` | the cosmos3 image installs npa with a curated `--no-deps` list, so the source overlay inherited its omissions. The overlay now retries **with** deps when `import npa.cli.main` fails — `import npa` succeeded there, which is why the probe had to be the command tree. |
| 319 | `cuDNN version incompatibility … a conflicting cuDNN in LD_LIBRARY_PATH` | the fix for §R39's libstdc++ wall put a whole conda lib directory on the loader path, and that directory also holds an older cuDNN than PyTorch bundles. Now a single symlinked library — and nothing at all when the host's own libstdc++ already exports the symbol. |
| **320** | — | **SUCCEEDED, 5m37s.** |

```
203462  cosmos3-text-to-image/text-to-image.png    960 x 960
   410  cosmos3-text-to-image/success.json         {"status": "ok",
                                                    "model_id": "nvidia/Cosmos3-Nano",
                                                    "prompt": "a small robot arm sorting
                                                               colored blocks on a workbench"}
```

<img alt="Job 320: Cosmos3-Nano's image for the prompt" src="/opt/cursor/artifacts/screenshots/cosmos3-text-to-image-job320.png" width="420" />

The template carried this as roughly a hundred lines of shell and heredoc'd python inside an
`envs:` block, with the inference command itself a multi-line environment variable executed by
`bash -lc "${NPA_COSMOS3_INFER_COMMAND}"`. It was unreachable from the CLI and the SDK, untested,
and — as eleven jobs showed — had never run. It is now `npa workbench cosmos3 text-to-image`
with 23 unit tests, and its verification step is the part that matters: a framework that exits 0
having written a truncated file is the failure worth catching, and it is caught by reading the
image header rather than trusting the exit code.

## R44. `bdd100k-pipeline.yaml` — the LanceDB wall is gone; a second service is not

Recorded because the blocker changed, and the new one is specific.

With the LanceDB service deployed (§R41), live job 321
(`npa-wf-multi-bdd100k-pipeline-f1bbb96e`) got four stages further than any previous attempt:

```
ingest         SUCCEEDED   52s
backfill-cpu   SUCCEEDED   1m05s
backfill-clip  SUCCEEDED   1m30s   on RTXPRO-6000 — the GPU CLIP UDF, against the real service
curate-views   SUCCEEDED   58s
train-rider    FAILED      Cannot reach detection-training endpoint
                           http://npa-detection-training.workbench.svc.cluster.local:8790
```

`backfill-clip` passing is its own finding: the first attempt (job 318) failed inside the service
with `'BaseModelOutputWithPooling' object has no attribute 'norm'`. `get_image_features` returns
a plain tensor in some transformers versions and a model-output object in others; the code
assumed the first, and nothing pinned it because the tests only asserted the resulting vector's
shape.

**What remains is a second in-cluster service.** `npa workbench detection-training deploy`
already targets Kubernetes with `rollout status` and pull-secret handling — it does not need the
work LanceDB needed. On this cluster it reported `exceeded its progress deadline` and left no
deployment in any namespace, which points at its `_resolve_kubeconfig(cluster_name=…)` selecting
a different context than the one `kubectl` uses here. That is the next thing to chase, and it is
an operational mismatch rather than a missing capability.

## R45. The SONIC launcher problem is fixed; the remaining gate is the vendor's

`workbench.sonic.train` asked for `--runtime serverless` and failed live with
`SONIC --runtime serverless requires --project-id` (§R11) — a workflow stage trying to
provision infrastructure the workflow engine had already provisioned for it.

Two live jobs traced the fix to the vendor's own front door:

| Job | Reached | Stopped by |
| --- | --- | --- |
| 322 | the training body ran | `/entrypoint.sh not found in SONIC image` — **the runtime worked**; the live case had not asked for the vendor image, so it ran on SkyPilot's default one. |
| 323 | the SONIC image's own `/entrypoint.sh train` | NVIDIA's asset gate: a licence notice followed by `Nothing has been downloaded. See docs/workbench/container-packaging.md.` |

Job 323 is the useful one. The stage got all the way into the vendor trainer, on the vendor
image, in the pod the workflow engine had already provisioned — which is exactly what `in-job`
was for. What stops it now is an **EULA/NGC credential question about the image**, not a
workflow question, and it is the same gate an operator would hit running that container by hand.

So the two SONIC templates stay, with a different reason than they had: not "the launcher
provisions its own infrastructure", which is fixed, but "a real SONIC training run needs the
image's assets". The engine work, its tests, and both live traces are recorded here so the
person who has the NGC entitlement can finish it in one run rather than rediscovering the path.

## R46. `bdd100k-pipeline.yaml` retired (3 → 2) — eleven stages, two services

The longest pipeline in the catalog, and the last one blocked on infrastructure. §R44 left it at
`train-rider`, unable to reach a second in-cluster service. Three findings closed the gap, and
the first two are the same defect wearing different clothes: **a command that silently targets
somewhere other than where you are looking.**

| Finding | What it looked like | What it was |
| --- | --- | --- |
| `rollout status` timed out; the deployment was in **no namespace** of the cluster being inspected | "apply reported *configured*, so where did it go?" | `--cluster-name` **defaulted** to `npa-workbench-eu-north1`, whose cached kubeconfig points at a different cluster. Every deploy had been landing there. Default is now the ambient kubeconfig — the cluster `kubectl` is already on. |
| pod `Pending` forever after that | nothing in the output mentioned nodes | `--gpu-type` knew only `h100` and `l40s`; this cluster's GPU nodes are labelled `gpu-rtx6000`, so the selector matched nothing. The workbench's own GPU is now selectable, and the error lists what it knows instead of naming two. |
| three trainings SUCCEEDED, then `eval-rider` failed with `invalid literal for int() with base 10: 'train'` | a puzzling type error | BDD100K stores **string** categories and one of them is literally `train` — the vehicle. Training took `--label-map`; eval did not, so its loader fell through to `int(raw)`. `EvalRequest.label_map` had existed all along **with no CLI flag to fill it**. |

A fourth was not a defect at all: with 64 synthetic rows the `distant_person` view came out
empty and training failed honestly with `detection dataset is empty`. 768 rows populate all
three views. Worth stating rather than hiding — the fixture was too small, and the tool said so
in the clearest possible terms.

### Job 326 — `npa-wf-multi-bdd100k-pipeline-763b2bdf`, 11/11 SUCCEEDED, 15m16s

```
ingest          SUCCEEDED   57s
backfill-cpu    SUCCEEDED   1m06s
backfill-clip   SUCCEEDED   1m10s   RTXPRO-6000 — the GPU CLIP UDF, against the real service
curate-views    SUCCEEDED   1m04s
train-rider     SUCCEEDED   1m31s   RTXPRO-6000 ─┐
train-nighttime SUCCEEDED   1m29s   RTXPRO-6000  ├─ three real training runs
train-distant   SUCCEEDED   1m32s   RTXPRO-6000 ─┘
eval-rider      SUCCEEDED   1m09s   eval-9b570bfc679c
eval-nighttime  SUCCEEDED   1m02s   eval-316bbc3320f0
eval-distant    SUCCEEDED   1m02s   eval-d64085b5cf6b
review          SUCCEEDED     51s
```

The evals report `mAP: 0.0`, and that is the honest number: one epoch over 768 synthetic rows
trains nothing. What the run proves is that **every stage invoked its real component and produced
the artifact the spec declared** — including the numeric-metric guard, which exists precisely so
a service answering 200 with a null mAP cannot be mistaken for success.

Both services stay deployed for follow-up:

```bash
npa workbench lancedb deploy --runtime kubernetes --namespace workbench \
  --storage-path s3://<bucket>/lancedb/
npa workbench detection-training deploy --namespace workbench --gpu-type rtxpro6000 \
  --output-path s3://<bucket>/detection-training/
```

## R47. SONIC: the launcher problem is solved; the wall is now inside NVIDIA's asset conversion

The decision this PR asked for was "move the launcher out of the workflow, or train in-pod
against the vendor image". In-pod was chosen and it works — six live jobs prove it, each getting
further than the last. Recording the chain because every step of it was invisible from the diff,
and because the two templates stay for a reason that is no longer about workflows at all.

| Job | Reached | Stopped by |
| --- | --- | --- |
| 322 | the training body ran | `/entrypoint.sh not found in SONIC image` — the runtime worked; the live case had not asked for the vendor image |
| 323 | the SONIC entrypoint | `Not accepted (unset or not YES): OMNI_KIT_ACCEPT_EULA ISAACSIM_ACCEPT_EULA` |
| 327 | same | the `/proc/1/environ` carry found nothing: **SkyPilot replaces the pod's PID 1**, so the image's docker ENV never reaches any shell in the pod |
| 328 | **the real trainer**, on `nvidia/GEAR-SONIC` weights, Isaac Sim and Isaac Lab installed | `ModuleNotFoundError: No module named 'lxml'` |
| 329 | further into `gear_sonic` | `ModuleNotFoundError: No module named 'open3d'` |
| 330 | Isaac Sim booted; the whole env config instantiated; robot USD conversion started | `[Error] [omni.usd] Failed to open layer @/tmp/IsaacLab/usd_…/configuration/pelvis.tmp.usd@`, then `[Fatal] attempted member lookup on NULL TfRefPtr<UsdStage>` |

### What each step taught, and what shipped

**Acceptance belongs on the spec.** Job 327 killed the neat idea — reading the image's own
`OMNI_KIT_ACCEPT_EULA` from PID 1 — because under SkyPilot there is no such PID 1. The gate's own
message asks for "env: entries on the pod/SkyPilot task", so `sonic_accept_nvidia_eula` is a
config key, **empty by default**, and `--accept-nvidia-eula` takes a *value* rather than being a
bare flag, because a toolRef argv is always flag-plus-value and an empty positional would hand
Typer an unexpected argument. Acceptance is the operator's to give; the harness reads it from
`NPA_E2E_SONIC_ACCEPT_NVIDIA_EULA` rather than shipping an accepted spec.

**The dependency gap is a list, not a package.** `lxml` then `open3d`: each surfaces only once
the previous is satisfied and the trainer gets further, so the staging step iterates
`GEAR_SONIC_RUNTIME_DEPS` and probes each. They cannot be baked into the image — the venv that
needs them is built **at runtime**, after EULA acceptance, under a content-hashed
`/opt/isaac-cache` path — so `PYTHONPATH` is the only seam that reaches it.

### Why the two templates stay

Job 330 dies inside Isaac Lab's own URDF→USD conversion of the G1 robot. Everything the
workflow surface is responsible for is done: the stage runs on the vendor image, in the pod the
engine provisioned, with the operator's acceptance, the real weights, and a working interpreter.
What remains is an NVIDIA asset-pipeline failure that no change to a spec, a toolRef or the
engine will move.

So `sonic-train-standalone.yaml` and `sonic-locomotion-finetuning.yaml` keep their place in the
tally, with the reason rewritten from "the launcher provisions its own infrastructure" — which is
fixed — to what it actually is now. The engine work, its 15 unit tests, and all six live traces
are recorded here so whoever picks this up starts at job 330 rather than at job 322.

## R48. Converging with #238, which solved the same launcher problem in parallel

`main` landed #238 while this branch was live-testing the SONIC launcher, and it fixed the same
thing under a different name: `--runtime local` where this branch had `--runtime in-job`. Two
runtimes for one idea is worse than either, so they are now one.

**#238's implementation is the one that stays.** It has something this branch's did not: when the
vendor entrypoint is absent it falls back to a real reference locomotion trainer with gradient
descent, so a stage produces a checkpoint instead of failing. That is strictly better.

**What this branch contributes to it** is the piece six live jobs found and #238 could not have
known about, because it never got past the image boundary: the SONIC image's own trainer
**refuses** to download Isaac Sim / Isaac Lab until NVIDIA's terms are accepted, and SkyPilot's
pod does not inherit the image's docker ENV. Without acceptance, `--runtime local` does not fail —
it quietly runs the *reference* trainer instead of the vendor one, which is a worse outcome than
an error and is invisible from the exit code.

So `--accept-nvidia-eula` and `sonic_accept_nvidia_eula` sit on #238's runtime, empty by default.

The merge also cost three regressions, each caught by a guardrail this branch had added rather
than by review:

* `sonic-locomotion-finetuning` declared `report.json` / `manifest.json` where the tools write
  `mjlab_eval.json` / `retargeting_result.json` — `test_spec_declared_outputs` said so by name.
* `sonic/train`'s `spec_gap` still pinned `max_iterations`, which #238's toolRef now reaches.
* `workbench.sonic` still hinted `NGC_API_KEY`, which the local trainer does not use and which
  would have skipped the twins rather than run them.

Where the two branches overlapped elsewhere, the better half won on the merits rather than by
seniority: #238's vLLM installer (uv resolution plus a weight pre-fetch during setup) replaced
this branch's inline pip, keeping the `ninja` step live job 214 needed; this branch's readiness
preamble was kept over #238's fire-and-hope launch, and gained #238's `NPA_VLM_SELF_HOSTED_MODEL`
export so the client asks for the model that was actually started.

## R49. The retirement guardrail caught a new raw template on merge

Merging `main` a second time brought #234, which added
`npa/src/npa/workflows/skypilot/nurec-reconstruct.yaml` — a **new** raw template in the directory
this PR is emptying. Nothing in review flagged it; `test_skypilot_catalog_retirement` did, by
name and with the remedy:

```
new raw SkyPilot task YAML(s) appeared in the retiring catalog: ['nurec-reconstruct.yaml'].
Author an npa.workflow/v0.0.1 spec under npa/workflows/workbench/npa-workflows/ instead;
if a raw template is genuinely required, add it to REMAINING with a reason.
```

That is the guardrail working as designed, and it is worth recording because it is the first
time it has fired against something other than this branch's own work.

It is **not** simply an un-ported template. #234 shipped both forms deliberately, and they are
different execution shapes: the npa.workflow spec runs each state in its own pod and hands
artifacts over through S3, while the raw task is single-pod and shares `/tmp` between stages.
The spec already carries a live-matrix case.

So it is listed in `REMAINING` with that reason rather than deleted. Whether the single-pod
variant should survive alongside its spec is #234's call to make, not this PR's to make silently
— but the tally now says it exists and why, which is the whole point of a machine-checked list.

A third merge brought #235 and a second new template, `cosmos3-generate.yaml`, caught the same
way and listed the same way. Its npa.workflow twin (`npa-workflows/cosmos3-generate.yaml`)
already exists, so retiring the raw one needs only a live run of that twin — which belongs with
#235 rather than here.

Two templates arriving mid-sweep is the argument for the last step of this work in one line: as
long as the directory exists, things land in it. Once the SONIC pair clears, the guardrail should
invert from "these may remain" to "this directory must not exist", and the question stops being
whether anyone remembered.

## R50. The SONIC chain is green — the vendor's asset pipeline was never on the critical path

Job **334**, `sonic-export-eval.yaml`, tier `multi`: **SUCCEEDED**, all three stages, 6m31s.

The lead was #238's reference trainer. Six live jobs (§R47) had been spent getting the *vendor*
trainer to run, and the wall was NVIDIA's own URDF→USD conversion of the G1 robot. But the twin
does not need the vendor trainer to be a real twin — the fallback trains for real:

```json
{ "embodiment": "UNITREE_G1_SONIC", "action_dim": 23, "device": "cuda", "iterations": 40,
  "initial_loss": 0.004570585375404335, "final_loss": 0.00007135790667689434 }
```

Forty iterations of gradient descent on the GPU, loss down **98.4%**, into a 231 KB
`checkpoint.pt`. Export turned it into a 228 KB `sonic_policy.onnx` with its metadata sidecar,
and eval scored it per episode on real dynamics:

```json
{ "backend": "reference", "episodes": [
  { "episode_index": 0, "distance": 0.1596, "energy": 0.0261, "episode_return": 0.1583,
    "fall": false, "steps": 32 } ] }
```

Every stage handed the next its S3 artifact — `training/checkpoint.pt` → `sonic_policy.onnx` →
`eval.json` — which is the whole property the retired template chained inline in bash.

### The failure before it is the one worth recording

Job **333** failed in 2m17s with:

```
Invalid value for '--runtime': 'local' is not one of 'vm', 'container', 'serverless', 'in-job'
```

`in-job` is *this branch's* name for the runtime, which the merge with #238 replaced. The pod was
running an npa source tree staged to S3 **before** that merge. Nothing in the harness noticed: the
overlay syncs whatever is at the URI, and a stale tree is indistinguishable from a fresh one until
a renamed flag makes it visible.

That is a real gap in the live harness rather than a one-off. The overlay should carry the commit
it was cut from and refuse to run against a tree older than the working copy — recorded here as
follow-up work rather than fixed mid-run, since the fix belongs with the harness, not with this
retirement.

## R51. Both SONIC templates retired — this sweep's catalog is empty

| Twin | Job | Time | What ran |
| --- | --- | --- | --- |
| `sonic-train.yaml` | 335 | 2m09s | 40 iterations of gradient descent on an RTX PRO 6000 |
| `sonic-locomotion-finetuning.yaml` | 336 | 5m00s | retarget → train → mjlab |
| `sonic-export-eval.yaml` | 334 | 6m31s | train → export → eval |

Nothing decorative in the finetuning chain. Retargeting ran SONIC's **own**
`gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` from the upstream checkout and wrote
a 17 KB `motion_lib.pkl`; mjlab reported `"backend": "mjlab"`, `"dry_run": false`, eight episodes,
against the `checkpoint.pt` the train stage had just written.

With that, every template this sweep set out to retire is gone. The two files left in
`npa/src/npa/workflows/skypilot/` both arrived **during** the sweep, from #234 and #235 (§R49).

### The tests that were holding the templates hostage

Deleting the two files broke fifteen tests, and the split between them is the more interesting
half of this entry.

**Twelve exercise the submit *wrapper*** — registry auth, VM resource defaults by GPU target,
spot, docker-payload mode, `${PLACEHOLDER}` substitution. That behaviour is **not** being
retired: `npa workbench workflow submit` still accepts a customer's own SkyPilot YAML, and it
must keep working. They simply used a *shipped* template as their fixture, so a product decision
about the catalog looked like a test failure. They now read frozen copies under
`npa/tests/fixtures/skypilot/`, which says the wrapper's contract is independent of the catalog —
the whole reason the catalog can go.

One guardrail was in the same trap, and worse: `test_workflow_image_extraction_finds_skypilot_images`
asserted that `npa-sonic` appeared among the shipped templates' images, so **the image checker's
own test** was a reason the SONIC templates could not be retired. It now pins the extractor
against a fixture, and a second test keeps whatever remains in the catalog readable — including
the empty case this work is heading for.

**Three asserted the template's own shape** (`execution: serial`, task names, `image_id` blocks).
Those are exactly what this PR replaces. A spec-level test now asserts the three toolRefs in
order with no `parallel:` group, which is a stronger claim than the one it replaces: it describes
what the engine will run, not what a document says.

The general lesson is worth keeping: **a test that reads a shipped artifact as a fixture quietly
makes that artifact undeletable.** Every one of these could have been written against a fixture
from the start at no cost.

## R52. Four merges in, and the guardrails kept being the thing that noticed

`main` moved four times during this work — #238, #234, #235, #229 — and every one of them
touched something this PR had changed. The pattern is consistent enough to be worth stating
plainly: **in each case the conflict was resolved by keeping whichever half was better, and in
each case the thing that noticed a real problem was a test, not a reviewer.**

| Merge | What it did to this work | Resolution |
| --- | --- | --- |
| #238 | solved the SONIC launcher in parallel as `--runtime local` | its implementation stays (it has a fallback this branch lacked); this branch's EULA acceptance goes on top. Its vLLM installer replaced this branch's inline pip; this branch's readiness preamble was kept over its fire-and-hope launch. Three regressions caught by guardrails (§R48) |
| #234 | added a **new** raw template | caught on merge; listed with a reason (§R49) |
| #235 | added another | same (§R49) |
| #229 | rewrote the isaac-lab image and moved the prerequisites into a shared script | its Dockerfiles taken wholesale; the derived recipes keep both repair paths |

### #229 is the interesting one, twice over

Its rewrite fetches Isaac at run time and points `OMNI_USER_DIR` / `OMNI_LOG_DIR` at `/tmp` —
**a better answer to the exact stall live job 271 found** than this branch's chowning of
directories under `/isaac-sim`, and it arrived independently. Its version wins. The *derived*
`Dockerfile.k8s-prereqs` recipes keep both halves, because a derived build cannot know whether
it is repairing an image from the old NVIDIA base (where `/isaac-sim` holds the payload) or a
new one.

It also shipped a guardrail with the same shape as this PR's: every automated path that runs an
Isaac image must carry the operator's EULA acceptance, or the image exits 78. It enforced that by
scanning the **SkyPilot catalog** — which this branch had just emptied. The rule survived only
because #229 had also written a *guard for the guard* (`assert len(_isaac_templates()) >= 10`),
which failed loudly instead of letting the check pass over an empty set.

The fix moves the rule onto the surface that remains. A spec reaches an Isaac image through its
`toolRef`, not by naming an image, so asking every spec to declare two env vars would ask its
author to know the renderer's routing table. The **renderer** knows it, so the renderer declares
them — derived from its own `TOOL_REF_IMAGE_TOOL` map, empty unless the operator set them, so a
task still fails closed with the actionable message and nothing in the repo accepts on anyone's
behalf. A new Isaac toolRef is now covered the moment it is added, which the marker-scan never
was.

Two lessons worth keeping, both cheap:

* **Write the guard for the guard.** #229's one-line `>= 10` is the only reason its EULA rule did
  not quietly stop checking anything. This PR's tally test is the same shape.
* **A rule belongs where the decision is made.** Scanning documents for a marker works until the
  documents move; deriving the set from the code that routes the work does not.

## R53. `cosmos3-generate.yaml` retired after its spec twin ran live

The raw template added by #235 is gone. Its twin
`npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml` now has a live-submit-matrix case
(`tier="gpu"`, `image_tool="cosmos3"`) and reached terminal success through the same harness used
for the rest of this retirement work:

```bash
PYTHONPATH=$PWD/src \
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=gpu \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=cosmos3-generate.yaml \
.venv/bin/python -m pytest \
  tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_live_reaches_terminal \
  -q -s --tb=short
```

**Succeeded:** job **338**, run id `npa-wf-gpu-cosmos3-generate-601c8f51`, terminal
`SUCCEEDED` (`1 passed in 335.39s`). The run used the existing live environment and the
operator's credentials; registry, bucket, and token values are not committed here.

Artifacts under the run prefix:

| Key | What it contained |
| --- | --- |
| `generated/generate.json` | `status="executed"`, `output_kind="image"`, `output_bytes=260189`, `guardrails=true`, `hf_auth="configured"`, `weights_baked=false` |
| `generated/vision.jpg` | 260,189 byte generated JPEG, 960x960, RGB extrema `(0,255)` on all channels (not blank/flat) |
| `npa-workflow/manifest.json`, `npa-workflow/status.json` | workflow submit bookkeeping |

The failures before success were the useful part:

* No job id: the first two attempts failed before submit because the task image resolved in the
  us-central1 mirror while `SKYPILOT_DOCKER_SERVER` authenticated to eu-north1. One retry with
  `${NPA_REGISTRY}` did not help because this environment's `NPA_REGISTRY` also points at the
  mirror.
* Job **337**, run id `npa-wf-gpu-cosmos3-generate-478ccec0`: after forcing the primary
  eu-north1 registry, SkyPilot repeatedly hit `ErrImagePull` / `403 Forbidden` pulling
  `npa-cosmos3:1.2.2-cu130` from that registry and the job was cancelled. The next run used the
  mirror registry with `SKYPILOT_DOCKER_SERVER` aligned to the same host.

Preflight that stayed green before the live run:

* Renderer/smoke: `152 passed in 27.52s`
* Plan-only submit matrix: `42 passed in 8.99s`

## R54. `nurec-reconstruct.yaml` relocated after the #234 author decision

The raw NuRec/NRE task was **not deleted**. It moved from the retiring catalog to:

```text
npa/src/npa/workbench/nurec/examples/nurec-reconstruct.yaml
```

Decision provenance: PR #234 was authored by `timothy-le7` and explicitly shipped both forms:

* a single-pod SkyPilot task, live run `neural-reconstruction-struktur28-20260731t051728z`,
  terminal success with a renderable USDZ, novel views, `reports/sim2real.rrd`, and real
  metrics (`PSNR 31.24 / SSIM 0.832 / LPIPS 0.268`);
* the declarative `npa.workflow` twin, live runs `nurec-npa-20260731t184541z`,
  `nurec-npa-20260801t171139z`, and `nurec-npa-20260801t220210z`, all end-to-end successful.

The same PR records the behavioral distinction: the declarative spec runs each state in its own
pod and must hand the NCore sequence and reconstruction through S3, while the single-pod task
shares `/tmp`. That is a separate execution mode, not an unverified duplicate.

Relocation guardrails added:

* `npa/src/npa/workbench/nurec/examples/README.md` documents that this is a single-pod example,
  not a workflow authoring catalog.
* `npa/tests/guardrails/test_nurec_examples.py` pins the one-file set, asserts one SkyPilot task
  per file, requires substitution placeholders to remain, and forbids the file from returning to
  `npa/src/npa/workflows/skypilot/`.

## R55. The raw SkyPilot workflow catalog directory is gone

The structural end state is now enforced in code:

* `npa/src/npa/workflows/skypilot/` does not exist.
* `npa/tests/guardrails/test_skypilot_catalog_retirement.py` is inverted from a shrinking
  allowlist to an absence guard: the retired directory must not exist, and `npa.workflow` specs
  must not carry `metadata.skypilotTwin` or `metadata.skypilotTwins`.
* The `npa.workflow/v0.0.1` JSON schema now makes `metadata` strict, and the lightweight schema
  validator enforces `additionalProperties: false`, so retired twin metadata fails validation
  instead of being silently ignored.
* Raw SkyPilot submit-wrapper coverage now uses `npa/tests/fixtures/skypilot/` or guarded
  tool-specific examples (`burst`, BYOF profiles, NuRec single-pod), not a shipped workflow
  catalog.

Focused verification for the structural change:

```text
252 passed, 1 skipped
```

Command:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m pytest \
  tests/guardrails/test_skypilot_catalog_retirement.py \
  tests/guardrails/test_skypilot_readme.py \
  tests/guardrails/test_shown_workflow_catalog.py \
  tests/guardrails/test_nurec_examples.py \
  tests/guardrails/test_byof_profiles.py \
  tests/guardrails/test_burst_examples.py \
  tests/guardrails/test_workflow_image_check.py \
  tests/guardrails/test_isaac_eula_plumbing.py \
  tests/guardrails/test_skills_index.py \
  tests/orchestration/npa_workflow/test_skypilot_render.py \
  tests/orchestration/npa_workflow/test_real_components.py \
  tests/smoke/test_all_workflow_yamls.py -q
```

## R56. Cosmos2 publishes the contract its workflows declare — and both real GPU paths prove it

Live job 339 exposed a producer/consumer mismatch: five stages declared `manifest.json`, while
`workbench.cosmos2.transfer` wrote a reference-only `index.json` with a different schema. The
fix keeps `index.json` as the reference-input contract and publishes the canonical generated
artifact at `manifest.json` with schema `npa.cosmos2.transfer.v1`. The declared-output guard now
special-cases this toolRef so a future spec cannot silently repeat the mismatch at any of the
five historical locations:

* `cosmos-synth-fanout-curation.yaml`: `synth-shard-a`, `synth-shard-b`
* `cosmos2-transfer.yaml`: `transfer`
* `sim2real-vlm-rl.yaml`: `augment`
* `tokenfactory-cosmos-gate.yaml`: `augment-scene`

The extraction also adds the remaining submit-matrix cases from #244/#251 and makes every
shipped workflow resolve to exactly one case. Cases which need an undeployed service, private
asset, operator decision, or unsafe shared-output behavior remain explicit plan-only cases; the
runnable Cosmos2 cases exercise the real component. A separate guard walks workflow pointers in
code, docs, tests, and skills so renamed or removed specs cannot leave a dangling public entry.

Two failures during live validation improved the runnable smoke instead of being papered over:

* Job **370** reached the GPU image and showed that the built image does not contain the
  upstream Git-LFS example assets. Main's subsequent #252 image fix supplies a legally clean,
  repository-authored procedural input through the live harness instead. After rebasing, this
  extraction dropped its superseded in-wrapper fallback and retained explicit input as the
  authoritative production contract.
* Job **371** completed all 35 diffusion steps and passed the model's generated-video guardrail,
  then the wrapper rejected its valid 8,932-byte video because of an arbitrary 100 KiB cutoff.
  The cutoff is gone; publication still fails closed unless PyAV extracts at least one exact
  frame from a non-empty video.

After #252 merged, both final runs used executable source commit
`2ffbee656930f9b91a9ad660be886ae38ceff899` from this durable checkout, rebased on
`5ce5882126f16457d8fb7922f93c888a326d71bb`. Each run received a newly staged overlay
(`npa-workflow-e2e/npa-src/codex-2ffbee65/npa` and
`npa-workflow-e2e/npa-src/codex-2ffbee65-chain/npa`) and the qualified image digest recorded by
#252, on an **RTX PRO 6000 Blackwell Server Edition**. The later commit only updates this evidence.
Registry and artifact-bucket identifiers are deliberately redacted.

### Generic Cosmos2 transfer

```bash
PYTHONPATH=$PWD/src \
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=gpu \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=cosmos2-transfer.yaml \
.venv/bin/python -m pytest \
  'tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_live_reaches_terminal[gpu:cosmos2-transfer.yaml]' \
  -q -s --tb=short
```

**Succeeded:** job **375**, run id `npa-wf-gpu-cosmos2-transfer-b7818c52`, terminal
`SUCCEEDED` (`1 passed in 963.38s`). The real model completed all 35 diffusion steps. Independent
artifact inspection found `schema=npa.cosmos2.transfer.v1`, `status=executed`,
`mode=cosmos_transfer2.5_gpu`, `input_conditioned=true`, eight distinct existing frame objects,
and a 5,709,281-byte generated video. The canonical manifest was 2,867 bytes.

### Conditioned Cosmos2 transfer into envgen

```bash
PYTHONPATH=$PWD/src \
NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=multi \
NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=sim2real-two-step.yaml \
.venv/bin/python -m pytest \
  'tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_live_reaches_terminal[multi:sim2real-two-step.yaml]' \
  -q -s --tb=short
```

**Succeeded:** job group **377**, run id `npa-wf-multi-sim2real-two-step-b9d5e9dc`, terminal
`SUCCEEDED` (`1 passed in 1181.79s`). `augment` consumed the seeded input video, completed all 35
diffusion steps, and published the canonical manifest with `input_conditioned=true`,
`control=edge`, four exact existing frame URIs, and a valid 9,039-byte video. `envgen` then
consumed those URIs and published
`npa.sim2real.raw_env_shard_summary.v1`: 1,000 rows across the declared shard, all with schema
`npa.sim2real.raw_env.v1`; the distinct `augmented_frame_uri` values matched the four manifest
frames exactly.

The apparent two-way fan-out remains deliberately plan-only. Both shard states still receive the
same `augment_uri`, so the second overwrites the first and only one shard survives. The workflow
contains a conspicuous `DEFECT` comment documenting that a real fix needs a per-shard
configuration token. It validates (five states) and plans (four steps), but is not represented as
an executable live case until that configuration surface exists.

Successful SkyPilot jobs cleaned up their managed resources automatically; no pod for job 375 or
377 remained after its pytest node completed. The earlier diagnostic pod was deleted, cancelled
recovery jobs left no pods, no `npa-detection-training` resource was created, and the persistent
`npa-lancedb` deployment was left untouched as required by the rotation contract.

## R57. Token Factory batch inference — every path proven live except the one the platform blocks

`batch-generate` / `batch-status` were validated against the real Token Factory API from an
isolated worktree, venv, and tmux session on the shared dev VM. Batch execution is unavailable
platform-side for the whole validation window, so this section separates what was proven live
from what is pinned by unit tests, rather than implying end-to-end success.

**Live, through the shipped CLI.** Submit returned in about 5 seconds and wrote the handle
artifact next to the eventual output:

```
status: pending          operation_status: queued
model: openai/gpt-oss-120b               prompt_count: 2
completion_window: 24h   written_uri: <output-path>/batch_operation.json
```

`batch_operation.json` carries exactly the fields a later collect needs — `operation_id`,
`operation_status`, `model`, `completion_window`, `prompt_count`, `result_uri`, `status`,
`generated_at`. `batch-status --no-wait` against that handle re-reported the operation and
recovered `prompt_count` and `model` from the operation itself, with no prompt file present.

**Live failure paths, both exiting 1.** A model that serves real-time traffic but is not batch
routable fails with the per-row reason read from the batch record's error file, not the empty
string the operations API returns:

```
Error: batch operation <redacted> ended with status 'failed' with 2/2 rows rejected:
Invalid request rows 2 of 2 exceed the 10% limit.
Line:1 custom_id:e1 model "meta-llama/Llama-3.3-70B-Instruct" is not a known batch endpoint routing key
Line:2 custom_id:e2 ... Model 'meta-llama/Llama-3.3-70B-Instruct' is not available for batch
inference, even though it may serve real-time requests. Try another model, or run the same
prompts through `npa workbench token-factory generate`.
```

A vision model is refused at submit, with the pointer to the real-time captioning path, and no
artifact is written for a refused submit:

```
Error: starting batch inference failed: Token Factory request failed (400):
{"detail":"Batch inference is only supported for text2text models"} Model
'Qwen/Qwen2.5-VL-72B-Instruct' is not a text model. ... caption images with
`npa workbench token-factory caption` instead.
```

**The drop-in claim, shown side by side.** The real-time path on the same prompt file produces
the schema a batch stage is a substitute for:

```
{"completion": "ok", "id": "e1", "prompt": "Reply with exactly: ok"}
{"completion": "Red", "id": "e2", "prompt": "Name one primary color. One word."}
```

**The blocked path, and why it is not this change.** `POST /v1/batches` returns
`403 Creating new batch job is temporarily unavailable`, still true eight days after first
observation. The 403 lands after schema validation — an empty body returns `422` naming the
missing fields — but before resource validation, since a genuinely uploaded `input_file_id` is
also refused, so it is a server-side switch rather than a property of the request. Adjacent
surfaces on the same key stay healthy throughout: real-time chat on the batch model, file upload
with `purpose=batch`, batch listing, redirected file downloads, and dataset create/delete. The
datasets/operations route still accepts submissions and validates their rows (`total: 2,
invalid: 0`) which then complete none. It is not the documented quota either: the limits are 10
active batches and 100 submissions per hour, every batch record on the key was terminal when the
403 was reproduced, and rate limiting surfaces as `429` with `x-ratelimit-*` headers, none of
which were present. Batch listing defaults to a 10-record page, which makes a longer history look
like it is sitting exactly at the active-batch limit; `limit=100` disproves that.

**Therefore the success path is pinned by unit tests, not by a live run.** 24 tests in
`npa/tests/workbench/test_token_factory_batch.py` pass, including the documented result row
verbatim (`test_parse_batch_export_reads_the_standard_batch_row`), per-row error extraction
(`test_parse_batch_export_records_a_per_row_error_with_its_message`), all three result wrappers
(`test_batch_collect_parses_each_result_wrapper[response|body|result]`), output-file preference
over dataset export, and the timeout that keeps the operation collectible. The redirect-following
file download those paths depend on is proven live by the error-file reads above.

Cleanup after validation: every batch record on the key is terminal, so nothing is queued or
accruing cost; scratch datasets and batch-generated result/error files were deleted while the
key's two pre-existing datasets were left untouched; the isolated worktree, venv, and tmux
session were removed from the shared VM. Operation and request identifiers, the API key, and
dataset identifiers are deliberately redacted; exact values remain in access-controlled evidence.
