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
