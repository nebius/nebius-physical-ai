---
name: debug-failed-run
description: Use when a workflow run failed, hung, or produced no artifacts — an ordered triage from run id to root cause across status, stage logs, S3 evidence, pod-level reasons, and the resume-vs-cancel decision.
---

# Debug a failed workflow run

Start from the run id and work outward. Every step below is read-only until the
final decision, so none of it costs GPU time or destroys the evidence you need.

## 0. Find the run id

```bash
npa workbench workflow list --project <alias> --limit 50 --json
```

Runs are durable S3 objects, so this works after the controller, the cluster, or
your shell is gone. If you only have an S3 prefix, every command below accepts an
`s3://` run prefix wherever it accepts a run id.

## 1. Status first — it names the failure class

```bash
npa workbench workflow status <run-id> --project <alias> --json
```

This is the single most informative command, because **for a PENDING job it
reports the pod-level reason** rather than just "pending". That distinction is
the whole triage:

- `PENDING` + `ImagePullBackOff` → registry problem, go to step 4. The job will
  never fail on its own; Kubernetes retries pulls forever.
- `PENDING` + `Unschedulable` → capacity or accelerator-name problem, step 5.
- `FAILED` → the payload ran and exited non-zero, step 3.
- `FAILED_STARTUP` → the controller itself failed identically several times
  (threshold `--startup-failure-threshold`, default 3). This is infrastructure,
  not your payload.

Add `--watch --interval 10` to follow a run to a terminal state. Use `--cached`
only when the live controller is unreachable: its output is explicitly marked
CACHED and is **not automation-trustworthy** — never gate a decision on it.

## 2. Enumerate what the run actually produced

```bash
npa workbench workflow artifacts <run-id> --project <alias> --json
npa workbench workflow artifacts <run-id> --stage <stage-name> --json
```

Compare produced artifacts against the stages the spec declares. The first stage
with no outputs is where to look, and it is often earlier than the stage that
reported the error. A stage that "succeeded" with a zero-byte or missing declared
output is a real failure that the exit code missed.

## 3. Read the failing stage's logs

```bash
npa workbench workflow logs <run-id> --stage <stage-name>
npa workbench workflow logs <run-id> --stage <stage-name> --follow   # live job
npa workbench workflow logs <run-id> --stage <stage-name> --cached   # S3 only
```

`--follow` tails live SkyPilot logs and needs the controller; `--cached` reads
persisted object-storage evidence only and never queries the controller, which is
what you want for a post-mortem on a torn-down run. `--json` emits the log-source
contract so you can see *which* source answered — worth checking before you
conclude "there are no logs", because an empty live tail and an unread S3 object
look identical otherwise.

## 4. Registry and image problems (the most common stall)

A `403` from the registry does not fail the job, it stalls it. Listing a
repository's tags is a *different permission* from pulling it, so a `200` on
`/v2/<repo>/tags/list` proves nothing.

```bash
npa workbench workflow preflight-images <spec.yaml> --project <alias> --json
```

This reproduces the exact manifest fetch a worker performs, with the credentials
the run injects, and reports each image `ok` / `not_found` / `forbidden`.

- `not_found` → the image was never built and pushed into *this* project's
  registry. Nothing mirrors workbench images into a registry created by
  `npa configure`; see `skills/atomic/build-and-push-image/SKILL.md`.
- `forbidden` → Nebius IAM registry tokens expire. Refresh the pull secret in the
  namespace that owns the pod rather than hand-minting per run; the canonical
  helper is `npa.clients.nebius_auth.mint_nebius_iam_token` and, for Kubernetes,
  `npa.workflows.sim2real.registry_auth.ensure_registry_pull_secret_for_images`.

## 5. Scheduling problems

```bash
npa workbench workflow gpus --cluster <name> --json
npa workbench workflow gpus --cluster <name> --spec <spec.yaml> --json
```

Two distinct causes present identically as `FAILED_PRECHECKS` or "cluster does
not contain any instances satisfying the request" — neither is a capacity
shortage:

- **Accelerator name mismatch.** Kubernetes names GPUs after node labels, and the
  name changes while the NVIDIA GPU operator is still labelling nodes. Submit
  remaps this automatically; `--no-resolve-accelerators` disables the remap.
- **`NAME:N` needs N GPUs on one node.** SkyPilot places all GPUs of a task on a
  single node, so `NAME:2` can never schedule across 2 nodes × 1 GPU. `gpus`
  prints the requestable quantity per node.

## 6. When the run looks fine but the controller does not

```bash
npa skypilot status --project <alias> --context <context>
npa skypilot verify --cluster <name> --output-format json
```

A silent multi-minute submit is usually the pinned Kubernetes client, not your
spec: SkyPilot 0.12.2 does not cap the client version and client 36+ makes every
`pod_config` fail validation, so the controller retries forever.
`npa skypilot bootstrap` pins a working client and repairs an existing venv.

## Decide: resume, re-load, cancel, or fix config

- **Resume** when the failure was transport/capacity/preemption and the completed
  waves are valid. Submit with `--resume-run <same-id>`; the runtime adopts
  complete waves from declared S3 outputs and resubmits only incomplete work.
  Never resume to paper over a bad input — you will inherit the bad artifacts.
- **Re-load artifacts only** when every stage succeeded and only the final
  artifact load failed. This never relaunches stages:

  ```bash
  npa workbench workflow load-artifact <run-id> --project <alias> --json
  ```

- **Cancel** when the run is wedged and you want the capacity back. Cancel is
  repeat-safe: a planned or staged run that never launched is a successful no-op.

  ```bash
  npa workbench workflow cancel <run-id> --project <alias> --json
  ```

- **Fix config and start a fresh run** for anything the run itself cannot repair:
  wrong bucket, wrong registry, missing gated-model acceptance, wrong GPU shape.

Always cancel before tearing down shared infrastructure; see
`skills/atomic/teardown-and-cost/SKILL.md` for the ordering.

## Post-mortem across runs

Once a run is triaged, ingest it so the next failure is comparable rather than
re-investigated from scratch:

```bash
npa workbench insights ingest-run --input-path <run-prefix> --output-path <store>
npa workbench insights compare --input-path <store> \
  --base-run <last-good> --candidate-run <failed>
```

Ingestion is non-invasive — it scans the run prefix for schemas the tools already
emit, so nothing needed to be instrumented in advance. See
`skills/tools/insights/SKILL.md`.

## Gotchas

- **A green pipeline is not a good result.** Completion proves the graph ran, not
  that the policy or dataset is any good. Read the measured metric, and never
  weaken a threshold to turn a run green.
- **Do not diagnose with raw `sky` commands.** Cancelling or relaunching by name
  bypasses the transactional launch reconciliation and can orphan or duplicate a
  managed job. Use the `npa` commands, which adopt one immutable job ID.
- **`--cached` output is evidence, not truth.** It is last-known state; a run can
  have moved on or died since.
- **Check the obvious config mismatch before deep triage.** `npa configure --show`
  against the run's actual bucket/registry catches a surprising share of "the
  cluster is broken" reports.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
