---
name: burst
description: Use to run one gang-scheduled multi-node GPU job through SkyPilot without authoring a workflow — `npa burst submit` wires torchrun rendezvous across nodes, while `submit-yaml` supports one task, `${VAR}` substitution, and optional generic registry credentials.
---

# Burst (one coupled multi-node GPU job)

Burst is the escape hatch for a **single** distributed task: gang-schedule N
nodes, run one command under `torchrun`, get logs, exit. It is deliberately not a
workflow surface — there is no plan, no stage graph, no decision artifact, and
nothing for a `toolRef` to describe.

**If you need a second stage, you need a workflow.** Author an
`npa.workflow/v0.0.1` spec under `npa/workflows/workbench/npa-workflows/` instead
(`skills/workflows/author-npa-workflow/SKILL.md`). `submit-yaml` rejects
multi-task documents, and a guardrail pins the example set so a pipeline cannot
quietly grow here.

## Submit a distributed run directly

```bash
npa burst submit \
  --image <registry>/<image>:<tag> \
  --nodes 2 \
  --gpu-per-node H100:8 \
  --entrypoint "python train.py --config configs/run.yaml" \
  --name <job-name> \
  --json
```

All four of `--image`, `--nodes`, `--gpu-per-node`, `--entrypoint` are required.
A plain image reference is rendered as `docker:<image>`. `--json` prints only the
serialized job handle, which is what you want when scripting.

**The entrypoint runs under `torchrun`, not directly.** Burst derives the
rendezvous from SkyPilot's node environment and exports `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE`, and `RANK`, then execs `torchrun` with `--nnodes`,
`--node-rank`, and `--master-addr` already set. Two consequences:

- **`torchrun` must exist in the image.** Burst checks and fails with an explicit
  message if it is missing.
- **Do not set the distributed variables yourself** and do not wrap your command
  in another `torchrun`. Pass the training command as if it were a single-process
  entrypoint.

`--gpu-per-node` is a SkyPilot accelerator spec such as `L40S:1` or `H100:8`, and
it is per node: all GPUs of one task land on one node. Confirm the name the
cluster actually advertises with `npa workbench workflow gpus --cluster <name>` —
Kubernetes names accelerators after node labels, so the spec string is discovered
rather than guessed.

## Submit a single-task YAML

```bash
npa burst submit-yaml <task.yaml> \
  --name <job-name> \
  --var NPA_RUN_ID=<run-id> \
  --var NPA_OUTPUT_URI=s3://<bucket>/<prefix>/<run-id>/ \
  --json
```

`submit-yaml` loads one SkyPilot task document, substitutes `${VAR}` placeholders
from repeated `--var`/`-v`, and **refuses to submit while any placeholder is
unresolved** — an unsubstituted `${VAR}` is an error, never a literal sent to the
cluster. `--name` defaults to the task name in the YAML.

Unlike the npa.workflow renderer, `${VAR}` placeholders are the intended surface
here. The committed example keeps them precisely so no concrete registry id,
bucket name, or run id is ever baked in:
`npa/src/npa/burst/examples/isaac-lab-cosmos-sdg-burst-smoke.yaml`.

## Registry authentication

Official NPA GHCR development and release images pull anonymously. Docker access
on your dev VM does not authenticate freshly created SkyPilot worker VMs to an
optional operator-controlled private registry. For that BYOF case, set
`SKYPILOT_DOCKER_SERVER`, `SKYPILOT_DOCKER_USERNAME`, and
`SKYPILOT_DOCKER_PASSWORD` for the image's exact registry host; `submit-yaml`
forwards only that explicit credential set. NPA never mints a provider registry
token. Kubernetes retries image pulls indefinitely, so bad BYOF credentials can
look like a job that never starts rather than one that fails.

## Watch and collect

```bash
npa burst status <job-id-or-handle-json>
npa burst logs <job-id-or-handle-json> --follow
npa burst logs <job-id-or-handle-json> --tail 200
```

Both accept either a job ID or the serialized handle `--json` printed at submit.
`--config <path>` points at a specific SkyPilot global config when you are not
using the default runtime.

## Gotchas

- **`--nodes N` with `--gpu-per-node NAME:M` requests M GPUs on each of N nodes.**
  It cannot spread one task's M GPUs across nodes. Asking for more GPUs per node
  than a node has will not schedule, and this presents as a pending job rather
  than a rejection.
- **Gang scheduling means all-or-nothing.** The job waits for all N nodes; partial
  capacity leaves it pending indefinitely, not running degraded.
- **A pending burst job is still holding your intent, not your capacity.** Check
  `status` before assuming the cluster is full, and cancel work you no longer
  want — see `skills/atomic/teardown-and-cost/SKILL.md`.
- **Burst has no durable S3 run ledger.** Unlike `workbench workflow`, there is no
  `artifacts`/`list` surface and no resume. If you want durable per-stage state,
  resume, and lineage, that is the workflow runtime, not burst.
- **Do not add example YAMLs to `npa/src/npa/burst/examples/`.** The set is pinned
  by `npa/tests/guardrails/test_burst_examples.py`; a new file must first answer
  "why is this not a spec?".

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_burst_examples.py -q
```
