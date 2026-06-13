# Sim2Real — Customer Demo Handoff

**Laptop = interface only.** All pipeline compute runs on your Nebius Kubernetes
GPU cluster. Artifacts land on S3; your machine syncs and opens Rerun.

**Trigger model:** upload a complete LeRobot batch to S3, then **explicitly**
start the pipeline. There is **no S3 polling** — you decide when the batch is
ready and when to run.

---

## Prerequisites (once per machine)

1. **Python 3.10+** and **kubectl** (`brew install python@3.12 kubectl` on Mac)
2. **Nebius CLI** for mk8s auth (`brew install nebius/tap/nebius`) — kubeconfig uses `nebius mk8s … get-token`
3. **NPA checkout** on branch `feat/sim2real-mandatory-stages`
3. **`npa configure`** → writes:
   - `~/.npa/config.yaml` — `storage.bucket`, `storage.registry`, `storage.endpoint_url`
   - `~/.npa/credentials.yaml` — S3 keys (and HF/NGC if needed by images)
4. **Kubeconfig** for your cluster → `~/.npa/clusters/<context>/kubeconfig`

**Operator shortcut (private repo):** if you have access to the private walkthrough
repo, run `./setup-npa-local.sh` once — it installs `config.yaml`, `credentials.yaml`,
and kubeconfig into `~/.npa/` (mode 600). Otherwise use `npa configure` manually.

---

## Production flow (upload → trigger)

### 1. Upload LeRobot data to S3

Upload a **complete** LeRobot dataset tree, for example:

```text
s3://<bucket>/sim2real-triggers/<batch-id>/lerobot-<task>/
  meta/info.json
  meta/episodes.jsonl
  data/…/*.parquet
  videos/…/*.mp4
```

Use your bucket credentials (`~/.npa/credentials.yaml`). Wait until the full
batch is uploaded before triggering — partial uploads are not detected automatically.

### 2. Trigger the pipeline

**Operator pack (Mac — recommended):**

```bash
cd ~/npa-sim2real-demo
./run.sh demo          # cleanup + submit (customer replication from scratch)
./run.sh status <RUN_ID>
./run.sh sync <RUN_ID>
```

**From repo checkout:**

```bash
export TRIGGER_DATASET_URI=s3://<bucket>/sim2real-triggers/<batch-id>/lerobot-<task>/
./ops/private/sim2real-rtxpro/trigger-pipeline.sh
```

Optional:

```bash
export TRIGGER_DATASET_ID=lerobot/<task>
export RUN_ID=<batch-id>          # ties run artifacts to your batch name
WAIT=0 ./ops/private/sim2real-rtxpro/trigger-pipeline.sh   # submit only
```

What happens:

| Step | Where | What |
| --- | --- | --- |
| 1 | Laptop | S3 preflight (`meta/info.json` or `data/*.parquet`) |
| 2 | Laptop | Bootstrap `npa/.venv`, preflight config/creds/kubeconfig |
| 3 | **Nebius cluster** | Submit K8s Job — orchestrator + GPU sibling Jobs |
| 4 | **Nebius cluster** | Staged CLI: `preamble` → `outer-iteration` × N → `finalize` |
| 5 | **Nebius S3** | Full artifact tree uploaded (`--upload-artifacts`) |
| 6 | Laptop | Sync `reports/sim2real.rrd` + stage JSON from S3 |
| 7 | Laptop | Open **Rerun web viewer** |

### 3. Next batch (real-world flywheel)

1. Deploy promoted checkpoint on your robot (**customer BYO** — Stage 12 seam)
2. Collect new teleop → upload new LeRobot batch to a **new or versioned** S3 prefix
3. Export new `TRIGGER_DATASET_URI` and run `trigger-pipeline.sh` again

Each batch = one explicit trigger. No background watcher required.

---

## Stock Franka setup (no mesh / robot upload)

Leave `ASSETS_URI`, `SCENE_SPEC_URI`, and `ROBOT_SPEC_URI` unset. Stage 2 uses
built-in Franka Panda + Isaac lift-cube tabletop. You only upload the LeRobot trigger.

Full operator walkthrough (Mac paths, stage checklist, S3 + jq + Rerun):
**[FRANKA-STOCK-GUIDE.md](./FRANKA-STOCK-GUIDE.md)**

## Demo / rehearsal (stock trigger)

For a smoke run without your own upload (uses default pusht path under your bucket):

```bash
./ops/private/sim2real-rtxpro/run-demo.sh
```

Reuse a completed cluster run (sync + Rerun only):

```bash
RUN_ID=<your-run-id> ./ops/private/sim2real-rtxpro/run-demo.sh
```

---

## Modes

```bash
# Full flow: preflight + submit + wait + sync + Rerun
./ops/private/sim2real-rtxpro/trigger-pipeline.sh

# Submit only — monitor separately
WAIT=0 ./ops/private/sim2real-rtxpro/trigger-pipeline.sh

# Sync + Rerun for a completed run
RUN_ID=<run-id> SUBMIT=0 ./ops/private/sim2real-rtxpro/run-demo.sh

# No browser
VISUALIZE=0 RUN_ID=<run-id> ./ops/private/sim2real-rtxpro/run-demo.sh
```

`run-local-demo.sh` is an alias for `run-demo.sh`.

---

## Configuration knobs

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRIGGER_DATASET_URI` | required for `trigger-pipeline.sh` | Uploaded LeRobot prefix on S3 |
| `TRIGGER_DATASET_ID` | `lerobot/pusht` | LeRobot dataset id metadata |
| `RUN_ID` | auto on submit | Pipeline run id (S3 artifact prefix) |
| `KUBECONTEXT` | from `~/.npa/config.yaml` | Kubernetes context |
| `INNER_ITERATIONS` | `1` | Inner loop depth |
| `OUTER_ITERATIONS` | `2` | Outer loop / loop-back |
| `S3_PREFIX` | `sim2real-b` | S3 prefix parent for run outputs |
| `SUCCESS_THRESHOLD` | `0.45` | Held-out promote threshold |

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `no LeRobot batch at …` | Finish upload; need `meta/info.json` or `data/*.parquet` |
| `TRIGGER_DATASET_URI` missing | Export S3 path before `trigger-pipeline.sh` |
| `config.yaml missing` | `npa configure` or private `setup-npa-local.sh` |
| Job failed | `./ops/private/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<run-id>` |
| Mac venv path error | `git pull` latest branch |
| `fork/exec /usr/local/bin/nebius: no such file` | `brew install nebius/tap/nebius`; scripts patch kubeconfig to your `nebius` path |
| `kubectl cannot reach cluster` | `nebius profile list` — need profile `npa-mk8s` from operator pack |

---

## Security — credentials never in git

| Secret | Where it lives |
| --- | --- |
| S3 keys, HF/NGC tokens | `~/.npa/credentials.yaml` (chmod 600) |
| Kubeconfig | `~/.npa/clusters/<context>/kubeconfig` |
| Bucket / registry / cluster | `~/.npa/config.yaml` |

Cluster submit uses Kubernetes `secretRef` — credentials are not embedded in Job YAML.

---

## Related scripts

| Script | Role |
| --- | --- |
| **`trigger-pipeline.sh`** | **Customer entrypoint** — upload then trigger |
| `run-demo.sh` | Demo/rehearsal or sync-only |
| `submit-k8s-staged-job.sh` | Cluster submit (called internally) |
| `monitor-k8s-job.sh` | Poll job until complete |
| `prestage-offline-run.sh` | S3 → local sync |
| `setup-local-operator.sh` | Generate `env.local` from config |
