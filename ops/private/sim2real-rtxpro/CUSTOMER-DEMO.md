# Sim2Real — Customer Demo Handoff

**Laptop = interface only.** All pipeline compute runs on your Nebius Kubernetes
GPU cluster. Artifacts land on S3; your machine syncs and opens Rerun.

This is the same workflow customers use in production: **submit → cluster → S3 → inspect**.

---

## Prerequisites (once per machine)

1. **Python 3.10+** and **kubectl**
2. **NPA checkout** on branch `feat/sim2real-mandatory-stages`
3. **`npa configure`** → writes:
   - `~/.npa/config.yaml` — `storage.bucket`, `storage.registry`, `storage.endpoint_url`
   - `~/.npa/credentials.yaml` — S3 keys (and HF/NGC if needed by images)
4. **Kubeconfig** for your cluster → `~/.npa/clusters/<context>/kubeconfig`

---

## Run the demo (one command)

```bash
git clone --branch feat/sim2real-mandatory-stages \
  https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
./ops/private/sim2real-rtxpro/run-demo.sh
```

What happens:

| Step | Where | What |
| --- | --- | --- |
| 1 | Laptop | Bootstrap `npa/.venv`, preflight config/creds/kubeconfig |
| 2 | **Nebius cluster** | Submit K8s Job — orchestrator + GPU sibling Jobs (Cosmos VLM, Isaac held-out, …) |
| 3 | **Nebius cluster** | Staged CLI: `preamble` → `outer-iteration` × N → `finalize` |
| 4 | **Nebius S3** | Full artifact tree uploaded (`--upload-artifacts`) |
| 5 | Laptop | Sync `reports/sim2real.rrd` + stage JSON from S3 |
| 6 | Laptop | Open **Rerun web viewer** — walk the timeline |

---

## Modes

```bash
# Full flow: submit + wait + sync + Rerun (~15–30 min on cluster)
./ops/private/sim2real-rtxpro/run-demo.sh

# Re-open a completed Nebius run (presentation / rehearsal)
RUN_ID=<your-run-id> ./ops/private/sim2real-rtxpro/run-demo.sh

# Submit only — monitor separately (long jobs)
WAIT=0 ./ops/private/sim2real-rtxpro/run-demo.sh
# later:
RUN_ID=<run-id> SUBMIT=0 ./ops/private/sim2real-rtxpro/run-demo.sh

# Sync + artifacts only, no browser
VISUALIZE=0 RUN_ID=<run-id> ./ops/private/sim2real-rtxpro/run-demo.sh
```

`run-local-demo.sh` is an alias for `run-demo.sh`.

---

## Rerun walkthrough (~30 s)

Open the URL printed by the script. In the viewer:

1. **rollouts/…/camera** — action rollouts (Stage 7)
2. **Critique overlays** — VLM scores (Stage 8, Cosmos on cluster)
3. **signal/reward** — RL signal (Stage 9)
4. **heldout/scores** — held-out eval (Stage 10, Isaac on cluster)

---

## Configuration knobs

Set via environment before `run-demo.sh` (same as cluster submit):

| Variable | Default | Meaning |
| --- | --- | --- |
| `KUBECONTEXT` | from `~/.npa/config.yaml` | Kubernetes context |
| `INNER_ITERATIONS` | `1` | Inner loop depth |
| `OUTER_ITERATIONS` | `2` | Outer loop / loop-back |
| `S3_PREFIX` | `sim2real-b` | S3 prefix parent |
| `SUCCESS_THRESHOLD` | `0.45` | Held-out promote threshold |

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `config.yaml missing` | `npa configure` |
| `kubeconfig not found` | Install cluster kubeconfig under `~/.npa/clusters/<context>/` |
| Job failed | `./ops/private/sim2real-rtxpro/monitor-k8s-job.sh sim2real-<run-id>` |
| No `.rrd` on S3 | Check `stage_14_rerun_viz` tier in `reports/sim2real-report.json` |
| Mac: script not found | `git pull` latest `feat/sim2real-mandatory-stages` |

---

## Security — credentials never in git

| Secret | Where it lives | Never |
| --- | --- | --- |
| S3 keys, HF/NGC tokens | `~/.npa/credentials.yaml` (chmod 600) | Committed files, YAML env blocks, logs |
| Kubeconfig | `~/.npa/clusters/<context>/kubeconfig` | Repo or walkthrough docs |
| Bucket / registry / cluster | `~/.npa/config.yaml` | Hardcoded in scripts (read at runtime) |

Cluster submit uses **Kubernetes `secretRef`** (`hf-ngc-tokens`, `npa-storage-credentials`) —
credentials are not embedded in generated Job manifests.

Generated gitignored files (`env.local`, `*.local.md`) may contain your bucket/registry
from config — do not commit them.

---

## What is NOT this workflow

**Reference rehearsal** (no cluster, laptop simulates stages) is for unit tests only —
not the customer path. Do not use for production demos.

---

## Related scripts

| Script | Role |
| --- | --- |
| `run-demo.sh` | **Customer entrypoint** (this doc) |
| `submit-k8s-staged-job.sh` | Cluster submit (called by run-demo) |
| `monitor-k8s-job.sh` | Poll job until complete |
| `prestage-offline-run.sh` | S3 → local sync |
| `setup-local-operator.sh` | Generate `env.local` from config |
