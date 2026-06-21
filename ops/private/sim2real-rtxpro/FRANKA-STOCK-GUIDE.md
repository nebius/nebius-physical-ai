# Sim2Real — Stock Franka Operator Guide

**Audience:** RTX PRO demo operators on a Mac laptop. GPU work runs on Nebius mk8s; artifacts
land on S3.

**Monitor stage-status fix:** use NPA `main` — run `./setup.sh` in the walkthrough repo to refresh the venv before `npa workbench workflow status`.

## First-time bucket setup (new region/bucket)

When `storage.bucket` changes (for example `lerobot-ccc9d3c7` in `us-central1`), run once:

```bash
cd ~/npa-sim2real-demo/nebius-physical-ai
git pull origin main

# 1) Copy validated stock LeRobot pusht trigger into your bucket
./ops/private/sim2real-rtxpro/seed-stock-trigger.sh

# 2) Pin the seeded URI in ~/.npa/config.yaml
#    storage.sim2real_stock_trigger_uri: s3://<bucket>/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/

# 3) Regenerate operator env + sync cluster secret (endpoint must match config)
./ops/private/sim2real-rtxpro/setup-local-operator.sh
./ops/private/sim2real-rtxpro/sync-cluster-storage-secret.sh

# 4) Re-source operator env
source ~/.npa/sim2real-operator.env   # or ops/private/sim2real-rtxpro/env.local
```

`submit-k8s-staged-job.sh` now preflights trigger read + `sim2real-b/` write + cluster
secret endpoint **before** applying the Job.

## Submit stock Franka run

```bash
cd ~/npa-sim2real-demo
./run.sh trigger
# or
cd ~/npa-sim2real-demo/nebius-physical-ai
./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh
```

## kubectl logs (orchestrator pod)

```bash
export KUBECONFIG="${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved"
export PATH="${HOME}/.nebius/bin:${PATH}"

JOB=sim2real-sim2real-staged-<RUN_ID>
kubectl --context npa-rtxpro-mk8s get pods -n default -l run-id=sim2real-staged-<RUN_ID>
kubectl --context npa-rtxpro-mk8s logs -n default -l run-id=sim2real-staged-<RUN_ID> --tail=200
```

Replace `<RUN_ID>` with the timestamp id (for example `20260615t172625z`).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Monitor: `stage_01_trigger` / `stage_02_assets` / `stage_06_tokens` PENDING while later stages SUCCEEDED | Status polled stale S3 keys (`assets_manifest.json`, `cosmos2-transfer-result.json` only) instead of `workflow_state.json` + consumed specs / augment manifest / envgen split | Re-run from current branch; monitor prefers `state/workflow_state.json`, then `consumed_scene_spec.json`+`consumed_robot_spec.json`, `augment/manifest.json`, and train+heldout `envs.jsonl` (tokens folded into envgen) |
| Monitor: early stages PENDING while later SUCCEEDED | Stale monitor artifact paths | `./setup.sh` in walkthrough repo to refresh NPA checkout |
| Monitor: `stage_01_trigger` / all stages PENDING, no S3 artifacts | Orchestrator died before first upload (often stage 2) | `kubectl logs` on orchestrator pod — see AccessDenied row below |
| Preflight: `no LeRobot batch` | Stock trigger not seeded on new bucket | `./seed-stock-trigger.sh` then set `sim2real_stock_trigger_uri` |
| Preflight: `cannot write to s3://.../sim2real-b/` | IAM keys lack PutObject on bucket/region | Fix bucket IAM; verify `storage.endpoint_url` matches bucket region |
| Pod: `AccessDenied` on `PutObject` to `sim2real-b/.../stage_02_assets/` | Cluster `npa-storage-credentials` stale (wrong endpoint or keys) | `./sync-cluster-storage-secret.sh` — secret endpoint must match `storage.endpoint_url` in `~/.npa/config.yaml` |
| `ValueError: Invalid endpoint:` (empty) during preflight or secret sync | `AWS_ENDPOINT_URL` empty — credentials lacked endpoint while shell exported blank | Re-run `./setup-local-operator.sh` or ensure `~/.npa/config.yaml` has `storage.endpoint_url: https://storage.us-central1.nebius.cloud`; `./sync-cluster-storage-secret.sh` reads config, not empty env |
| `./run.sh trigger` syncs old failed run instead of submitting | Stale `RUN_ID` still exported in shell | `unset RUN_ID` then `./run.sh trigger` (trigger always clears RUN_ID; unset if you exported it earlier) |
| Pod: `AccessDenied` on trigger read | Same credential/endpoint mismatch | Sync secret; confirm trigger exists with `seed-stock-trigger.sh` |
| `git clone` failure in pod | Wrong `NPA_SOURCE_REF` or GitHub outage | Job uses `main` by default |
| ImagePullBackOff | Stale registry token | Re-run submit (refreshes `npa-nebius-registry`) |

### RUN_ID `sim2real-staged-20260615t172625z` (investigated)

- **Job env:** `NPA_SIM2REAL_TRIGGER_DATASET_URI=s3://lerobot-ccc9d3c7/sim2real-triggers/trigger-validate-20260611T154016Z/lerobot-pusht/` (matches `~/.npa/config.yaml`)
- **Pod logs:** failed uploading `stage_02_assets/consumed_scene_spec.json` with `AccessDenied`
- **Cluster secret:** `npa-storage-credentials.AWS_ENDPOINT_URL` was `https://storage.eu-north1.nebius.cloud` while bucket/job used `us-central1`
- **Operator creds:** same AccessDenied on read/write to `lerobot-ccc9d3c7` until IAM + endpoint are aligned

**Mac recovery steps:**

1. `./seed-stock-trigger.sh` (if preflight reports missing trigger)
2. `./sync-cluster-storage-secret.sh`
3. `./setup-local-operator.sh` and re-source env
4. Delete failed job: `kubectl --context npa-rtxpro-mk8s delete job sim2real-sim2real-staged-20260615t172625z -n default`
5. Re-submit: `./run.sh trigger` (preflight must pass before apply)

No re-seed needed if step 1 preflight already passes; re-trigger required after secret sync.

## Hosted Rerun viewer (shared per cluster)

Stage 14 uploads `reports/sim2real.rrd` to S3. NPA deploys **one LoadBalancer per mk8s
cluster** (not per `run_id`). The `public_url` stays stable so teammates can bookmark it;
pointing the viewer at a new run updates the served recording without a new external IP.

```bash
cd ~/npa-sim2real-demo
./run.sh rerun-host sim2real-staged-<RUN_ID>

# Or from the repo checkout:
npa workbench sim2real rerun serve --run-id sim2real-staged-<RUN_ID>
```

Serve a different completed run on the **same** URL:

```bash
./run.sh rerun-host sim2real-staged-<OTHER_RUN_ID>
```

Teardown the shared viewer for this cluster (`--destroy` is cluster-scoped, not run-scoped):

```bash
npa workbench sim2real rerun serve --run-id sim2real-staged-<ANY_VALID_RUN_ID> --destroy
```

The E2E report JSON includes `rerun_serve.public_url` when auto-serve runs during
`run_finalize`. Deployment name pattern: `npa-sim2real-rerun-viewer` or
`npa-sim2real-rerun-<k8s-context-slug>` (for example `npa-sim2real-rerun-npa-rtxpro-mk8s`).

If `public_url` is pending, the LoadBalancer is still provisioning or hit a VPC public-IP
quota (`vpc.ipv4-address.public.count`). Wait and re-run serve; inspect
`kubectl describe svc npa-sim2real-rerun-npa-rtxpro-mk8s`. Do not use laptop port-forward
on the default operator path.
