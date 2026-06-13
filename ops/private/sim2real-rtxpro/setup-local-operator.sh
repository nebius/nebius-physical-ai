#!/usr/bin/env bash
# Generate gitignored operator files from ~/.npa/config.yaml (no secrets written).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/npa/.venv/bin/python"

"${PY}" - <<'PY' "${OUT_DIR}"
import os, sys, yaml
from pathlib import Path
out = Path(sys.argv[1])
cfg = yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
storage = cfg.get("storage") or {}
projects = cfg.get("projects") or {}
rtx = projects.get("rtxpro") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
registry = storage.get("registry", cfg.get("registry", ""))
run_id = "rtxpro-demo"
env_lines = [
    f"NPA_SIM2REAL_RUN_ID={run_id}",
    f"NPA_SIM2REAL_BUCKET={bucket}",
    "NPA_SIM2REAL_PREFIX=sim2real-b",
    "NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht",
    f"NPA_SIM2REAL_TRIGGER_DATASET_URI=s3://{bucket}/sim2real-triggers/{run_id}/lerobot-pusht/",
    f"ASSETS_URI=s3://{bucket}/sim2real-assets/pusht/",
    f"SCENE_SPEC_URI=s3://{bucket}/sim2real-assets/pusht/scene-spec.json",
    f"AWS_ENDPOINT_URL={endpoint}",
    f"S3_ENDPOINT_URL={endpoint}",
    "NPA_SIM2REAL_SIM_BACKEND=isaac",
    "NPA_SIM2REAL_ISAAC_TASK=Isaac-Lift-Cube-Franka-v0",
    "INNER_ITERATIONS=2",
    "OUTER_ITERATIONS=2",
    "SUCCESS_THRESHOLD=0.45",
    "ROLLOUT_COUNT=3",
    "HELDOUT_ENV_COUNT=8",
    "NPA_SIM2REAL_K8S_CONTEXT=npa-rtxpro-mk8s",
]
if registry:
    reg = registry.rstrip("/")
    env_lines.extend([
        f"TRAINER_IMAGE={reg}/npa-lerobot-vlm-rl:0.1.0",
        f"VLM_IMAGE={reg}/npa-cosmos3-reason:3.0.1-genuine-sm120",
        f"EVAL_IMAGE={reg}/npa-sim2real-eval:0.1.1-genuine-sm120",
        f"ISAAC_IMAGE={reg}/npa-isaac-lab:2.3.2.post1",
    ])
(out / "env.local").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
md = f"""# RTX PRO Sim2Real — Local Operator Runbook (generated)

> Secrets: `~/.npa/credentials.yaml` (HF, NGC, S3 keys). **Not copied here.**

## Cluster

- Context: `npa-rtxpro-mk8s`
- Kubeconfig: `~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig`
- Project: `{rtx.get('project_id', '')}`

## Storage

- Bucket: `{bucket}`
- Endpoint: `{endpoint}`
- Trigger: `s3://{bucket}/sim2real-triggers/{run_id}/lerobot-pusht/`
- Assets: `s3://{bucket}/sim2real-assets/pusht/`

## Commands

```bash
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
source {out}/env.local
npa workbench health sim2real --checks all \\
  --k8s-context npa-rtxpro-mk8s \\
  --k8s-kubeconfig ~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
npa workbench workflow submit npa/workflows/workbench/sim2real/runbook.yaml \\
  --env-file {out}/env.local
```

## Rerun

Golden run for offline walkthrough: `rtxpro-staged-2x2-20260613t011356z`
(`stage_14_rerun_viz` tier **WORKS**, `.rrd` on S3).

```bash
./ops/private/sim2real-rtxpro/prestage-offline-run.sh
pip install rerun-sdk
rerun /tmp/sim2real-prestage/rtxpro-staged-2x2-20260613t011356z/reports/sim2real.rrd
# Or after manual download from s3://{bucket}/sim2real-b/<run-id>/reports/sim2real.rrd
```

## Validated local accuracy delta (staged reference mode)

| Run | Inner iters | Success rate | Reward trend | Decision |
| --- | --- | --- | --- | --- |
| baseline | 1 | 0.0 | [0.29] | loop back |
| trained | 2 | 1.0 | [0.29, 0.72] | promote |

Reproduce: `/tmp/run_staged_twice.sh` on nebius-dev-vm (see PR validation log).
"""
(out / "RUNBOOK.local.md").write_text(md, encoding="utf-8")

golden_run = "rtxpro-staged-2x2-20260613t011356z"
reg_display = registry.rstrip("/") if registry else "<configure storage.registry>"
bucket_display = bucket or "lerobot-d87cf691"
repo_root = out.resolve().parents[2]
walkthrough = f"""# Sim2Real Demo — Private Walkthrough (generated)

> Machine-specific notes. **Gitignored.** Regenerate: `./ops/private/sim2real-rtxpro/setup-local-operator.sh`

## Customer demo (cluster compute, laptop = Rerun)

```bash
cd {repo_root}
./ops/private/sim2real-rtxpro/run-demo.sh
RUN_ID={golden_run} ./ops/private/sim2real-rtxpro/run-demo.sh   # reuse completed run
```

See `{repo_root}/ops/private/sim2real-rtxpro/CUSTOMER-DEMO.md`.

---

## Infrastructure

| Item | Value |
| --- | --- |
| Cluster context | `npa-rtxpro-mk8s` |
| Kubeconfig | `~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig` |
| Bucket | `{bucket_display}` |
| S3 endpoint | `{endpoint}` |
| Artifact prefix | `sim2real-b/<run-id>/` |
| Registry | `{reg_display}` |
| Project | `{rtx.get('project_id', '')}` |
| Golden pre-staged run | `{golden_run}` |

Canonical S3 root for a run:

```text
s3://{bucket_display}/sim2real-b/<run-id>/
```

Golden Rerun recording:

```text
s3://{bucket_display}/sim2real-b/{golden_run}/reports/sim2real.rrd
```

## Three ways to run the demo

### 1. Local reference (**default for walkthrough**)

```bash
cd {repo_root}
./ops/private/sim2real-rtxpro/run-demo.sh
```

Opens local Rerun web viewer. Artifacts: `/tmp/sim2real-local/<run-id>/`

### 2. Cluster live (GPU siblings)

```bash
export KUBECONFIG=~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig
INNER_ITERATIONS=1 OUTER_ITERATIONS=2 \\
  ./ops/private/sim2real-rtxpro/submit-k8s-staged-job.sh
tmux attach -t sim2real-cluster-live
```

Images (from registry):

- Orchestrator: `{reg_display}/npa-lerobot-vlm-rl:0.1.0`
- VLM: `{reg_display}/npa-cosmos3-reason:3.0.1-genuine-sm120`
- Held-out Isaac: `{reg_display}/npa-isaac-lab:2.3.2.post1`
- Augment: `{reg_display}/npa-cosmos2-transfer:2.5.0`

### 3. Offline walkthrough (pre-staged S3 — fallback if live job slow)

```bash
./ops/private/sim2real-rtxpro/prestage-offline-run.sh {golden_run}
{repo_root}/npa/.venv/bin/rerun \\
  /tmp/sim2real-prestage/{golden_run}/reports/sim2real.rrd
```

Headless VM → browser URL:

```bash
cd {repo_root}
npa/.venv/bin/npa rerun host \\
  /tmp/sim2real-prestage/{golden_run}/reports/sim2real.rrd \\
  --allow-host-creds
```

## Visualize locally

After **local demo** (web viewer URL printed by script) or **prestage sync**:

```bash
RUN_DIR=/tmp/sim2real-local/<run-id>
{repo_root}/npa/.venv/bin/rerun "$RUN_DIR/reports/sim2real.rrd" --web-viewer
```

| `stage_14_rerun_viz` tier | `.rrd` present? |
| --- | --- |
| WORKS | Yes — open with `rerun` |
| WARN | No — install `rerun-sdk` in orchestrator |
| SEAM | No — `NPA_SIM2REAL_RERUN=0` |

## Demo scale knobs

| Knob | Local default | Cluster submit |
| --- | --- | --- |
| `INNER_ITERATIONS` | 1 | 1 |
| `OUTER_ITERATIONS` | 2 | 1–2 |
| `ROLLOUT_COUNT` | 2 | 2 |
| `HELDOUT_ENV_COUNT` | 4 | 4 |
| `NPA_ENV_COUNT` | 0 (fast) | 10000 |
| `SUCCESS_THRESHOLD` | 0.45 | 0.45 |

## Presentation checklist

1. `./ops/private/sim2real-rtxpro/prestage-offline-run.sh` — Rerun tab ready
2. Optional live submit 15–30 min before room
3. Public script: `docs/workbench/guides/sim2real-demo-script-10min.md`
4. Verify tier: `jq '.components[] | select(.name==\"stage_14_rerun_viz\")' reports/sim2real-report.json`

## Secrets (never copy here)

- `~/.npa/credentials.yaml` — S3, HF, NGC
- `~/.npa/clusters/npa-rtxpro-mk8s/kubeconfig`
"""
(out / "DEMO-WALKTHROUGH.local.md").write_text(walkthrough, encoding="utf-8")
print(f"Wrote {out}/env.local, {out}/RUNBOOK.local.md, and {out}/DEMO-WALKTHROUGH.local.md")
PY

chmod 600 "${OUT_DIR}/env.local" 2>/dev/null || true
