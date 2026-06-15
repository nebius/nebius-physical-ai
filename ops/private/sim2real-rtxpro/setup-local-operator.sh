#!/usr/bin/env bash
# Generate gitignored operator files from ~/.npa/config.yaml (no secrets written).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
OUT_DIR="${SCRIPT_DIR}"
PY="${ROOT}/npa/.venv/bin/python"

"${PY}" - <<'PY' "${OUT_DIR}"
import os, sys, yaml
from pathlib import Path
out = Path(sys.argv[1])
cfg = yaml.safe_load(Path.home().joinpath(".npa/config.yaml").read_text())
storage = cfg.get("storage") or {}
projects = cfg.get("projects") or {}
first_project = next(iter(projects.values()), {}) if projects else {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
registry = storage.get("registry", cfg.get("registry", ""))
k8s_context = str(storage.get("k8s_context", "") or "")
if not k8s_context:
    for proj in projects.values():
        if isinstance(proj, dict) and proj.get("k8s_context"):
            k8s_context = str(proj["k8s_context"])
            break
run_id = "sim2real-demo"
golden_run = str(storage.get("sim2real_golden_run_id", "") or "")
trigger_run = golden_run or "trigger-validate-20260611T154016Z"
env_lines = [
    f"NPA_SIM2REAL_RUN_ID={run_id}",
    f"NPA_SIM2REAL_BUCKET={bucket}",
    "NPA_SIM2REAL_PREFIX=sim2real-b",
    "NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht",
    f"NPA_SIM2REAL_TRIGGER_DATASET_URI=s3://{bucket}/sim2real-triggers/{trigger_run}/lerobot-pusht/",
    "# Leave ASSETS_URI / SCENE_SPEC_URI unset to use built-in stock Isaac tabletop scene.",
    f"AWS_ENDPOINT_URL={endpoint}",
    f"S3_ENDPOINT_URL={endpoint}",
    "NPA_SIM2REAL_SIM_BACKEND=isaac",
    "NPA_SIM2REAL_ISAAC_TASK=Isaac-Lift-Cube-Franka-v0",
    "INNER_ITERATIONS=2",
    "OUTER_ITERATIONS=2",
    "SUCCESS_THRESHOLD=0.45",
    "ROLLOUT_COUNT=8",
    "VLM_REASON2_MODEL=nvidia/Cosmos-Reason2-8B",
    "VLM_REASON3_MODEL=nvidia/Cosmos-Reason2-2B",
    "NPA_SIM2REAL_VLM_DUAL_REASON=1",
    "HELDOUT_ENV_COUNT=8",
    "NPA_ENVGEN_SHARD_COUNT=16",
    "NPA_SIM2REAL_K8S_MAX_PARALLEL_GPUS=16",
]
if k8s_context:
    env_lines.append(f"NPA_SIM2REAL_K8S_CONTEXT={k8s_context}")
if registry:
    reg = registry.rstrip("/")
    env_lines.extend([
        f"TRAINER_IMAGE={reg}/npa-lerobot-vlm-rl:0.1.0",
        f"VLM_IMAGE={reg}/npa-cosmos3-reason:3.0.1-genuine-sm120",
        f"AUGMENT_IMAGE={reg}/npa-cosmos2-transfer:2.5.0",
        f"POLICY_IMAGE={reg}/npa-sim2real-reference-policy:0.1.1",
        f"EVAL_IMAGE={reg}/npa-sim2real-eval:0.1.1-genuine-sm120",
        f"ISAAC_IMAGE={reg}/npa-isaac-lab:2.3.2.post1",
    ])
(out / "env.local").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
md = f"""# Sim2Real — Operator Runbook (generated, gitignored)

> Secrets: `~/.npa/credentials.yaml` only. **Never commit env.local or this file.**

## Cluster

- Context: `{k8s_context or "<set storage.k8s_context or projects.*.k8s_context>"}`
- Kubeconfig: `~/.npa/clusters/{k8s_context or "<context>"}/kubeconfig`

## Storage

- Bucket: `{bucket or "<configure storage.bucket>"}`
- Endpoint: `{endpoint}`
- Trigger: `s3://{bucket or "<bucket>"}/sim2real-triggers/{run_id}/lerobot-pusht/`

## Commands

```bash
export KUBECONFIG=~/.npa/clusters/{k8s_context or "<context>"}/kubeconfig
source {out}/env.local
./ops/private/sim2real-rtxpro/run-demo.sh
```

## Rerun (reuse completed run)

```bash
RUN_ID={golden_run or "<completed-run-id>"} ./ops/private/sim2real-rtxpro/run-demo.sh
```
"""
(out / "RUNBOOK.local.md").write_text(md, encoding="utf-8")

reg_display = registry.rstrip("/") if registry else "<configure storage.registry>"
bucket_display = bucket or "<configure storage.bucket>"
repo_root = out.resolve().parents[2]
golden_display = golden_run or "<completed-run-id>"
walkthrough = f"""# Sim2Real Demo — Private Walkthrough (generated, gitignored)

> From your ~/.npa/config.yaml only. **Do not commit this file.**

## Customer demo

```bash
cd {repo_root}
./ops/private/sim2real-rtxpro/run-demo.sh
RUN_ID={golden_display} ./ops/private/sim2real-rtxpro/run-demo.sh
```

---

## Infrastructure

| Item | Value |
| --- | --- |
| Cluster context | `{k8s_context or "<from config>"}` |
| Bucket | `{bucket_display}` |
| Registry | `{reg_display}` |
| Golden run (optional) | `{golden_display}` |

Canonical S3 root for a run:

```text
s3://{bucket_display}/sim2real-b/<run-id>/
```

Golden Rerun recording (when `{golden_display}` is set):

```text
s3://{bucket_display}/sim2real-b/{golden_display}/reports/sim2real.rrd
```

## Secrets (never copy here)

- `~/.npa/credentials.yaml` — S3, HF, NGC
- `~/.npa/clusters/{k8s_context or "<context>"}/kubeconfig`
"""
(out / "DEMO-WALKTHROUGH.local.md").write_text(walkthrough, encoding="utf-8")
print(f"Wrote {out}/env.local, {out}/RUNBOOK.local.md, and {out}/DEMO-WALKTHROUGH.local.md")
PY

chmod 600 "${OUT_DIR}/env.local" 2>/dev/null || true
