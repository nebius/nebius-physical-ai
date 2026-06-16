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
    "NPA_SIM2REAL_SIM_BACKEND=genesis",
    "INNER_ITERATIONS=2",
    "OUTER_ITERATIONS=2",
    "SUCCESS_THRESHOLD=0.45",
    "ROLLOUT_COUNT=3",
    "HELDOUT_ENV_COUNT=8",
    "NPA_SIM2REAL_K8S_CONTEXT=npa-rtxpro-mk8s",
]
if registry:
    reg = registry.rstrip("/")
    from npa.deploy.images import supported_tool_version
    trainer_tag = supported_tool_version("lerobot-vlm-rl")
    eval_tag = supported_tool_version("sim2real-eval")
    vlm_tag = supported_tool_version("cosmos3-reason")
    env_lines.extend([
        f"TRAINER_IMAGE={reg}/npa-lerobot-vlm-rl:{trainer_tag}",
        f"VLM_IMAGE={reg}/npa-cosmos3-reason:{vlm_tag}",
        f"EVAL_IMAGE={reg}/npa-sim2real-eval:{eval_tag}",
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

```bash
# After download from s3://{bucket}/sim2real-b/<run-id>/reports/sim2real.rrd
pip install rerun-sdk
rerun sim2real.rrd
```

## Validated local accuracy delta (staged reference mode)

| Run | Inner iters | Success rate | Reward trend | Decision |
| --- | --- | --- | --- | --- |
| baseline | 1 | 0.0 | [0.29] | loop back |
| trained | 2 | 1.0 | [0.29, 0.72] | promote |

Reproduce: `/tmp/run_staged_twice.sh` on nebius-dev-vm (see PR validation log).
"""
(out / "RUNBOOK.local.md").write_text(md, encoding="utf-8")
print(f"Wrote {out}/env.local and {out}/RUNBOOK.local.md")
PY

chmod 600 "${OUT_DIR}/env.local" 2>/dev/null || true
