---
name: cosmos3-env-troubleshoot
description: Use when Cosmos3 setup, fetch, inference, CUDA, uv, Docker, Hugging Face, GitHub, NGC, or checkpoint staging fails in NPA or in an upstream Cosmos framework checkout.
---

# Cosmos3 Environment Troubleshooting

## Source And Attribution

Adapted from NVIDIA cosmos-framework
`skills/atomic/cosmos3-env-troubleshoot/SKILL.md`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Used under OpenMDW-1.1.
See `skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1` and
`skills/NOTICE-NVIDIA-COSMOS3`.

## When To Use

Use this skill for import errors, missing Python packages, CUDA or torch
failures, Docker GPU runtime problems, failed source clones, Hugging Face 401s,
NGC credential errors, and inference runtime tracebacks. For "where is this
file or config" questions, use
`skills/atomic/cosmos3-codebase-nav/SKILL.md`.

## First Rule

Never print or commit secrets. Token-bearing values include `GITHUB_TOKEN`,
`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `NGC_API_KEY`, AWS keys, and any env var
selected through `NPA_COSMOS3_*_TOKEN_ENV`.

## NPA Diagnostic Ladder

1. Confirm the NPA command surface:

   ```bash
   npa/.venv/bin/npa workbench cosmos --help
   ```

   `check` and `fetch` should be present. `skill` and `skills` should not be
   present.

2. Run a redacted access check:

   ```bash
   npa/.venv/bin/npa workbench cosmos check --output json
   ```

   Interpret statuses, not secrets. Expected status labels are `configured`,
   `missing`, `reachable`, `failed`, or `skipped`.

3. For source-only failure isolation:

   ```bash
   npa/.venv/bin/npa workbench cosmos fetch --skip-checkpoint --output json
   ```

4. For inference workflow issues, inspect:

   ```bash
   npa/.venv/bin/python - <<'PY'
   import yaml
   from pathlib import Path
   p = Path("npa/src/npa/workflows/skypilot/cosmos3-text-to-image-inference.yaml")
   doc = yaml.safe_load(p.read_text())
   print(doc["name"])
   print(doc["envs"]["NPA_COSMOS3_NO_GUARDRAILS"] == "")
   PY
   ```

5. If a SkyPilot run fails on the GPU node, collect logs and env status without
   dumping token values:

   ```bash
   printenv | rg '^(NPA_COSMOS3|COSMOS3|HF_HOME|LD_LIBRARY_PATH)='
   python --version
   which python
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

## Common Error Signatures

| Error | Likely cause | Fix |
| --- | --- | --- |
| `Hugging Face auth missing` | `HF_TOKEN` or configured HF env var is unset | Set token and accept the model license upstream before fetch |
| HF 401 or gated repo denied | Token lacks accepted license or repo access | Accept terms for the model and retry with the same token |
| `git ls-remote` fails | Source URL wrong or GitHub auth missing for private fork | Check `NPA_COSMOS3_SOURCE_REPO` and selected GitHub token env var |
| `ModuleNotFoundError: cosmos_framework` | Upstream checkout was not installed | Run `uv sync --all-extras --group=cu130-train` from the upstream checkout |
| PyTorch `_functionalization` import error | NGC container library path conflict | Run `export LD_LIBRARY_PATH=` before Python imports |
| CUDA shared library error | CUDA major version mismatch | Align torch CUDA version with host driver CUDA support |
| Docker `runtime name: nvidia` error | Docker NVIDIA runtime not configured | Run `sudo nvidia-ctk runtime configure --runtime=docker` on the host |

## Upstream Remediation

In an upstream Cosmos framework checkout:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --all-extras --group=cu130-train --reinstall
source .venv/bin/activate
export LD_LIBRARY_PATH=
python -c "import cosmos_framework; print('cosmos_framework import ok')"
```

Use `cu128-train` if the driver stack requires older CUDA. Use the inference-only
group only when training dependencies are intentionally not needed.

## Bug Report Template

When the failure remains unresolved, give the user a concise report with:

- NPA commit and branch.
- Exact NPA command or SkyPilot workflow used.
- Redacted `NPA_COSMOS3_*` values.
- OS, Python, torch, and CUDA versions.
- Whether `check` passed and whether `fetch --skip-checkpoint` passed.
- Full traceback with secrets removed.
- Whether guardrails were left on or explicitly disabled.
