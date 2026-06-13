# Sim2Real — Local Demo Walkthrough

**100% local.** No Kubernetes cluster, no S3, no Nebius credentials.

## One command (after clone)

```bash
git clone --branch feat/sim2real-mandatory-stages \
  https://github.com/nebius/nebius-physical-ai.git ~/nebius-physical-ai
cd ~/nebius-physical-ai
./ops/private/sim2real-rtxpro/run-local-demo.sh
```

The script **creates `npa/.venv` on first run** if missing, installs `npa`, runs the
pipeline, and starts a local Rerun web viewer.

**Open the URL printed at the end** in your browser.

## Manual setup (optional)

```bash
cd ~/nebius-physical-ai
python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install -U pip
npa/.venv/bin/python -m pip install -e npa
./ops/private/sim2real-rtxpro/run-local-demo.sh
```

Requires **Python 3.10+** (`python3 --version`).

## What to show in Rerun (~30 s)

| Entity | Stage |
| --- | --- |
| `rollouts/.../camera` | 7 Action rollouts |
| critique overlays | 8 VLM critique |
| `signal/reward` | 9 RL signal |
| `heldout/scores` | 10 Held-out eval |

## Options

```bash
VISUALIZE=0 ./ops/private/sim2real-rtxpro/run-local-demo.sh
OPEN_RERUN=1 ./ops/private/sim2real-rtxpro/run-local-demo.sh   # native app (macOS)
MODE=staged ./ops/private/sim2real-rtxpro/run-local-demo.sh
```

## Troubleshooting

| Error | Fix |
| --- | --- |
| `no such file: run-local-demo.sh` | `git pull` — need latest `feat/sim2real-mandatory-stages` |
| `python3 not found` | Install Python 3.10+ (Homebrew: `brew install python@3.11`) |
| `rerun-sdk missing` | Re-run script (bootstraps venv) or `npa/.venv/bin/pip install -e npa` |

Private walkthrough: https://github.com/timothy-le7/npa-sim2real-demo-walkthrough
