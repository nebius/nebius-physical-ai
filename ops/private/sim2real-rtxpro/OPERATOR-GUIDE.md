# Sim2Real — Operator Quick Start (Mac)

**Laptop = remote control.** GPUs run on Nebius mk8s. Results land on S3.

---

## Setup once

1. Install: `xcode-select --install`, `brew install kubectl awscli nebius/tap/nebius`
2. Configure: `npa configure` + `nebius mk8s cluster get-credentials --context npa-rtxpro-mk8s`
3. Copy operator env:
   ```bash
   cp ops/private/sim2real-rtxpro/sim2real-operator.env.example ~/.npa/sim2real-operator.env
   chmod 600 ~/.npa/sim2real-operator.env
   ```
4. Install run script:
   ```bash
   cp ops/private/sim2real-rtxpro/mac-run.sh ~/npa-sim2real-demo/run.sh
   chmod +x ~/npa-sim2real-demo/run.sh
   ```

---

## Every new terminal — paste this

```bash
bash <<'EOF'
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
[[ -f "$HOME/.npa/sim2real-operator.env" ]] && source "$HOME/.npa/sim2real-operator.env"
DEMO="$HOME/npa-sim2real-demo"
REPO="$DEMO/nebius-physical-ai"
if [[ ! -d "$REPO/.git" ]]; then
  git clone --branch feat/sim2real-mandatory-stages https://github.com/nebius/nebius-physical-ai.git "$REPO"
fi
(cd "$REPO" && git fetch origin feat/sim2real-mandatory-stages && git checkout feat/sim2real-mandatory-stages && git pull --ff-only origin feat/sim2real-mandatory-stages || git reset --hard origin/feat/sim2real-mandatory-stages)
cp "$REPO/ops/private/sim2real-rtxpro/mac-run.sh" "$DEMO/run.sh" && chmod +x "$DEMO/run.sh"
cd "$DEMO" && ./run.sh demo
EOF
```

---

## Three commands after submit

```bash
cd ~/npa-sim2real-demo
./run.sh status <RUN_ID>    # watch progress
./run.sh sync <RUN_ID>      # download + open Rerun
```

---

## All commands

| Command | Does |
|---------|------|
| `./run.sh demo` | Reset + submit |
| `./run.sh status <id>` | Monitor |
| `./run.sh sync <id>` | Results + Rerun |
| `./run.sh cleanup` | Clear old jobs/tmp |
| `./run.sh rehearsal` | View a past golden run (no cluster) |

---

## Cosmos versions (automatic — no action needed)

| Stage | Image | Tag |
|-------|-------|-----|
| Augment | `npa-cosmos2-transfer` | `2.5.0` |
| VLM | `npa-cosmos3-reason` | `3.0.1-genuine-sm120` |

Models: `Cosmos-Reason2-8B` + `Cosmos-Reason2-2B` (dual eval). Needs `HF_TOKEN` in cluster secrets.

---

## If something breaks

| Problem | Fix |
|---------|-----|
| `Usage: rehearsal\|trigger\|full` | Re-run paste block (updates `run.sh`) |
| `git not found` | `xcode-select --install` |
| `401 ImagePullBackOff` | Re-run `./run.sh demo` (refreshes registry auth) |
| Empty S3, job gone | Run failed — use `./run.sh status` on a **new** run |

S3 artifacts: `s3://<bucket>/sim2real-b/<RUN_ID>/`

More detail: [FRANKA-STOCK-GUIDE.md](./FRANKA-STOCK-GUIDE.md) · [CUSTOMER-DEMO.md](./CUSTOMER-DEMO.md)
