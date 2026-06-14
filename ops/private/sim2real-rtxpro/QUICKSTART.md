# Sim2Real — Quick Start

Your laptop is the remote control. GPUs run on Nebius. Python runs in a **virtualenv**
(`python3 -m venv` + `pip install -e npa`) — never install `npa` with system pip.

Secrets live in `~/npa-sim2real-demo/private/` only (not in git).

Works on **Mac and Linux**.

---

## Once — first-time setup (paste)

Installs prerequisites, clones the repo into `~/npa-sim2real-demo/`, scaffolds `private/`,
creates the virtualenv. First `pip install -e npa` can take several minutes.

```bash
bash <<'EOF'
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/opt/python@3.12/libexec/bin:${HOME}/.nebius/bin:${PATH}"
BRANCH="${NPA_BRANCH:-feat/sim2real-mandatory-stages}"
DEMO="${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
REPO="${DEMO}/nebius-physical-ai"
mkdir -p "${DEMO}"
if [ ! -d "${REPO}/.git" ]; then
  git clone --depth 1 --branch "${BRANCH}" \
    https://github.com/nebius/nebius-physical-ai.git "${REPO}"
fi
bash "${REPO}/ops/private/sim2real-rtxpro/first-time-setup.sh" "${REPO}"
EOF
```

Virtualenv location (auto-created):

```text
~/npa-sim2real-demo/nebius-physical-ai/npa/.venv/
  bin/python   bin/pip   bin/npa
```

---

## Once — your credentials (edit, not paste)

Replace `YOUR-*` in these files with **your** Nebius project values:

```bash
${EDITOR:-nano} ~/npa-sim2real-demo/private/config.yaml
${EDITOR:-nano} ~/npa-sim2real-demo/private/credentials.yaml
```

Mac alternative: `open -e ~/npa-sim2real-demo/private/config.yaml`

Kubeconfig:

```bash
export PATH="${HOME}/.nebius/bin:${PATH}"
nebius mk8s cluster get-credentials --context YOUR-K8S-CONTEXT
CTX=YOUR-K8S-CONTEXT
mkdir -p ~/npa-sim2real-demo/private/clusters/$CTX
cp ~/.npa/clusters/$CTX/kubeconfig ~/npa-sim2real-demo/private/clusters/$CTX/kubeconfig
chmod 600 ~/npa-sim2real-demo/private/clusters/$CTX/kubeconfig
```

Upload stock demo trigger to **your** bucket:

```bash
cd ~/npa-sim2real-demo
./run.sh seed-trigger
```

---

## Every new terminal — run pipeline

`run.sh` syncs the repo, refreshes the virtualenv, and installs `private/` into `~/.npa/`.

```bash
cd ~/npa-sim2real-demo && ./run.sh demo
```

---

## After submit

```bash
cd ~/npa-sim2real-demo
./run.sh status <RUN_ID>
./run.sh sync <RUN_ID>
```

---

## Commands

| Command | When |
|---------|------|
| `./run.sh demo` | Submit pipeline |
| `./run.sh status <id>` | Watch stages (`npa`) |
| `./run.sh sync <id>` | Download + Rerun |
| `./run.sh seed-trigger` | Re-upload stock trigger |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `first-time-setup.sh: No such file` | Branch missing operator pack — use latest `feat/sim2real-mandatory-stages` or ask for a push |
| `python3 not found` | Mac: `brew install python@3.12` · Linux: `apt install python3 python3-venv` |
| `template placeholders` | Finish editing `private/config.yaml` + `credentials.yaml` |
| `no LeRobot batch` | `./run.sh seed-trigger` |
| Re-create venv | `bash nebius-physical-ai/ops/private/sim2real-rtxpro/bootstrap-npa-venv.sh ~/npa-sim2real-demo/nebius-physical-ai` (delete `.venv` first) |

More: [CUSTOMER-HANDOFF.md](./CUSTOMER-HANDOFF.md)
