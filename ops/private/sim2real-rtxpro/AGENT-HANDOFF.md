# Agent Handoff — Sim2Real Mac Operator Pack

**Read this first in a new chat.** Then execute the verification checklist below.

---

## Goal

Mac operators and customers run the **staged sim2real pipeline** on Nebius mk8s with:

- **Laptop = remote control** (submit, monitor, sync, Rerun)
- **Secrets only on machine** at `~/npa-sim2real-demo/private/` → synced to `~/.npa/` at runtime
- **Python via virtualenv** — `python3 -m venv` + `pip install -e npa` — never global pip
- **Public repo** = scripts + `private/*.example` templates only (no credentials, no infra IDs)

Branch: `feat/sim2real-mandatory-stages`

---

## Canonical docs (keep these in sync)

| Doc | Audience | Purpose |
| --- | --- | --- |
| **[MAC-QUICKSTART.md](./MAC-QUICKSTART.md)** | Mac operator / customer | **Primary** — 3-step paste blocks |
| [CUSTOMER-HANDOFF.md](./CUSTOMER-HANDOFF.md) | Customer onboarding | Public vs private, security |
| [OPERATOR-GUIDE.md](./OPERATOR-GUIDE.md) | Internal | Short index → MAC-QUICKSTART |
| [README.md](./README.md) | Index | Entry point |

---

## Canonical scripts

| Script | When |
| --- | --- |
| `mac-first-time.sh` | Once: brew tools, clone, `private/` scaffold, venv |
| `bootstrap-npa-venv.sh` | Creates `npa/.venv` with pip |
| `setup-customer-demo.sh` | Scaffold `~/npa-sim2real-demo/private/` from `*.example` |
| `seed-stock-trigger.sh` | Upload lerobot/pusht to customer bucket |
| `FRESH-SETUP-AND-RUN.sh` | Session: sync repo, venv, `./run.sh demo` |
| `~/npa-sim2real-demo/run.sh` | Daily: `demo`, `status`, `sync`, `seed-trigger` |
| `verify-paste-demo.sh` | CI-style check (no cluster submit) |

---

## Layout

```text
~/npa-sim2real-demo/
  run.sh                              # from mac-run.sh
  private/                            # NEVER in public git
    config.yaml
    credentials.yaml
    operator.env
    clusters/<k8s-context>/kubeconfig
  nebius-physical-ai/                 # public clone
    npa/.venv/                        # virtualenv (pip install -e npa)
    ops/private/sim2real-rtxpro/      # operator pack (in repo)
```

---

## Mac flow (must stay this simple)

### Once — first time

See [MAC-QUICKSTART.md](./MAC-QUICKSTART.md) — one paste block installs tools, clones repo, creates venv with pip.

### Once — credentials

Edit `~/npa-sim2real-demo/private/config.yaml`, `credentials.yaml`, kubeconfig.
Then: `cd ~/npa-sim2real-demo && ./run.sh seed-trigger`

### Every terminal

```bash
cd ~/npa-sim2real-demo && ./run.sh demo|status|sync
```

(`run.sh` syncs repo + venv each session; no separate paste block needed.)

---

## Platform code (already on branch)

- `npa workbench workflow status <run-id>` routes sim2real runs (not distill)
- Registry auth refresh before K8s submit (`registry_auth.py`)
- Rerun modes: local / Nebius public IP (`RERUN_HOST=nebius`) / share (`RERUN_MODE=share`)

---

## Agent task — TEST AND RECHECK

**Do this every time setup docs or scripts change.**

### 1. Doc consistency

- [ ] [MAC-QUICKSTART.md](./MAC-QUICKSTART.md) is the single source for paste blocks
- [ ] No hardcoded bucket, registry ID, cluster name, or operator credentials in tracked files
- [ ] All other guides link to MAC-QUICKSTART (not duplicate paste blocks)
- [ ] `private/*.example` use only `YOUR-*` placeholders

### 2. Script smoke test (no cluster)

```bash
bash ops/private/sim2real-rtxpro/verify-paste-demo.sh
```

### 3. Manual command walkthrough

Run each step on a clean temp `HOME` or document blockers:

1. `mac-first-time.sh` — creates demo dir, venv, templates
2. Fill minimal test `private/` (see `verify-paste-demo.sh` for fixture shape)
3. `bootstrap-npa-venv.sh` — idempotent, prints venv path
4. `./run.sh seed-trigger` — needs real S3 creds (skip in CI)
5. `./run.sh demo` — needs real cluster (skip in CI)
6. `./run.sh status <id>` — uses `npa workbench workflow status`
7. `./run.sh sync <id>` — S3 sync + Rerun

### 4. Simplicity rules

- Prefer **one paste block** per phase (once / daily)
- Prefer **`cd ~/npa-sim2real-demo && ./run.sh <cmd>`** over long heredocs for daily use
- Always mention **virtualenv path** when documenting Python/npa
- Never add operator infra to public markdown

### 5. Git hygiene

- Real secrets: `~/npa-sim2real-demo/private/` or gitignored `*.local.md`
- `.gitignore` blocks `ops/private/**/private/*` except `*.example`

---

## Known gaps / follow-ups

- `install-prereqs.sh` requires Homebrew on Mac (document if missing)
- First `pip install -e npa` can take several minutes — note in MAC-QUICKSTART
- Customer must accept HF gated models + cluster `hf-ngc-tokens` secret for Cosmos stages
- Linux customers: no `mac-first-time.sh`; use CUSTOMER-HANDOFF manual path

---

## Paste into new chat

```
Load ops/private/sim2real-rtxpro/AGENT-HANDOFF.md and MAC-QUICKSTART.md.

Task: Test and recheck all Mac setup command steps for maximum simplicity.
- Run verify-paste-demo.sh
- Ensure MAC-QUICKSTART.md is the single canonical paste-block doc
- Remove duplicate/conflicting instructions in other md files
- Confirm virtualenv+pip flow (bootstrap-npa-venv.sh) is documented and works
- No credentials or infra IDs in public repo
- Keep daily flow to: cd ~/npa-sim2real-demo && ./run.sh demo|status|sync

Branch: feat/sim2real-mandatory-stages
```
