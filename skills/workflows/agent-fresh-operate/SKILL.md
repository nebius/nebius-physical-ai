---
name: agent-fresh-operate
description: Use when deploying, tearing down, or reproducing a fresh NPA agent VM from scratch — npa-driven destroy/fresh-setup, profile selection, tiered verify gates, and teardown failure recovery.
---

# Agent Fresh Operate

## When To Use

Use this skill to **operate** a clean agent VM lifecycle on the **operator/dev VM**:

- First-time `fresh-setup` on a project alias
- Teardown → redeploy loops (“reproduce from scratch”)
- Validate `/api/models` and `/api/chat` after deploy
- Debug destroy/fresh-setup failures (IAM, orphan VMs, ingress rules)

For chat UX, API shapes, and Rerun iframe behavior, use `npa-agent`. For
`npa configure` / object-storage provisioning, use `nebius-infra`.

## Entry Points

- `npa/.venv/bin/npa agent fresh-setup` — initialize project env + deploy + bootstrap
- `npa/.venv/bin/npa agent destroy` — npa-driven teardown (ingress cleanup, TF destroy, orphan VM delete)
- `npa/scripts/agent_fresh_setup_loop.sh` — destroy → fresh-setup → smoke chat (loop until success)
- `npa/scripts/agent_mature_verify_loop.sh` — bootstrap-first mature loop (existing agents; not fresh deploy)

All `npa agent …` and `nebius` commands run on the **operator/dev VM** with
`~/.npa/config.yaml` and `~/.npa/credentials.yaml`. Cloud agents sync the
target branch to the dev VM before live tests.

## Procedure

1. **Preconditions (dev VM).**
   ```bash
   cd ~/nebius-physical-ai
   git checkout <branch> && npa/.venv/bin/pip install -e npa -q
   nebius profile activate "${NPA_NEBIUS_PROFILE:-npa-mk8s}"
   export NPA_NEBIUS_PROFILE="${NPA_NEBIUS_PROFILE:-npa-mk8s}"
   export NPA_SSH_KEY="${NPA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
   ```

2. **Teardown (npa-driven — no manual `nebius vpc` edits).**
   ```bash
   npa/.venv/bin/npa agent destroy --project <alias> --name agent
   ```

3. **Fresh deploy.**
   ```bash
   npa/.venv/bin/npa agent fresh-setup \
     --project <alias> --name agent \
     --project-id <project-id> \
     --tenant-id <tenant-id> \
     --region us-central1
   ```
   Expect **compute PermissionDenied with VM SA attachment** on some cross-project
   profiles; npa retries apply without attached `service_account_id`.

4. **Smoke gate (default “done” for fresh deploy).**
   ```bash
   source ~/.npa/agents/<alias>/agent/auth.env
   BASE="$(npa/.venv/bin/npa agent status --project <alias> --name agent --json \
     | npa/.venv/bin/python -c 'import json,sys; print(json.load(sys.stdin).get("public_url","").rstrip("/"))')"
   curl -sk -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/api/models"
   curl -sk -u "${AGENT_USER}:${AGENT_PASSWORD}" -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}]}' \
     "${BASE}/api/chat"
   ```

5. **Optional full gates.**
   - Grounded chat: ask “what is the current sim2real status” → `"grounded": true`
   - Live regression: `NPA_AGENT_CHAT_LIVE=1 npa/.venv/bin/npa agent verify-live --project <alias> --name agent`
   - Mature loop: `bash npa/scripts/agent_mature_verify_loop.sh` (bootstrap-first)

6. **One-command loop.**
   ```bash
   export NPA_AGENT_PROJECT=<alias> NPA_AGENT_NAME=agent
   export NPA_AGENT_PROJECT_ID=<project-id> NPA_AGENT_TENANT_ID=<tenant-id>
   export NPA_AGENT_REGION=us-central1 NPA_NEBIUS_PROFILE=npa-mk8s
   bash npa/scripts/agent_fresh_setup_loop.sh
   ```

## Verify Tiers

| Tier | Checks | Use when |
|------|--------|----------|
| **Smoke** | `status --json`, `/api/models`, hello `/api/chat` | Fresh deploy validated |
| **Grounded** | sim2real status chat → `grounded: true` | Chat router wired |
| **Live** | `verify-live` | Pre-merge regression |
| **Mature** | `agent_mature_verify_loop.sh` + Franka | Chat/router code changes |

Do not block a smoke deploy on `verify-live` UI wiring markers alone.

## Gotchas

- **Profile vs project.** Cross-project deploy needs a profile with compute IAM on
  the target project (commonly `npa-mk8s`). `cursor-sa` may lack VPC/compute on
  foreign projects. Never use `tle` in scripts (interactive auth hang).
- **Compute PermissionDenied + SA.** First TF apply may fail attaching `npa-agent`
  SA to the VM; npa retries without SA attachment. Bare compute denial → stop and
  report IAM gap to operator.
- **Destroy: disk/SG in use.** Orphan cloud VM may exist outside TF state after a
  failed apply/rollback. `npa agent destroy` deletes matching instances by name
  before TF destroy; retry destroy if preconditions fail once.
- **`fresh-setup --replace`.** Destroy must run **before** updating project env
  (otherwise TF backend keys drift mid-destroy).
- **502 / SyntaxError on chat.** Re-bootstrap; check embedded `\n` escaping in
  bootstrap `backend.py` template.
- **Ingress rules.** Stale `allow-npa-*` rules can block security-group delete;
  destroy path removes npa-managed ingress first.
- **Cloud agent → dev VM.** Sync branch (`git pull` or tar/scp), confirm
  `npa agent --help` lists `fresh-setup`, then run live loop on dev VM.

## Symptom → Action

| Symptom | Action |
|---------|--------|
| `PermissionDenied: service compute` then success after retry message | Expected SA-attachment retry; no action |
| `PermissionDenied: service compute` on retry without SA | Operator IAM on target project |
| `Agent config not found` after destroy | Run `fresh-setup` |
| Destroy fails, instance name `agent-<alias>-agent` still listed | Re-run `npa agent destroy` (orphan cleanup) |
| Chat 502, health false | `npa agent bootstrap --project <alias> --name agent` |
| `verify-live` UI version mismatch but chat OK | Smoke tier passed; fix UI marker separately |

## Verify (repo)

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
bash -n npa/scripts/agent_fresh_setup_loop.sh
# Smoke-only against an existing agent (no destroy/deploy):
NPA_FRESH_SETUP_SKIP_DESTROY=1 NPA_FRESH_SETUP_SKIP_DEPLOY=1 \
  NPA_AGENT_PROJECT=<alias> bash npa/scripts/agent_fresh_setup_loop.sh
```
