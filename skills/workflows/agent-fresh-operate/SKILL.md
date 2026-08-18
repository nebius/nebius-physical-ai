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
- If the project stanza is already gone, resume only from the opaque receipt ID
  printed before removal (`agent destroy --receipt <id> --name <name> --yes`) or
  from exact `--project-id/--instance-id` provider identity. Conflicting receipt,
  operation-journal, record, or exact identities stop before deletion; NPA never
  performs a display-name/prefix VM sweep.
- `npa/scripts/agent_fresh_setup_loop.sh` — destroy → fresh-setup → smoke chat (loop until success)
- Exact-name retries after client transport loss adopt a healthy exact VM or
  resume its first incomplete phase. Do not use `--replace` solely because the
  final Terraform/SSH response was lost; mismatched or unavailable evidence is
  indeterminate and resumable.
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

   `NPA_SSH_KEY` is the SSH **private-key path** used after provisioning. It is
   not cloud-init key content and must never be passed as
   `--ssh-public-key-path`. That option defaults to the matching
   `~/.ssh/id_ed25519.pub` and accepts exactly one OpenSSH public-key record.
   For a non-default private key, pass its existing matching `.pub` file. If it
   is absent, derive only the public record with `ssh-keygen -y -f "$NPA_SSH_KEY"`
   into an owner-controlled `.pub` file, verify the two fingerprints match, and
   pass that `.pub` path; never log either key's contents.

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
   profiles; npa retries apply without attached `service_account_id` and now emits
   a loud WARNING when it does — a VM without an attached SA cannot self-mint IAM
   tokens and needs an alternative token source (grant the deploying identity
   `compute.admin`/equivalent, or inject a token on the VM).

   **Agent VM IAM auth = attached service account (not a copied operator token).**
   The VM authenticates to Nebius IAM using its attached `npa-agent` service
   account: `get_iam_token()` self-mints fresh tokens from the metadata/token-file
   sources the SA populates. npa no longer copies the operator's short-lived IAM
   token onto the VM (no `NEBIUS_IAM_TOKEN`/`TF_VAR_iam_token` in
   `/opt/npa-agent/nebius.env`, no `/root/.npa/nebius-token`, no `agent-bootstrap`
   profile) — that token went stale and forced re-bootstrap. S3 access keys and
   the service API keys (Token Factory / HF / NGC) are still staged: object
   storage is HMAC-based and cannot use an IAM bearer token, and the product keys
   are independent of the SA. They are staged only after VM creation through the
   verified SSH channel; they never enter Terraform/cloud-init/user-data.
   On-VM Terraform (`npa cluster …`) mints a fresh
   token at run time via `nebius iam get-access-token`.

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
- **CPU destroy output.** Canonical Terraform outputs are `platform`/`preset`
  plus `cpu_platform`/`cpu_preset`. Deprecated `gpu_platform`/`gpu_preset`
  remain GPU-only machine compatibility fields (null for CPU agents) and are
  suppressed from human destroy progress, so `cpu-d3` is never presented as a
  GPU fact.
- **`fresh-setup --replace`.** Destroy must run **before** updating project env
  (otherwise TF backend keys drift mid-destroy).
- **502 / SyntaxError on chat.** Re-bootstrap; check embedded `\n` escaping in
  bootstrap `backend.py` template.
- **Ingress rules and default SGs.** Stale `allow-npa-*` rules can block a
  non-default security-group delete, so destroy removes NPA-managed ingress
  first. Nebius default security groups cannot be deleted directly; if the
  provider surfaces that specific refusal, destroy deletes the parent only when
  this agent's Terraform state proves the whole network is NPA-owned. A
  reused/shared/unproven network is preserved with an ownership explanation.
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
