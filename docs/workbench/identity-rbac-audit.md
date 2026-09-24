# Identity, RBAC, and Audit Trail — Design

**Status:** design proposal for issue #524. Not implemented. Phase 0 of the
plan in §7. This document deliberately proposes no behavior change; it defines
the model that later phases build on. For the supported operating model and
access boundaries in the current implementation, see
[Team operation and access boundaries](team-operation.md).

## 1. Problem

From #524: the workbench has no multi-user story. There is no notion of teams
or roles, no per-user attribution of runs, and no audit trail of who ran what.
`npa configure` provisions per-project storage and credentials live in
`~/.npa/` with strong file permissions — a single-operator model. That blocks
team or enterprise adoption: you cannot answer "who launched this GPU run,
under which project, and who approved it," nor scope one operator's access
without handing them the whole project. Agent-driven operation makes this
worse: unattributed agent runs are unauditable runs.

## 2. Identity model

### 2.1 Identity classes

| Class | Example | Root of trust today |
|---|---|---|
| Human operator | Researcher running `npa workbench …` on a dev VM | Nebius IAM user via CLI profile (human OAuth: `skills/atomic/nebius-headless-oauth`, `skills/atomic/vm-nebius-auth`) |
| Service account | CI runner, scheduled eval | Project-scoped Nebius IAM service account (`skills/atomic/nebius-service-account-auth`) |
| Workload | `npa-agent` VM | Attached VM service account (`/mnt/cloud-metadata/token`); today the single shared `npa-agent` SA created by `npa.clients.nebius.bootstrap_agent_environment` |
| Delegated agent run | Agent acting for a human operator | On-behalf-of chain: workload identity + originating human identity |

### 2.2 Decision: both — Nebius IAM principal plus a workbench identity record

#524 asks whether the unit is a Nebius IAM principal, a workbench-managed user,
or both. Answer: **both, with a strict division of labor.**

- **Nebius IAM is the root of trust for anything that touches the cloud.** It
  already provides `viewer` / `editor` / `admin` project roles plus custom
  groups and access permits (documented in
  `skills/atomic/nebius-service-account-auth/SKILL.md`). NPA will not build a
  parallel identity provider.
- **The workbench keeps a local identity record** that binds one CLI invocation
  to a resolved principal and adds what Nebius IAM does not cover: project
  scoping *inside* a Nebius project, team membership, and NPA-specific grants
  (Token Factory key issuance, gated-artifact access approvals).

Identity record (`~/.npa/identities.yaml`, mode 0600, same private-store
discipline as `npa/clients/credentials.py`):

```yaml
identities:
  - id: "op-timothy"
    kind: human            # human | service-account | workload
    principal_id: "nebius-iam-subject-id-from-iam-whoami"
    display_name: "Timothy"
    auth_source: "nebius-profile:default"   # or "vm-metadata", "authorized-key:<name>"
    project_scope: "<nebius-project-id>"
    registered_at: "2026-09-19T00:00:00Z"
```

### 2.3 Resolution order (fail closed for writes)

1. `--as <identity-id>` / `NPA_IDENTITY` — explicit selection; always audited.
2. Attached VM identity (npa-agent workloads via instance metadata token).
3. `NPA_NEBIUS_PROFILE` or the default `nebius` CLI profile, resolved through
   `nebius iam whoami`.
4. Unresolvable → **write operations are refused** (`npa identity whoami`
   explains why); local read-only commands keep working.

`npa agent auth-profile --profile …` (`npa/cli/agent_auth.py`) remains the
human-login path; identity resolution sits on top of it, not beside it.

## 3. Run attribution

Every run/job submission is attributed to an identity in the run ledger and
surfaced in `status`/`list` output, as #524 requires.

- **Authoritative ledger:** `RunManifest`
  (`npa/src/npa/orchestration/npa_workflow/run_state.py`, schema
  `npa.workflow.run.v1`) gains a `submitted_by` block in schema v2:
  `{kind, principal_id, display_name, on_behalf_of?}`. v1 manifests keep
  working and surface as `submitted_by: unknown (pre-identity)`.
- **Client ledger:** `~/.npa/workflow-submissions/<project>/<run_id>.json`
  (`npa.workflow.submission.v1`) records the same block — its module docstring
  already names "launch identity" as a recorded side effect, so this fills an
  existing slot rather than inventing one.
- **Surfaces:** `npa workbench workflow status` and `list` gain a
  `submitted-by` column reading the manifest block.

## 4. RBAC

### 4.1 Roles

`viewer` / `operator` / `admin`, default-mapped onto Nebius IAM project roles
(`viewer`→viewer, `operator`→editor, `admin`→admin). NPA-level grants cover what
IAM cannot express: Token Factory key issuance, gated-artifact
(`access_approval.py`) approvals, and project-local admin inside a shared
Nebius project. Roles bind **per project** (project id from
`npa.clients.project_credential_store`), never globally.

### 4.2 Permission matrix

| Surface | viewer | operator | admin |
|---|---|---|---|
| `workbench workflow status` / `list` | ✅ | ✅ | ✅ |
| `workbench workflow submit` | ❌ | ✅ | ✅ |
| `workbench workflow cancel` (own runs) | ❌ | ✅ | ✅ |
| `workbench workflow cancel` (others' runs) | ❌ | ❌ | ✅ |
| `npa configure` / storage provisioning | ❌ | ❌ | ✅ |
| Project destroy (`project_destroy.py`) | ❌ | ❌ | ✅ (audited, confirm) |
| `npa agent deploy` / `destroy` | ❌ | ✅ deploy / ❌ destroy | ✅ |
| `workbench token-factory issue-key` | ❌ | ❌ | ✅ |
| Gated-artifact access approval | ❌ | request | approve ✅ |

### 4.3 Enforcement

- A `require_role(...)` gate in the CLI command layer, alongside the existing
  `intent_boundary` in `npa/lifecycle_intent.py`.
- **Phase 1: warn-only.** Denials are written to the audit log but still
  allowed — this measures blast radius before anything breaks.
- **Phase 2: enforce** on destroy / configure / token-issue / agent-destroy.
- Honest boundary: CLI-side gates are UX, not a security boundary, until the
  Phase 3 control-plane service exists. **Nebius IAM remains the real gate for
  cloud mutations throughout.**

## 5. Audit trail

Append-only log of control-plane actions
(`configure`, `submit`, `cancel`, `destroy`, `token-issue`, `approve-access`,
`provision`, `iam-change`), as #524 requires.

- **Location:** `~/.npa/audit/<project>/<YYYY-MM>.jsonl` on the operator
  machine, mirrored to `<project-bucket>/npa-audit/` for team visibility.
- **Event schema** (`npa.audit.event.v1`): `event_id` (uuid), `ts`, `actor`
  (identity record, §2.2), `action`, `target` (`{project, run_id?,
  resource?}`), `params_sha256` (secret-free — same `_SECRET_KEY` scan pattern
  already used in `submission_state.py`), `result`, `prev_hash` (hash chain
  for tamper evidence).
- **Implementation reuses `provisioning_journal.py` patterns:** fcntl-locked
  appends, no credential material, terminal events retained and mirrored into
  the teardown-receipt store.
- **Reader:** `npa audit log [--project] [--since] [--action]`.
- **Retention:** 13 months local rolling; S3 lifecycle policy on the mirror
  (Object Lock evaluated in §8).

## 6. Threat model notes

- **Shared `npa-agent` service account.** Every agent in a project runs as one
  SA (`npa/cli/agent_iam.py` documents the leftovers problem on destroy).
  Per-agent attribution is impossible until Phase 2 mints per-agent SAs and
  extends the existing IAM-cleanup path.
- **Single-operator trust root.** `~/.npa/credentials.yaml` (0600) means any
  process as the user inherits every grant. This design does not weaken the
  existing `PERMISSIONS_WARNING` discipline; it adds attribution on top.
- **Bearer secrets without per-user issuance.** `NEBIUS_TOKEN_FACTORY_KEY`
  (`npa/clients/token_factory.py`) is project-wide. Phase 2 adds
  `npa workbench token-factory issue-key --for <identity> --expires`.
- **Self-asserted identity.** Phase 1 identity records are locally asserted —
  acceptable only because Nebius IAM still gates actual cloud mutations. This
  document must not be read as claiming the CLI gate is a security boundary.
- **Agent delegation.** Agents receive scoped, expiring delegation tokens
  carrying `on_behalf_of`; they never receive the operator's OAuth cache
  or refresh credentials.

## 7. Phased implementation plan

- **Phase 0 — this document.** (Done.)
- **Phase 1 — foundation, no behavior change.** Identity resolution +
  `npa identity whoami`/`list`; `RunManifest` v2 with `submitted_by`;
  `status`/`list` columns; audit-log writer + `npa audit log`; RBAC gates in
  warn-only mode.
- **Phase 2 — enforcement.** `require_role` enforced on destroy / configure /
  token-issue / agent-destroy; per-agent service accounts; Token Factory key
  issuance; agent on-behalf-of delegation tokens.
- **Phase 3 — server.** Control-plane service with server-side authN/Z; CLI
  gates become UX hints. (There is no control-plane server today —
  `npa/server/app.py` is the LeRobot inference server, unrelated.)
- **Non-goals:** replacing Nebius IAM; SSO/SAML; cross-tenant isolation;
  billing attribution.

## 8. Open questions

1. New team members: default `viewer` or deny-by-default?
2. S3 audit mirror with Object Lock for tamper evidence — worth the
   cost/retention trade-off?
3. Break-glass admin flow: allow with mandatory post-hoc approval, or require
   pre-approval always?
4. Should warn-only mode (Phase 1) have a time-boxed expiry so it cannot
   become permanent?
