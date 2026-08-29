---
name: npa-agent
description: Use when operating the NPA agent VM, chat UX, API grounding, bootstrap deployment, or verify-live checks.
---

# NPA Agent VM

The NPA agent is a public HTTPS workbench VM with basic-auth UI, grounded chat,
Sim Assets + Cameras panels, embedded Rerun viewer, and Sim2Real submit hooks.

## When To Use

- Deploy, bootstrap, or verify an agent VM (`npa agent …`)
- **Fresh deploy / teardown loops:** load `skills/workflows/agent-fresh-operate/SKILL.md`
- Debug chat hallucinations (raw `GET /api/…` replies) or false “Loaded Franka” messages
- Fix Rerun iframe black screen (basic auth + wasm fetch)
- Operate customer HTTPS access and sign-in UX
- **Describe this / viewer feedback:** load `skills/atomic/agent-visual-feedback/SKILL.md`

## Bootstrap And Verify

```bash
npa/.venv/bin/npa agent fresh-setup --project <alias> --name agent \
  --project-id <project-id> --tenant-id <tenant-id> --region <region> \
  --tf-var ssh_cidr_block=<operator-cidr> \
  --tf-var application_cidr_block=<operator-cidr>
npa/.venv/bin/npa agent bootstrap --project rtxpro --name agent
# Existing agents missing credentials: refresh long-lived npa-agent SA + restage VM env
npa/.venv/bin/npa agent bootstrap --project rtxpro --name agent --refresh-credentials
NPA_AGENT_CHAT_LIVE=1 npa/.venv/bin/npa agent verify-live --project rtxpro --name agent
bash npa/scripts/verify_agent_franka.sh
bash npa/scripts/verify_agent_rerun_bundle.sh
bash npa/scripts/verify_byof_onboarding_live.sh
```

### Audit the capability surface without a VM

Before claiming the agent does or does not support something, render the exact
`backend.py` that bootstrap installs, run it against a sandbox state root, and
probe it. No cluster, no VM, no Token Factory call, no quota:

```bash
npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py --json audit.json
```

It reports the registered route count, each parameterless `GET`'s real outcome,
and whether every advertised chat intent still matches and produces a grounded
reply.

Add `--serve-live` to probe a real `uvicorn backend:app` process on loopback,
started with the same argument list as the deployed `npa-agent-backend` systemd
unit, instead of driving the ASGI app in-process:

```bash
npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py --serve-live
```

Use the served tier when the question is "would this work on a VM", because it
is the only tier that exercises import-time and lifespan behavior under the real
server and the real websocket flags. It needs `uvicorn` and `websockets` in the
venv (bootstrap installs both on the VM; a repo venv may not have them). The two
offline tiers should report identical route counts and outcomes — a divergence is
itself the finding. Neither binds a public port, so both are safe on a shared
machine.

Point `--base-url` at a **deployed** agent to audit the VM itself. Routes are
then enumerated from the deployment's own `/openapi.json`, and the report diffs
that against the local render, which is how you find a VM running older code:

```bash
npa/.venv/bin/python npa/scripts/audit_agent_capabilities.py \
  --base-url https://<agent-ip>/api \
  --auth-env ~/.npa/agents/<project>/<name>/auth.env \
  --insecure --allow-mutations
```

`--insecure` is normal — the ingress uses a self-signed certificate. The
capability probes POST to `/chat`, run memory, and retrieval, so they need
`--allow-mutations` against a real deployment; route probing is read-only and
always runs. `routes_missing_on_deployment` and `routes_absent_from_render`
should both be `0`.

A deployed agent legitimately reports *better* outcomes than the sandbox, and
the difference tells you what the sandbox could not know: `/artifacts/runs`
answers `200` once S3 is staged (sandbox: `400`), `/sim-viz/rrd` and
`/sim-viz/rrd-blob` answer `200` once the stock demo recording exists (sandbox:
`404`), and the LeIsaac routes return `404 "No LeIsaac runtime is registered"`
rather than the sandbox's `403` transport refusal because real public HTTPS
satisfies that guard. Most important: chat workflow authoring emits a **runnable
YAML** on a configured deployment where the sandbox correctly declines with
unresolved placeholders — the fail-closed refusal is a configuration state, not
a capability ceiling.

Read the outcome classes rather than a pass/fail count:

- `answered` — the capability responded.
- `needs_arguments` — required query parameters missing (by design).
- `gated` — refused by a transport/auth guard (LeIsaac requires same-origin
  authenticated HTTPS, so `403` off a real deployment is correct).
- `absent_in_sandbox` — depends on staged state a sandbox has no reason to have
  (`.rrd` recordings, the LeIsaac client bundle).
- `error` — the only class that indicates a defect.

Routes that need a staged dependency say so instead of returning empty success:
`GET /api/artifacts/runs` answers `400` with "S3 discovery is not configured on
this agent" when no bucket/credentials are staged. Treat that as the honest
answer it is, not as a broken route — and never report it as "no runs found".

`npa agent deploy` provisions a dedicated long-lived **`npa-agent`** service account when
IAM allows it; otherwise bootstrap reuses existing terraform_state / saved credentials.
Persists `ssh_key_path` and non-secret deployment identity on the agent record;
storage credentials remain only in the owner-only project credential store.
Bootstrap stages `llm.env`, `s3.env`, and `nebius.env` on the VM and resolves SSH
from the agent record (or `--ssh-key` / `NPA_SSH_KEY`) — not from workbench SSH config.

Deploy/bootstrap persists pre-mutation through health-verification checkpoints
and emits secret-free structured heartbeats during long calls. After lost client
transport, reconcile the exact remote setup marker and authenticated
`/api/models`: adopt matching healthy evidence, resume incomplete phases, and
preserve ambiguous/mismatched evidence without replacing the VM.

Agent VM creation is credential-free: Terraform/cloud-init receives no S3 HMAC
keys, product tokens, or basic-auth password. After the exact VM identity and SSH
channel are verified, bootstrap stages runtime credentials with owner-only SFTP
uploads and atomic installs. A client failure resumes staging on that VM; it does
not recreate the instance or copy secrets into Terraform state/user-data.

SSH and application ingress are separate and empty by default. Agent deploy and
fresh-setup require explicit `ssh_cidr_block` and `application_cidr_block`
Terraform values because verified bootstrap and public HTTPS health checks use
those paths. Any `/0` additionally requires the matching
`allow_world_open_ssh=true` or `allow_world_open_application=true`
acknowledgement. Post-deploy reconciliation reuses the application CIDR and
never creates a world-open fallback rule.

All `npa agent …` and `nebius` IAM commands run on the **operator/dev VM**.
The **agent VM** only receives staged `/opt/npa-agent/*.env` files.
For human no-browser profile setup or recovery on a remote operator/dev VM, load
`skills/atomic/vm-nebius-auth/SKILL.md`. Do not use that flow to replace the
agent VM's attached-service-account metadata profile.

### Credential fallback (when `npa-agent` cannot be created)

Bootstrap tries in order:

1. **`npa-agent` SA** — create or reuse if IAM allows
2. **Saved operator credentials** — `~/.npa/credentials.yaml` S3 keys + optional `nebius.service_account_id`
3. **Project terraform_state keys** — `projects.<alias>.terraform_state` from the original deploy
4. **SA id discovery** — parse `lerobot-training` id from IAM errors when `agent-sa` cannot read IAM

Bootstrap persists the resolved SA id into the agent record, `credentials` block, and
`~/.npa/credentials.yaml` when discovered.

For the full BYOF live pipeline (agent + container + GPU on the configured project):

```bash
export NPA_E2E_PROJECT=rtxpro
export NPA_BYOF_LIVE_PIPELINE=1
bash npa/scripts/verify_byof_onboarding_live.sh
```

Project Kubernetes settings resolve from `~/.npa/config.yaml` (`projects.<alias>.kubernetes`)
and `~/.npa/clusters/<cluster>/kubeconfig` — not from any operator VM hostname.

For real BYOF container build/push/inspect, set `NPA_BYOF_LIVE_CONTAINER=1` and run
`bash npa/scripts/verify_byof_onboarding_live.sh` on a host with Docker and
explicit credentials for the operator-controlled registry. Default validation
repo is LeIsaac; override with `NPA_BYOF_REPO_URL` / `NPA_BYOF_REPO_REF`.

For full BYOF GPU smoke (SkyPilot submit), also set `NPA_BYOF_LIVE_GPU=1` and run
the same script on a host with Docker, `nebius`, `sky`, and any required explicit registry pull
access. GPU train YAML and SkyPilot config resolve from the project `kubernetes`
block (`gpu_profile: rtxpro`, `byof_train_yaml`, `skypilot_config`).

Auth secrets live at `~/.npa/agents/<project>/<name>/auth.env` (`AGENT_USER`, `AGENT_PASSWORD`).
Agent bootstrap now stages operator config + credentials on the VM at `~/.npa/{config,credentials}.yaml` so the VM can run infra commands without re-entering project metadata. Bootstrap also installs Nebius CLI (if missing) and seeds a `cursor-sa` profile backed by `/mnt/cloud-metadata/token` when the VM has attached SA metadata; if token-backed profile setup is present but unusable, bootstrap fails fast instead of silently skipping it.
Token Factory model selection is configurable via `--llm-model` and `--llm-models` (`NPA_AGENT_LLM_MODEL` and `NPA_AGENT_LLM_MODELS` on the VM), with `/api/models` exposed for UI/model picker refresh.

## Customer HTTPS Access

- Public URL: `https://<public_ip>/` (self-signed cert on VM IP)
- `npa agent status` may print that canonical HTTPS endpoint in local operator
  output or an explicitly requested handoff only after its authenticated probe
  succeeds and an unauthenticated request returns `401`, proving HTTP Basic Auth
  is enforced. The status payload records `endpoint_disclosure_allowed=true` and
  `basic_auth_enforced=true` when this narrow exception applies.
- This exception never covers `direct_url`, credential-bearing URLs, usernames,
  passwords, auth-file contents, or an endpoint whose protection was not just
  verified. Keep those values in the owner-only `0600` credential store.
- Sign-in form at `/login-help.html` and `/welcome` (mobile-safe XHR/fetch sign-in; URL-embed fallback on desktop only)
- On phones: open `/healthz` first to accept the self-signed certificate, then sign in at `/login-help.html`
- Mobile chat uses `sessionStorage` basic-auth fallback — sign out by clearing site data or use `/login-help.html` again
- All `fetch` calls use `credentials: "include"` for session basic auth
- Never suggest `localhost`, `127.0.0.1`, or port `8080` — use same-origin `/api/…` paths

## Chat Maturity Patterns

Typed GPU placement failures and consented preemptible fallback use
`skills/atomic/gpu-allocation-fallback/SKILL.md` and the grounded
`/api/agent/gpu-allocation/*` routes.

Intent router in `npa/src/npa/cli/agent_chat.py` (embedded in remote `backend.py` at bootstrap).

All 35 routed intents are listed below with a trigger phrase verified to match
(`npa/scripts/audit_agent_capabilities.py` exercises exactly these). Do not
assume an unlisted capability is missing without re-running that audit, and do
not add a rule for one of these without checking which existing intent already
claims the phrasing — earlier rules win.

**Run / viewer state**

| Intent | Verified trigger | APIs |
|--------|------------------|------|
| `sim2real_status` | "what is the current sim2real status" | sim-viz/status, workflows/sim2real/status |
| `watch_sim` | "watch the sim until the blob and iframe both report success" | sim-viz/status, sim-viz/rrd, sim-viz/rrd-blob |
| `start_sim2real` | "start the sim2real pipeline" | workflows/sim2real/submit |
| `drive_sim2real` | "autonomously drive the sim2real outer loop" | agent/sim2real/drive, workflows/sim2real/{submit,status} |
| `load_franka` | "load the franka demo" | sim-viz/load-franka-demo, sim-viz/status |
| `list_recordings` | "list the available recordings" | sim-viz/recordings, sim-viz/runs |
| `find_artifacts` | "what can I view?" | artifacts/runs, artifacts/run/{run_id}, sim-viz/load-artifact |
| `foxglove_viewer` | "open foxglove" | foxglove/status, foxglove/config, foxglove/load-artifact |
| `sim_assets` | "show me the sim assets" | sim-assets, sim-assets/selection |
| `cameras` | "which cameras are selected" | sim-assets/cameras |

**Workflow authoring** — all four templates share
`workflows/draft`, `workflows/validate`, `workflows/plan`. Picking the wrong one
is the common failure, so match the qualifier, not just the word "workflow":

| Intent | Verified trigger | Picked when |
|--------|------------------|-------------|
| `create_vlm_rl_workflow` | "create a sim-to-real workflow yaml" | sim-to-real authoring, **or** any "quality gate" / "policy rollout" / "heldout eval" / "vlm critic" phrasing |
| `create_gate_workflow` | "create a token factory gate workflow" | Token Factory / Cosmos scene-reasoning gate |
| `create_loop_gate_workflow` | "create a sim2real workflow with a loop gate" | explicit "loop gate" / "decision gate" |
| `create_data_factory_workflow` | "create a PAIDF workflow yaml" | PAIDF / video augmentation / scenario fan-out |
| `create_rl_policy_workflow` | "create an RL policy training workflow" | RL policy training |
| `create_workflow` | "create a 2-step sim2real npa.workflow" | explicit two-step, or generic `npa.workflow` |
| `workflow_execute_guidance` | "how do I actually run this workflow" | validate/plan/submit + tools |

`create_vlm_rl_workflow` is matched **before** `create_gate_workflow` and claims
"quality gate", so reach the Token Factory gate through its own wording.

**Infrastructure**

| Intent | Verified trigger | APIs |
|--------|------------------|------|
| `infra_backends` | "which infra backends are available" | infra/k8s, infra/provision, workflows/submit |
| `mk8s_provision` | "provision an mk8s cluster" | infra/mk8s, infra/mk8s/provision, infra/k8s |
| `live_infra_loop` | "run the live infra loop" | infra/k8s, infra/provision, workflows/*, tools |
| `soperator` | "deploy a slurm cluster" | infra/soperator/{validate,deploy,status/{name}} |
| `tenant_resources` | "what tenant resources do I have" | resources |
| `configure_s3` | "configure S3 bucket access" | tools (nebius-infra) |
| `onboard_solution` | "containerize this github repo as a workbench solution" | tools, workflows/validate, workflows/plan |
| `cosmos3` | "set up cosmos3" | tools (skill steps run on the operator machine) |

**Tool capability questions** — all answer from `tools`. "what can `<tool>` do"
is a verified trigger for each of `cosmos`, `lancedb`, `sonic`, `lerobot`,
`groot`, `genesis`, `mjlab`, `isaac lab` (`*_capabilities`), and
`component_capabilities` answers the generic "what components are available".

Every routed intent must have an `INTENT_APIS` entry. That map is not just reply
metadata: `_semantic_route` derives the semantic fallthrough's `known_intents`
from its keys, so an intent absent from it can never be reached by a paraphrase
the regex misses. `test_every_intent_declares_its_apis` enforces this.

**BYOF onboarding:** load `skills/workflows/byof-onboard/SKILL.md` (source of truth for base profiles, workloads, live verify). Chat replies reference this skill path — do not paste the full procedure into `agent_chat.py`.

Rules:

- **Never** return only `GET /api/…` as the chat reply
- Always unpack **run_id**, **stage**, **rerun_ready**, **camera** in markdown (`**key**: \`value\``)
- Grounded replies set `"grounded": true` and `"apis_used": ["sim-viz/status", …]`
- LLM fallback injects `format_live_context_block(state)` JSON snapshot into the system prompt
- Workflow drafting should pick a template by **intent + workflow capabilities** (sim2real loop-gate, VLM-RL loop, tokenfactory-cosmos gate, or simple two-step), not by hardcoded endpoint-only replies.
- PAIDF and sim-to-real drafting resolves the staged agent bucket and configured
  Kubernetes accelerator/profile before rendering. A conflicting requested GPU
  fails closed; absent infrastructure remains an explicit placeholder/warning.
- Chat-generated sim-to-real loads the single canonical compositional
  `npa.workflow/v0.0.1` graph. It must not emit the retired
  `workbench.sim2real.run` monolith or legacy demo/echo toolRefs.

## Workflow Draft / Validate / Plan / Submit Loop

These workflow operations are provider-neutral: the caller owns its model or
reasoning configuration, while NPA owns validation, translation, and execution.
The complete bounded lifecycle is documented in
`docs/workbench/agent-workflow-operations.md`.

Use the VM as a grounded drafting surface, then run operator-machine commands for real workflow execution:

```bash
# Agent VM draft surface
GET  /api/workflows/draft
POST /api/workflows/draft
POST /api/workflows/validate
POST /api/workflows/plan
POST /api/workflows/submit
```

```bash
# Operator machine (authoritative execution path)
npa/.venv/bin/npa workbench workflow validate-spec <spec.yaml> --json
npa/.venv/bin/npa workbench workflow plan-spec <spec.yaml> --run-id <run_id> --json
npa/.venv/bin/npa workbench workflow run-spec <spec.yaml> --plan-only --scheduler-plan --json
```

`POST /api/workflows/draft` **saves** a draft (it requires a `yaml` body and
returns validation + plan); it does not generate one. Chat is the generator.

Chat emits `workflow_yaml` **only after validation and planning both succeed**.
Without a staged bucket and configured accelerator it returns
`Could not generate runnable workflow YAML yet.` and names the unresolved
placeholders (`<configure-s3-bucket>`, `<configure-gpu-accelerator>`). That is
the fail-closed contract working, not a defect — do not "fix" it by relaxing the
gate, and do not report a placeholder refusal as a broken drafting path. It also
means workflow authoring cannot be fully exercised on an agent with no staged
infrastructure; `npa agent verify-live` asserts the YAML branch, so run it
against a bootstrapped VM.

Guidance:

- Keep config grouped: runtime knobs first, then `*_uri` keys under prefix paths.
- For multi-step specs, include explicit state descriptions, resources, inputs/outputs schemas, loop/gate transitions, and terminal leaves.
- If transitions exist, plan with `--assume-decision promote_checkpoint|loop_back`.

## Rerun Iframe Fix

Rerun wasm inside `/rerun/?url=…` cannot send HTTP basic auth and does not
reliably consume parent-created `blob:` URLs across browsers. Bootstrap publishes
the active recording to unauthenticated `/rerun/recordings/sim2real.rrd` for the
iframe, while the parent page still validates authenticated access by fetching
`/api/sim-viz/rrd-blob`.

Use this order:

1. Publish/copy the `.rrd` to `/opt/npa-agent/recordings/sim2real.rrd`.
2. Point iframe `url=` at same-origin `/rerun/recordings/sim2real.rrd?t=...`.
3. Also `fetch("/api/sim-viz/rrd-blob", { credentials: "include" })` as the
   authenticated blob/bytes health gate and fallback.

Do not point the iframe directly at `/api/sim-viz/rrd` (black screen / auth
failure in browser contexts).

Submitted Sim2Real runs must not reuse the stock Franka/demo `.rrd` as if it
were run-specific data. `POST /api/workflows/sim2real/submit` and chat requests
such as "start/run the Sim2Real pipeline" should launch the agent-local Sim2Real
runner, update the standalone Run status/logs panel from
`/api/workflows/sim2real/status` and `/api/workflows/sim2real/runs/{run_id}`,
and only mark Rerun ready after a real run recording URI is present.
Run-specific recordings should open on a useful 3D scene overview (world/table,
world/franka/*, world/cube) plus rollout/signal panels, not only sparse rollout
or held-out image streams.

## HTTP API Reference

All paths are under `/api/` (nginx proxies to FastAPI backend on `:8787`).

### `GET /api/health`

```json
{
  "ok": true,
  "tool_refs": 19,
  "capabilities": {
    "gpu_allocation_fallback": {
      "status": "available",
      "grounded": true,
      "routes": [
        "POST /api/agent/gpu-allocation/attempt",
        "POST /api/agent/gpu-allocation/consent"
      ]
    }
  }
}
```

The GPU allocation routes are embedded-backend capabilities, not workbench
`toolRef`s. `attempt` accepts typed placement evidence and returns a zero-token
decision; `consent` declines without consuming another action's confirmation or
accepts only the exact single-use, action-digest-bound confirmation token.

### `GET /api/session`

```json
{
  "selection": {"robot_preset": "franka", "sim_backend": "isaac", "scene_spec_uri": "stock://scene/default"},
  "sim_viz": {"run_id": "franka-demo", "stage": "demo", "camera": "workspace", "rerun_ready": true},
  "latest_submit": {},
  "camera_selection": ["workspace"],
  "chat_history": []
}
```

### `POST /api/chat`

Request: `{"messages": [{"role": "user", "content": "what is the current sim2real status"}]}`

Grounded response:

```json
{
  "ok": true,
  "model": "nvidia/Cosmos3-Super-Reasoner",
  "reply": "**Sim2Real status** … **run_id**: `franka-demo` …",
  "grounded": true,
  "apis_used": ["sim-viz/status", "workflows/sim2real/status"]
}
```

### `GET /api/sim-viz/status`

```json
{
  "run_id": "franka-demo",
  "stage": "demo",
  "camera": "workspace",
  "rrd_uri": "file:///opt/npa-agent/sim2real.rrd",
  "rerun_ready": true,
  "rerun_iframe_url": "/rerun/?url=…"
}
```

### `POST /api/sim-viz/load-franka-demo`

Body: `{"camera": "workspace"}` → generates `.rrd`, restarts Rerun service, returns `sim_viz`.

### Artifact-first discovery + load

- `GET /api/artifacts/runs?prefix=&limit=100` discovers artifact-backed run
  prefixes from every bounded native S3 page in every project bucket with
  verified effective list access. The Agent access panel's
  **List artifacts** / **Browse / preview** actions add the selected
  `project_id` and `resource_bucket`; the backend verifies that pair against
  effective access before searching and returns the selected provenance. Follow
  `next_cursor` until it is empty; `pagination_complete=false` means a bounded
  source scan or access scope was incomplete. Category/state/source-cache roots
  are not runs. `q` filters the cached bounded discovery index in process.
  `total_runs` is present only when that source index is complete; otherwise it
  is null and `observed_run_count`, `observed_match_count`, `query_complete`,
  and `total_runs_scope=unavailable` describe the bounded observation without
  presenting it as a global total. Lightweight rows preserve
  `summary_complete=false` and unknown viewability/count fields until enriched.
- `GET /api/artifacts/run/{run_id}` returns an S3-native artifact page with
  `render` hints. The UI follows every opaque `next_cursor` with the returned
  `run_ref`, `project_id`, `resolved_prefix`, and `bucket` (as
  `resource_bucket`), merges/deduplicates the pages, then computes the global
  preferred recording. A page-1 video therefore cannot auto-open while a
  later-page RRD/MCAP is still undiscovered. Repeated cursors, incomplete pages,
  source changes, cancellation, and authorization failures stop selection
  rather than leaving a partial page presented as the complete run.
- `POST /api/sim-viz/load-artifact` loads only a discovered inventory object. Send
  the server-issued `run_id`, `run_ref`, `project_id`, `resource_bucket`,
  `resolved_prefix`, `source_selected=true`, and exact `key`. `s3_uri` may be
  displayed as provenance but is not a browser authorization selector.
  URI-only requests return versioned error `npa.agent.api_error/v1` with code
  `run_id_required_for_s3_uri`; this deliberately prevents arbitrary S3 reads.
- Unknown types are still listed and selectable (`render="download"` fallback).

Exact lookup returns `409 ambiguous_run_id` when the same ID has multiple
`(project_id, bucket, resolved_prefix)` sources, `404 run_not_discovered` only
after complete effective-scope discovery, and an access/incomplete error when a
source could not be searched. Select the returned source explicitly rather than
accepting an arbitrary first match.

The search field is exclusively for discovered NPA workflow/artifact runs.
Directories under `/home/ubuntu/codex-runs/...` identify Codex maintenance jobs
on an operator machine; they are not NPA run IDs and are never published into
customer artifact storage merely to make them searchable.

`artifacts/run` returns exactly one native S3 page per request, capped at 1,000
objects. Its `count`, `artifacts`, and `preferred` fields are page-local. Clients
must follow every `next_cursor` with the same exact source tuple before selecting
a viewer; cursors are opaque and stable only for the S3 listing they came from.
The shipped UI does this automatically and labels the merged page count. A run
that changes while pages are being followed inherits native S3 listing
consistency and requires a fresh first-page load when source identity or cursor
continuity changes.

`GET|HEAD /api/artifacts/content` and `/api/artifacts/download` require that same
exact source tuple plus `key`. MP4 GET/HEAD/Range responses stream the object with
truthful `Content-Type`, `Content-Length`, `Accept-Ranges`, and `Content-Range`;
the UI HEAD-checks those facts and surfaces JSON/authorization errors before it
creates a video element.

### Stage evidence contract

`GET /api/workflows/sim2real/runs/{run_id}` and the matching status endpoint
return `npa.stage-evidence/v1` rows. Pass the discovered `project_id`,
`resource_bucket`, and `resolved_prefix` when selecting a cross-project run;
stage-detail requests preserve the same verified scope.

- Explicit workflow manifests, durable status records, reports, and agent event
  state may establish `Succeeded`, `Failed`, `Running`, `Skipped`, `Pending`, or
  `Not run`, with authoritative provenance.
- Artifact presence establishes only `Observed output`; missing output establishes
  neither an attempt nor an outcome. `Not run` requires an authoritative graph or
  status record that says so.
- Artifact-only runs show only observed logical groups and use summaries such as
  `6 observed groups · execution status unavailable`. Never attach the canonical
  Sim2Real graph to an unrelated run or report `N/M succeeded` without a grounded
  denominator and explicit success evidence.
- Unknown artifact layouts remain visible. The local Franka fixture is labeled as
  demo evidence and must be cleared before another run is rendered. Browser loads
  abort and generation-check stale detail requests so a prior graph cannot return.
- Each row exposes status label, evidence type/source, authority/confidence,
  diagnostic reason, timestamps, and artifact count. Inline artifact JSON is
  recursively credential-redacted before it is returned to the UI.

### `GET /api/access`

Returns the non-secret effective access report used by artifact discovery and
the UI's **Agent access** panel:

```json
{
  "apiVersion": "npa.agent.access/v1",
  "identity": {
    "tenant_id": "<tenant-id>",
    "deployment_project_id": "<project-id>",
    "deployment_project_name": "<project-alias>",
    "service_account_id": "<service-account-id>",
    "credential_source": "instance_metadata",
    "credential_profile": "cursor-sa",
    "credential_config": "/root/.nebius/config.yaml"
  },
  "status": "partial",
  "scope": "partial_tenant",
  "projects": []
}
```

Effective access is evaluated in layers: list projects visible under the tenant,
list object-storage resources separately for each project, then verify S3 list
and read access without writing. A denied/unavailable project remains in the
report and does not hide accessible projects. If tenant/project listing is not
permitted, the configured deployment project and bucket may remain usable, but
a tenant-configured agent reports `partial_tenant`; it never silently calls that
fallback healthy `single_project` scope. Use
`GET /api/access?refresh=true` after IAM changes.
`npa agent verify-live` validates this schema and the matching UI wiring on a
bootstrapped VM.

The VM service account therefore needs tenant project-list visibility,
per-project `storage bucket list`, and S3 `ListBucket`/`GetObject` for projects
that should be searchable. `npa agent deploy` continues to request the existing
tenant editors-group membership when the operator can manage IAM; when that is
not possible, bootstrap reuses available credentials and the access report shows
their actual narrower reach.

Bootstrap creates and verifies the root `cursor-sa` profile against the exact
attached service-account ID, scrubs ambient/static IAM-token variables for
inventory commands, and verifies tenant project listing before calling a
tenant-configured deployment successful. Short-lived bootstrap IAM tokens are
not staged into the backend systemd environment.

Tenant-wide access is read-only at the agent product boundary. This is enforced
by the application, not by structurally read-only IAM credentials: the attached
service account may still hold tenant-level editors-group grants and must be
handled as privileged. Workflow submission and artifact writes remain scoped by
the application to the configured home/deployment project; artifact deletion is
not exposed. An
arbitrary caller-supplied S3 URI is still limited to configured buckets. The
only cross-project exception is an exact object key selected from a requested
discovered run; it is verified against effective bucket access before loading.

### `GET /api/sim-viz/rrd-blob`

Authenticated octet-stream of `.rrd` bytes (for parent blob URL).

### `GET /api/sim-assets`

Scene/robot specs + current selection and `resolved_uris`.

### `GET /api/sim-assets/cameras`

```json
{"cameras": [{"name": "workspace", "placement": "stock_workspace", "fov": 60.0}], "selected": ["workspace"]}
```

### `POST /api/workflows/sim2real/submit`

Submits workflow with current selection; updates `latest_submit` and `sim_viz.run_id`.

### Run History Quick Switching

- `GET /api/sim-viz/runs` lists indexed run snapshots.
- `GET /api/sim-viz/recordings` lists available `.rrd` recordings.
- `POST /api/sim-viz/load-run` or `POST /api/sim-viz/select-run` switches active run quickly.

### `GET /api/tools`

```json
{"tool_refs": ["workbench.genesis.train", "..."]}
```

## Source Layout

- CLI + bootstrap: `npa/src/npa/cli/agent.py`
- Chat router (testable): `npa/src/npa/cli/agent_chat.py`
- Franka verify script: `npa/scripts/verify_agent_franka.sh`
- Fresh deploy loop: `npa/scripts/agent_fresh_setup_loop.sh` (see `skills/workflows/agent-fresh-operate/SKILL.md`)
- Mature deploy loop: `npa/scripts/agent_mature_verify_loop.sh`

## Security / Guardrails

- Never leak credentials, auth env, or opaque secrets into chat or workflow YAML.
- A canonical agent `https://<public_ip>/` is shareable only under the verified
  Basic Auth endpoint exception above; do not infer protection from a saved URL.
- Use same-origin HTTPS paths (`/api/...`) for browser actions; avoid localhost guidance.
- Do not hardcode project IDs, tenant IDs, bucket names, registry IDs, usernames, or public IPs in examples.
