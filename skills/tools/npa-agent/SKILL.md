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
npa/.venv/bin/npa agent fresh-setup --project rtxpro --name agent --project-id <project-id> --tenant-id <tenant-id> --region us-central1
npa/.venv/bin/npa agent bootstrap --project rtxpro --name agent
# Existing agents missing credentials: refresh long-lived npa-agent SA + restage VM env
npa/.venv/bin/npa agent bootstrap --project rtxpro --name agent --refresh-credentials
NPA_AGENT_CHAT_LIVE=1 npa/.venv/bin/npa agent verify-live --project rtxpro --name agent
bash npa/scripts/verify_agent_franka.sh
bash npa/scripts/verify_agent_rerun_bundle.sh
bash npa/scripts/verify_byof_onboarding_live.sh
```

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

Intent router in `npa/src/npa/cli/agent_chat.py` (embedded in remote `backend.py` at bootstrap):

| Intent | Example triggers | APIs |
|--------|------------------|------|
| `sim2real_status` | "current status", "workflow status" | sim-viz/status, workflows/sim2real/status |
| `sim_assets` | "sim assets", "selection" | sim-assets, sim-assets/selection |
| `cameras` | "cameras", "workspace camera" | sim-assets/cameras |
| `tools_catalog` | "tools", "toolRef" | tools |
| `configure_s3` | "configure S3", "bucket" | tools (nebius-infra) |
| `cosmos3` | "cosmos3", "setup cosmos" | skill steps (operator machine) |
| `load_franka` | "load franka", "show demo" | sim-viz/load-franka-demo |
| `find_artifacts` | "what can I view?", "browse artifacts" | artifacts/runs, artifacts/run/{id}, sim-viz/load-artifact |
| `onboard_solution` | "containerize github repo", "onboard workbench solution" | tools, workflows/validate, workflows/plan |
| `create_data_factory_workflow` | "create PAIDF YAML", "fan out augmented variants" | workflows/draft, workflows/validate, workflows/plan |
| `create_vlm_rl_workflow` | "create sim-to-real YAML", "VLM-RL workflow" | workflows/draft, workflows/validate, workflows/plan |

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
