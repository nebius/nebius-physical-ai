# The self-hosted `npa` agent

`npa agent` deploys a **browser workbench VM** into one of your Nebius projects:
an HTTPS UI behind basic-auth login, grounded chat with a Nebius Token Factory
default (`nvidia/Cosmos3-Super-Reasoner`) or an explicitly configured
OpenAI-compatible provider, Sim Assets and Cameras panels, an
embedded [Rerun](https://www.rerun.io) viewer for `.rrd` recordings, and
draft/validate/plan/submit endpoints for `npa.workflow/v0.0.1` specs.

It is **optional**. Workflows submit and run without it; the agent is where you
go to look at what they produced.

| | |
| --- | --- |
| **Prerequisites** | Everything in [the quickstart](quickstart.md), plus Terraform 1.x, an SSH key pair, writable S3, and either a Token Factory key or an owner-only custom-provider config |
| **Typical deploy time** | ~20 minutes |
| **Teardown** | `npa agent destroy` — see [Tear it all down](teardown.md) |
| **Operator runbooks** | [npa-agent skill](../skills/tools/npa-agent/SKILL.md) · [fresh-operate loop](../skills/workflows/agent-fresh-operate/SKILL.md) |

## Deploy it

After `npa configure`, deploy interactively. There are no project or tenant ids
to type, because setup reuses the projects `configure` already saved:

```bash
npa agent preflight   # includes a cleaned writable-S3 probe before any VM work
npa agent setup       # pick a configured project → deploys the VM
npa agent status --project <alias> --name agent
```

`npa agent setup` prompts when you have more than one configured project and
deploys into the one you pick.

For scripted deploys, `npa agent fresh-setup` takes the identity explicitly:

```bash
npa agent fresh-setup --project <alias> \
  --project-id <project-id> --tenant-id <tenant-id> --region <region>
```

By default, the preflight also reserves quota for the canonical follow-on GPU
cluster so a successful VM does not make the next quickstart step impossible.
When the agent UI is intentionally the only new infrastructure, or a separately
managed cluster has already been prepared, pass `--agent-only` to check and
provision only the VM's capacity:

```bash
npa agent fresh-setup --project <alias> \
  --project-id <project-id> --tenant-id <tenant-id> --region <region> \
  --agent-only
```

`fresh-setup` provisions the VM with Terraform. `npa agent bootstrap` refreshes
only the UI/backend/nginx layer on an existing VM, without touching infra.

For one custom OpenAI-compatible provider, keep its settings outside the
checkout in a mode-`0600` JSON file. The API key stays in a separate mode-`0600`
file and is never passed on the command line:

```json
{
  "provider": "custom",
  "base_url": "https://models.example/v1",
  "api_key_file": "/owner/private/provider.key",
  "model": "provider/model",
  "models": ["provider/model"],
  "timeout_seconds": 180,
  "max_concurrency": 8
}
```

Pass only the config path to `fresh-setup`, `deploy`, or `bootstrap` with
`--llm-config-file <owner-only-json>`. NPA reads the key locally, transfers the
runtime environment through the private SFTP staging path, and installs it as
`/opt/npa-agent/llm.env` with mode `0600`; no provider secret enters Terraform,
cloud-init, a recovery command, or a shell argument. Custom-provider timeouts
must be at least 180 seconds and concurrency is bounded to eight. The owner-only
agent record retains the config-file path and non-secret settings so later
bootstraps reproduce the same provider, while systemd reloads the staged
environment after service restart or VM reboot.

When the agent's read-only identity can access a known artifact bucket but
cannot enumerate every project or bucket in the tenant, configure that exact
source as an owner default. Create a mode-`0600` JSON file outside the checkout:

```json
[
  {
    "project_id": "<artifact-project-id>",
    "bucket": "<artifact-bucket>",
    "resolved_prefix": "<directory-above-run-id>"
  }
]
```

Then refresh the existing agent with
`npa agent bootstrap --project <alias> --name <agent-name>
--artifact-source-file <owner-only-json>`. Bootstrap persists the tuple in the
owner-only NPA configuration and stages it in the service environment, so an
ordinary exact run-id search continues to use the source after a restart. The
tuple grants no access: the backend still verifies live S3 list/read capability,
and exact searches do not fall through to broader tenant discovery. Later
bootstraps reuse the persisted source without requiring the file again.
Passing a new source file explicitly replaces the saved default after a
successful bootstrap.

The `whole_path_capacity` check first reads the tenant quota aggregate. A
project-scoped administrator may be forbidden from that tenant-wide read even
though they can manage the deployment project. In that specific case, preflight
checks the project's quota allowances instead and reports `WARN`, not `FAIL`.
Finite project limits still block with exact required/used/limit/shortfall
diagnostics. Other quota-read, identity, configuration, and capacity failures
remain fail-closed. Because project allowances cannot prove remaining capacity
in the unreadable tenant aggregate, the provider may still reject the apply; the
warning says to have a tenant administrator inspect the named quota if that
happens.
## Sign in to the UI

The machine that runs `npa agent setup`, `deploy`, or `fresh-setup` stores the
UI's HTTP Basic Auth credentials in this owner-only file:

```text
~/.npa/agents/<project-alias>/<agent-name>/auth.env
```

`<project-alias>` and `<agent-name>` are the values passed to `--project` and
`--name` (`agent` is the default name). The file is created with mode `0600` and
contains `AGENT_USER` and `AGENT_PASSWORD`. It is on the **operator machine**,
not the agent VM.

Load it in the shell that will operate the agent, then ask `status` for the
verified customer URL:

```bash
PROJECT_ALIAS=<alias>
AGENT_NAME=agent
AUTH_FILE="$HOME/.npa/agents/$PROJECT_ALIAS/$AGENT_NAME/auth.env"

test -r "$AUTH_FILE" || { echo "Agent UI credential file not found" >&2; exit 1; }
source "$AUTH_FILE"
npa agent status --project "$PROJECT_ALIAS" --name "$AGENT_NAME"
```

Open the reported `public_url` and sign in with `AGENT_USER` and
`AGENT_PASSWORD`. To copy them into a browser without putting either value in
shell history, print them only when you are ready to use them:

```bash
printf 'Username: %s\n' "$AGENT_USER"
printf 'Password: %s\n' "$AGENT_PASSWORD"
```

The password output is sensitive: keep it out of logs, chat, screenshots, and
credential-bearing URLs. Do not loosen the file permissions or commit the file.
Deploy output reports the credential file path but redacts the password, and
`npa agent status` uses the file for its health probes without returning either
credential.

HTTPS uses a self-signed certificate. If a browser does not show the sign-in
flow, open `<public_url>/healthz` once to accept the certificate, then open
`<public_url>/login-help.html`. This is also the required order on phones.

`npa agent bootstrap`, including `--refresh-credentials`, reuses the existing UI
login. Replacing the deployment creates a new password, and `npa agent destroy`
removes the local agent directory, including `auth.env`. If the file for a
healthy deployment is lost, neither `status` nor the browser can reveal the
password; restore an owner-only backup or replace the agent to generate a new
login.

## What setup is doing while you wait

Setup prints **four bounded phases** around Terraform, SSH installation, and a
final probe. Phase 3 — installing agent services over SSH — can be quiet for
several minutes while packages and images are set up. Secret-free progress
heartbeats continue throughout, and the command prints an exact remote
diagnostic to run from another shell:

```bash
ssh -i <ssh-key-path> <user>@<public-ip> \
  sudo journalctl -u cloud-final -u npa-agent-backend -n 100
```

Deploy and bootstrap are **reconciled phased operations**. If a client loses the
final Terraform/SSH response, repeating the exact same command adopts a matching
healthy VM or resumes its first incomplete phase. It does not replace a healthy
VM because a response went missing.

When an agent fails before its final config record is written,
`npa agent status --project <alias> --name <name> --json` reads the operation
journal instead. It reports the typed partial state, the exact created-resource
ids with current provider evidence, and structured NPA-only resume/destroy
commands — never credentials.

> **Terraform output names.** The current outputs are `platform` / `preset` (and
> `cpu_platform` / `cpu_preset` for the CPU-only agent). The older
> `gpu_platform` / `gpu_preset` outputs are deprecated aliases kept for existing
> state, and may contain CPU values.

## How the VM authenticates

The agent VM authenticates to Nebius AI Cloud through an **attached `npa-agent`
service account** granted the tenant `editors` role. It mints short-lived IAM
tokens from the Nebius VM metadata endpoint on demand, so **no static key is
stored on the VM**.

## Optional LeIsaac UI

LeIsaac navigation and capability polling are disabled by default. Enable them
for one agent with the exact key
`projects.<project-alias>.agents.<agent-name>.ui.leisaac_enabled` in the
**operator machine's** `~/.npa/config.yaml` (`$NPA_CONFIG_DIR/config.yaml` when
that directory is configured). Merge this minimal example into the existing
project and agent record:

```yaml
projects:
  my-project:
    agents:
      agent:
        ui:
          leisaac_enabled: true
```

Only a YAML boolean `true` enables the UI. The default is `false`; an absent
key, malformed section, string such as `"true"`, or number such as `1` keeps
it disabled. Browser storage and URL parameters cannot enable it.

Apply either an enable or disable change to the existing deployment with:

```bash
npa agent bootstrap --project my-project --name agent
```

Bootstrap reads the operator config, regenerates the served HTML, and restarts
the agent services. It preserves this setting across later record updates.
Editing the config or restarting nginx/the backend alone does not regenerate
the static UI; bootstrap again, then reload already-open browser pages. Set the
key to `false` or remove it and repeat that lifecycle to hide LeIsaac again.

When enabled, the normal LeIsaac tab appears and checks readiness. Enabling its
UI does not launch a simulator or grant additional access. Existing runtime,
transport, and controller authorization still apply. See
[LeIsaac teleoperation](workbench/leisaac-teleoperation.md) for operation and
immutable episode browsing.

To verify a deployed UI, run `npm run cy:live-access` from
`npa/tests/browser`, with `NPA_AGENT_BASE_URL`, `NPA_AGENT_USER`, and
`NPA_AGENT_PASSWORD` supplied through a protected runner environment. This
checks real project/bucket interactions and expects LeIsaac hidden. Set
`NPA_AGENT_EXPECT_LEISAAC=true` when verifying an already enabled deployment;
it changes only the test expectation, not the deployment configuration. Keep
live output and screenshots in access-controlled evidence outside Git.

## What the agent can see

The agent is **tenant-aware for read-only discovery**. Its *Agent access* panel
and `GET /api/access` show the running identity's effective access, project by
project. Artifact search spans only the buckets where the agent can both
associate the bucket with a visible project and verify S3 object-list access.
Partial access is expected, and is reported rather than hidden.

Workflow submission and artifact writes/deletes stay **scoped to the deployment
project**. Caller-supplied S3 URIs remain configuration-scoped; an exact artifact
selected from a discovered cross-project run can be read without widening those
mutation boundaries.

> **This boundary is enforced by the agent application, not by a structurally
> read-only IAM credential.** Deployments may still attach a service account with
> tenant-level editors grants, so treat that credential as privileged even though
> cross-project mutation endpoints are not exposed.

### Paging run artifacts

`GET /api/artifacts/run/{run_id}` returns at most one native S3 page (up to
1,000 objects) — never the whole run. A truncated response includes
`next_cursor`; repeat the request with that cursor plus the returned
`resolved_prefix` and `bucket` (as `resource_bucket`) until `truncated=false`.
The bundled UI initially loads only the first page; **List artifacts** follows
the remaining cursors before selecting a run-wide preferred recording. Filter
and sort changes reuse the pages already loaded for that exact source. Older
consumers that assumed a complete array must migrate to cursor following:
page-local counts and `preferred` selection describe only the page you were
handed.

## Related

- [Quickstart](quickstart.md) — install, configure, credentials
- [Deploy the Physical AI Data Factory](workbench/guides/physical-ai-data-factory-deploy.md) — §3 deploys the agent as part of a full run
- [Tear it all down](teardown.md) — `npa agent destroy` and its ordering
- [CLI reference: `npa agent`](cli/agent.md)
