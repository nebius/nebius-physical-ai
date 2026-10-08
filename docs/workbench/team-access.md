# Optional Workbench team mode

Team mode adds an authenticated gateway for shared Kubernetes clusters. Existing
identity-provider users and groups receive workspace grants; each allocated
person has worker/controller namespaces, an explicit GPU quota, and a dedicated
storage principal and bucket. Every stage stays inside its run's allocation.

This implementation requires live deployment qualification before production.
Local operator commands retain their existing behavior. Team users must use the
gateway and must not receive its credentials or access to its private SkyPilot
API. Cluster administrators remain trusted administrators.

## Identity, projects, and configuration

The gateway verifies JWT signatures, issuer, audience, subject and expiration
against an administrator-configured HTTPS JWKS endpoint. It requires `iss`,
`aud`, `sub`, `iat`, and `exp`. Groups must be a list of strings. RS256 and ES256
are supported. Proxy email headers and request-body identities confer no access.
Obtain a token for the Workbench audience through the organization's existing
login flow; this version has no separate login screen or password directory.

A workspace can span clusters in different projects or clouds. Users need
workspace access, not a Nebius account. Membership in one workspace grants
nothing in another. A compatible Nebius issuer can be configured after verifying
its tokens and claims; Nebius IAM roles and SAML assertions are not automatically
translated. There is no automatic Nebius membership synchronization.

Keep configuration and credentials private, and use absolute file paths. Example GPU values below are
illustrative administrator choices, not defaults.

```yaml
api_version: npa.team/v1
identity:
  issuer: https://identity.example.com
  audience: workbench
  jwks_url: https://identity.example.com/.well-known/jwks.json
  groups_claim: groups
  algorithms: [RS256]
state_dir: /state/team
sky_endpoint: http://127.0.0.1:46581
sky_python: /opt/sky/bin/python
disabled_subjects: []
clusters:
  training:
    context: training
    kubeconfig: /etc/npa-team/training.yaml
    api_server_url: https://kubernetes.example.com
    api_server_cidr: 192.0.2.10/32
    api_server_port: 443
workspaces:
  robotics:
    grants:
      - {kind: group, value: robotics-researchers, role: runner}
    gpu_limit: 2
    gpu_limits: {training: 2}
    allocations:
      - subject: example-external-subject
        gpu_limit: 2
        clusters: {training: 2}
        storage:
          endpoint: https://objects.example.com
          bucket: example-personal-artifacts
          prefix: personal
          credentials_file: /etc/npa-team/person-storage.json
          principal: example-personal-storage-principal
        shared_inputs: []
```

`grants` select external `subject` or `group` values. Roles `reader`, `runner`,
and `admin` are cumulative; run visibility remains personal for all three.
Membership does not mint credentials or assign capacity: administrators add the
person's allocation. GPU values are integers, including zero for CPU-only access.
Personal cluster allocations must fit the person's `gpu_limit`; their workspace
sums must fit `gpu_limits`, whose sum must fit the workspace `gpu_limit`.
These are concurrent whole-NVIDIA-GPU caps, not reservations, GPU-hour budgets,
MIG allocations, or fair queues.
Reapply enrollment after quota changes; existing use can outlast a quota reduction.

`api_server_url` must be reachable from controllers and match their TLS trust.
`api_server_cidr` identifies the exact address controllers dial, retaining the
URL hostname for TLS verification. Account for the CNI's
treatment of service translation. Default network policies allow paired
namespaces, cluster DNS, public HTTP/HTTPS, and the controller's explicit API
address. Private registries, private object endpoints, nonstandard DNS labels,
or other private services require reviewed policy changes.

Each allocation requires a distinct storage principal and bucket, with cloud IAM
allowing only its bucket and explicit shared read-only inputs. Credential files
are mode `0600` JSON with `aws_access_key_id`, `aws_secret_access_key`, and optional
`aws_session_token`. `shared_inputs` lists full S3 prefixes; it permits declared
workflow inputs but does not grant cloud IAM permissions. Optional
`workload_secrets_file` references private JSON containing only `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, `NGC_API_KEY`, or `NVIDIA_API_KEY`. Secrets are assigned
explicitly and never inherited from the operator's environment.

## Provision and deploy

Perform the normal administrator credential preflight. For a personal Nebius
storage allocation, run:

```bash
npa workbench team storage-create --project research \
  --endpoint https://objects.example.com --output-dir /private/person-storage
```

This creates a fresh account, bucket, dedicated capability group and bucket-scoped
`storage.object-editor` permit. It never uses the ordinary shared project storage
group or falls back to tenant-wide editors. Add the generated storage fragment to
the allocation. Partial provisioning retains a private recovery receipt without
automatically deleting resources or data.

```bash
npa workbench team render --config /private/team.yaml --output-dir /private/team-rendered
npa workbench team enroll --config /private/team.yaml \
  --workspace robotics --cluster training --subject example-external-subject
npa workbench team export-kubeconfig --config /private/team.yaml \
  --output-path /private/sky-kubeconfig.yaml
```

Review rendered resources before enrollment. They include namespaces, fixed
service accounts, namespaced controller RBAC, read-only node discovery, quotas,
network policies, and validating admission rules. Workers receive no Kubernetes
token and cannot select another account. Controllers cannot write cluster-wide
RBAC. Submissions verify installed resources and selected RBAC denials; missing
controls reject new work. Admission support and an enforcing CNI are required.
These checks do not establish that the CNI actually enforces the network policy.

Open the [interactive HTML example](../demos/team-access.html) locally to try
user switching, workspace and cluster selection, GPU limits, run recovery,
private artifacts, and offboarding. Its infrastructure is explicitly simulated;
the evidence tab contains recorded checks from the actual local API and policy.
Rebuild those receipts without cloud credentials using
`npa/.venv/bin/python npa/scripts/build_team_demo.py --output-path /private/team-access.html`.
The example's browser regression runs in `npm run cy:mock` from
`npa/tests/browser/`. With its dependencies and Chrome installed, run only this
example using `node --test team_demo.test.cjs` from that directory. It checks
the desktop interactions, mobile layout, and absence of outgoing HTTP requests.

The exported kubeconfig is **server-only**. Controllers instead use their own
projected, rotating Kubernetes account token. Never upload operator credentials
into workload images or shared artifact buckets.

Build `npa/docker/team-server/Dockerfile` privately from the repository root using
the normal image security gates. Select a cluster-compatible kubectl image with
the `KUBECTL_IMAGE` build argument. Render deployment resources:

```bash
npa workbench team render-service --namespace workbench-system \
  --image private-registry.example.com/team-server:reviewed \
  --secret workbench-team-config --claim workbench-team-state \
  --output-path /private/team-service.yaml
```

Supply a dedicated management namespace outside the execution namespaces, a
Secret, a ReadWriteOnce PVC, and HTTPS ingress. The Secret contains `team.yaml`,
`sky-server.yaml`, exported `sky-kubeconfig.yaml`, each enrolled kubeconfig, and
referenced credential files. Configure their paths under `/etc/npa-team/`.
An init container copies them into private files; roll out after Secret changes.

One pod runs two CPU containers: the gateway and pinned SkyPilot 0.12.2, listening
on loopback. Only the gateway has a Service. The PVC retains both ledgers and
authoritative runtime state. One replica, `Recreate`, and a process lock prevent
duplicate supervisors. No separate VM is required; active-active mode is absent.

An existing private Linux host can also run the services. Install NPA, uvicorn
and kubectl; install `skypilot[kubernetes]==0.12.2` in a separate environment.
Start `/opt/sky/bin/python -m sky.server.server --host 127.0.0.1 --port 46581`
with `IS_SKYPILOT_SERVER=true`, `SKYPILOT_DISABLE_USAGE_COLLECTION=1`, a private
`HOME`, the rendered `SKYPILOT_GLOBAL_CONFIG`, and exported `KUBECONFIG`. Then
run `npa workbench team serve --config /private/team.yaml` behind HTTPS. The
gateway defaults to `127.0.0.1:8443`; `--host` and `--port` change its listener.
It clears inherited cloud credentials and source-overlay settings.

## Submit, recover, and offboard

```bash
export NPA_TEAM_ENDPOINT=https://workbench.example.com
# Supply NPA_TEAM_TOKEN through the organization's approved login flow.
npa workbench team submit --spec workflow.yaml --workspace robotics \
  --cluster training --idempotency-key my-first-run
npa workbench team run "$RUN_ID" --action status
```

`--endpoint` overrides `NPA_TEAM_ENDPOINT`. `--token-env` changes the default
`NPA_TEAM_TOKEN` variable. Retain the idempotency key for retries: identical
submissions return the same run, while changed documents conflict. `run --action`
also supports `logs`, `artifacts`, `cancel`, and `resume`; `team list --workspace`
lists personal runs. Commands emit JSON. The SDK exports `TeamClient` and
`SubmitRequest` through `npa.sdk.workbench.team`, including streamed downloads.
Agents and browser integrations must use these routes, not the scheduler API.

Ownership and launch intents live in the private server ledger. S3 holds artifacts
and readable runtime-record copies; those copies never authorize access or decide
whether a job was already launched. Restarted runs become `recovery_required`.
Resume reconciles exact scheduler IDs. Missing launch acknowledgements never
trigger blind resubmission. Cancellation remains `cancelling` until observed
provider state proves termination.

Repeat cancellation to recheck a nonterminal result. Internal SkyPilot workspaces
only select placement; the gateway owns authorization. They deliberately avoid
SkyPilot's startup-only private-user registration policy. The scheduler's default
role is `user`, controller consolidation is disabled, and its non-default
loopback port prevents the SDK from auto-starting an unmanaged replacement server.

Remove grants or add `disabled_subjects` to block access. Already-issued JWT group
claims remain valid until expiration; use the denylist for immediate local
revocation. An admitted workflow retains its verified group snapshot for its
execution; each new wave still checks current local grants and the denylist.
An identity-provider group change alone does not revoke that admitted workflow.
Jobs are not automatically destroyed. A local operator with access
to the private server configuration can cancel after offboarding:

```bash
npa workbench team stop-run "$RUN_ID" --config /private/team.yaml
```

Arbitrary pod configuration, server file mounts, public SkyPilot services, and
workflow changes to cloud/context are rejected. All stages use the run namespace.
Automatic cloud-token exchange and refresh are not implemented; rotate scoped
storage credentials through the operator's existing process. The controller's
Kubernetes token rotates normally. Removing Workbench access does not revoke
cloud keys already issued to jobs: stop affected runs, revoke their recorded
access keys through cloud IAM, and replace the private credential file before
granting access again. The provisioning receipt records the exact key and account.

Before production, qualify two real users: CPU/GPU jobs, quota rejection,
cross-user status/log/artifact denial, cross-bucket read/write denial, rejected
identity/token overrides, network isolation, restart/reconciliation, and
offboarding/cancellation. The container, admission expressions, CNI, cloud IAM,
OIDC provider, and actual SkyPilot controller launch require live evidence.
Offline tests and schema validation alone do not prove production isolation.

Set `RUN_ID` to the server-issued ID returned by submission. For opt-in admission,
quota and storage checks, set `NPA_INTEGRATION_E2E=1`, `NPA_TEAM_LIVE_E2E=1`, and
`NPA_TEAM_LIVE_CONFIG` to a private configuration containing two allocated people,
then run `npa/.venv/bin/python -m pytest npa/tests/e2e/test_team_access_live.py`.
The tests use Kubernetes server dry-run and create/remove isolated S3 probe keys.
The separate `NPA_TEAM_SKY_PYTHON` test setting selects an isolated 0.12.2
interpreter for `npa/tests/workbench/team/test_sky_compatibility.py`; that test
uses a real local API with synthetic identities and no cluster credentials.
