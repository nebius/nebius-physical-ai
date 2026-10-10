# Optional Workbench team mode

Team mode gives a shared Kubernetes installation a small authenticated execution
boundary. Administrators create persistent local accounts, assign local groups
and explicit allocations, and operate the service. Users keep using the
existing CLI, HTTP API, or Python SDK with a personal access-key file.

This is deliberately not a standalone Workbench portal or browser-login
product. There is no self-service signup, browser session, cloud-account
provisioning, or direct user access to the private scheduler.

Each enrolled allocation has a personal worker namespace, an explicit
concurrent-GPU cap, and a separately scoped storage principal and bucket. A
workspace grant permits a role; an allocation supplies the cluster capacity and
storage scope needed to submit. Both are required for execution.

## Responsibilities and boundaries

| Role | Uses | Does not receive |
| --- | --- | --- |
| Administrator | `npa workbench team account`, policy configuration, enrollment, and service deployment | User keys, worker tokens, or a user-facing portal |
| User | A personal key file with the CLI, API, or SDK | Scheduler, Kubernetes, or administrator credentials |
| Team service | TLS gateway, durable ownership ledger, and loopback private scheduler | A public scheduler API or a per-user controller |

The gateway authorizes every request against the current local account and
policy. Users can see and operate only their own runs. A reader can inspect only
their own existing runs; a runner also needs an allocation before submitting.
Group membership by itself does not create storage, a namespace, or GPU
capacity.

The service is a trusted administrator of enrolled worker namespaces. Worker
isolation protects users from other workers; it does not isolate a compromised
gateway. The deployment and its workload boundaries still require live
qualification before production use.

## Administrator: establish local accounts and policy

Set `account_namespace` once when creating an installation. It is the stable
ownership domain for local accounts, allocations, runs, and explicitly linked
external identities. Keep it unchanged for the life of the installation.

Use private, absolute paths for state and credentials. The placeholders below
are illustrative and are not an inventory of a real installation.

```yaml
api_version: npa.team/v1
account_namespace: <generate-and-retain-one-uuid>
state_dir: /private/npa-team/state
sky_endpoint: http://127.0.0.1:46580
sky_python: /private/npa-team/sky/bin/python
clusters:
  training:
    context: training
    kubeconfig: /private/npa-team/training-kubeconfig.yaml
workspaces:
  robotics:
    grants:
      - {kind: group, value: researchers, role: runner}
      - {kind: group, value: reviewers, role: reader}
    gpu_limit: 2
    gpu_limits: {training: 2}
    allocations:
      - subject: <stable-user-id-from-account-create>
        gpu_limit: 1
        clusters: {training: 1}
        storage:
          endpoint: https://objects.example.invalid
          bucket: example-user-artifacts
          prefix: personal
          principal: example-user-storage-principal
          credentials_file: /private/npa-team/user-storage.json
```

Create users and their local groups on the service host. The `create` response
contains the generated stable `id`; record it privately and use that exact value
as the allocation `subject`. The name is a local administration label, not an
authentication credential.

```bash
npa workbench team account create --config /private/npa-team/team.yaml \
  --name researcher-a --group researchers
npa workbench team account create --config /private/npa-team/team.yaml \
  --name reviewer-a --group reviewers
npa workbench team account list --config /private/npa-team/team.yaml
```

An allocation is selected in configuration, then enrolled explicitly. Its GPU
limit is a concurrent whole-GPU cap, not a reservation, GPU-hour budget, MIG
allocation, fair queue, or cross-cluster entitlement. Per-cluster allocations
must fit both the person's and workspace's configured limits.

```bash
npa workbench team render --config /private/npa-team/team.yaml \
  --output-dir /private/npa-team/rendered
npa workbench team enroll --config /private/npa-team/team.yaml \
  --workspace robotics --cluster training \
  --subject "$WORKBENCH_USER_ID"
```

Review the rendered resources before applying them. They include personal
namespaces, quotas, network policy, admission boundaries, and worker service
accounts without Kubernetes API tokens. Each allocation needs a distinct
storage principal and bucket whose cloud IAM permits only that allocation's
scope. `shared_inputs` can name declared read-only inputs, but it cannot grant
cloud IAM access by itself.

Use `npa workbench team storage-create` only when its Nebius storage setup
matches the deployment you operate; otherwise provide an equivalently narrow
storage grant through the private configuration. Workload storage keys are
separate from personal API access keys.

## Issue and deliver a personal key file

Issue a key after the account exists. `issue-key` creates a new file and
refuses to overwrite one; its JSON response includes a credential ID and path,
not the secret. Deliver the resulting mode-0600 file privately to the user.

```bash
npa workbench team account issue-key --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --output-file /private/key-delivery/researcher-a.key
```

The service stores a digest of the key, not its plaintext. Do not place a key
in a workflow, repository, shell history, URL, or shared configuration. Users
need no Nebius account, Kubernetes credential, or scheduler credential.

## User: submit and inspect work from a key file

Each user-facing team command accepts `--token-file`. It reads one personal key
from a regular mode-0600 file and takes precedence over `--token-env`; prefer it
to putting a bearer credential in an environment variable. Set the endpoint and
keep the delivered file private.

```bash
export NPA_TEAM_ENDPOINT=https://team.example.invalid
TEAM_KEY_FILE="$HOME/.config/npa/team.key"

npa workbench team whoami --token-file "$TEAM_KEY_FILE"
npa workbench team submit --spec workflow.yaml --workspace robotics \
  --cluster training --idempotency-key researcher-a-001 \
  --token-file "$TEAM_KEY_FILE"
```

Retain the idempotency key. Retrying the same user, workspace, key, and
document returns the same run; retrying a changed document conflicts rather
than silently launching different work. Save the returned run ID, then use the
same key-backed client for status, logs, artifacts, cancellation, and recovery.

```bash
npa workbench team list --workspace robotics --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action status --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action logs --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action artifacts --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action cancel --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action resume --token-file "$TEAM_KEY_FILE"
```

`resume` reconciles a `recovery_required` run's recorded scheduler identity. If
a launch acknowledgement is missing, do not submit a replacement blindly:
retain the idempotency key and ask an administrator to reconcile the original
run. A `cancelling` run retains that intent across a service restart and cannot
be resumed. Repeat `cancel` (or the administrator's `stop-run`) to retry exact
cancellation: saved request IDs are reconciled before cancellation, but an
unidentified launch remains pending rather than being guessed or relaunched.

A run becomes terminal `failed` only when the canonical runtime records an
exact scheduler job in a terminal `FAILED` state and every recorded wave is
terminal. Its status response includes the safe fixed failure object
`{"code":"scheduler_terminal_failure",...}`; it never copies worker output
or credentials. `recovery_required` instead means at least one launch or
scheduler observation is unknown or still active, including driver failures.

The HTTP API uses the same key as a bearer credential. A raw client should read
the private file only long enough to make its request, then discard the value.
For example, a user can retrieve their own run status without receiving access
to the scheduler:

```bash
TEAM_TOKEN="$(<"$TEAM_KEY_FILE")"
curl --fail-with-body \
  --header "Authorization: Bearer $TEAM_TOKEN" \
  "$NPA_TEAM_ENDPOINT/v1/runs/$RUN_ID"
unset TEAM_TOKEN
```

The SDK uses the same `TeamClient` boundary. It does not provide a second
submission implementation or bypass authorization.

```python
from pathlib import Path

import yaml

from npa.sdk.workbench.team import SubmitRequest, TeamClient

endpoint = "https://team.example.invalid"
key = Path.home().joinpath(".config/npa/team.key").read_text().strip()
document = yaml.safe_load(Path("workflow.yaml").read_text())

client = TeamClient(endpoint, key)
try:
    result = client.submit(
        SubmitRequest(
            workspace="robotics",
            cluster="training",
            idempotency_key="researcher-a-001",
            workflow=document,
        )
    )
    print(client.run(result["id"], "status"))
finally:
    client.close()
```

## Offboarding and key lifecycle

Revoke a specific key when it is lost or replaced. Disable the account to block
all of its local keys and linked external identities. These actions take effect
for future requests and new workflow waves; they do not cancel already admitted
work or rotate workload storage keys.

```bash
npa workbench team account revoke-key --config /private/npa-team/team.yaml \
  --key-id "$WORKBENCH_KEY_ID"
npa workbench team account update --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" --disabled
npa workbench team stop-run "$RUN_ID" --config /private/npa-team/team.yaml
```

Cancel known active runs separately and revoke or rotate any static storage
credentials in the storage system. Local group changes are checked for future
requests; they do not retroactively change a run's recorded authorization
snapshot.

## Optional external identity linking

An installation may later configure an external issuer and HTTPS JWKS endpoint
to verify externally minted JWTs. This is an optional API authentication path,
not a browser-login setup: Workbench does not host an authorization-code
callback, a login page, directory synchronization, or automatic account
creation.

After the administrator verifies the provider's immutable issuer and subject,
they link it to the existing local account:

```bash
npa workbench team account link --config /private/npa-team/team.yaml \
  --user "$WORKBENCH_USER_ID" \
  --issuer "$EXTERNAL_ISSUER" --subject "$EXTERNAL_SUBJECT"
```

Unlinked external subjects are denied. An email address, display name, or
matching group never creates a link. Local groups, allocations, ownership IDs,
and storage scopes remain authoritative, so linking a credential does not move
existing runs or data. Keep a local key until the external API path has been
separately qualified.

## Deploy and qualify the service

Use the normal credential and image-security preflight before deploying. Render
the CPU gateway and private scheduler sidecar into a private manifest, keep the
scheduler loopback-only, and expose only the TLS gateway.

```bash
npa workbench team render-service --namespace workbench-system \
  --image registry.example.invalid/npa-team:reviewed \
  --secret team-config --claim team-state \
  --output-path /private/npa-team/service.yaml
```

The service needs a private configuration source, a persistent volume for its
accounts and run ledger, and a single active replica. There is no separate
server VM or per-user controller. Do not give users the rendered kubeconfig,
private scheduler endpoint, or administrator configuration.

Before production, qualify the actual deployment with personal keys: valid and
revoked keys, disabled accounts, foreign-run denial, GPU quota rejection,
storage isolation, network policy, admission controls, restart/recovery, and
TLS transport. Static workload storage credentials are scoped but are not
automatic cloud federation; key revocation does not cancel running jobs or
rotate those storage credentials.

For an offline illustration of the policy concepts, see the
[team-access example](../demos/team-access.html). It is clearly simulated,
contains no live endpoint or credential, makes no network request, and is not a
product interface or deployment proof.
