# Shared Workbench quickstart

An operator runs one persistent Workbench service on Nebius. Users and coding
agents send authenticated requests to its HTTPS endpoint; the service checks
permissions and runs workflows through its private SkyPilot scheduler.

| Role | Needs |
| --- | --- |
| Operator | Nebius and Kubernetes permissions, server configuration, durable storage, and TLS |
| User or coding agent | NPA client, shared HTTPS endpoint, personal key, and an enrolled allocation |

Users do not need a Nebius account, Kubernetes credentials, VDI, a local proxy,
or port forwarding. Install the client on [macOS, Linux, or WSL2](../install.md).

## Operator: install once and grant access

1. Prepare the management namespace, private server configuration Secret,
   durable state PVC, and TLS Secret for the endpoint's DNS name. The selected
   cluster needs a CPU node pool and Nebius LoadBalancer support. Follow the
   [operator setup guide](team-access.md#deploy-and-qualify-the-service) to create
   a private setup file with the selected project, cluster, pinned server image,
   and resource references.
2. Run setup from a management host or CI with the required permissions:

   ```bash
   npa workbench team setup --input-path /private/npa-team/setup.yaml \
     --output-path /private/npa-team/installation.json
   ```

   Setup deploys or adopts the service, creates or reuses its HTTPS LoadBalancer,
   and records the retained address. Point the endpoint's DNS name to that
   address. Setup reports `ready` after verifying TLS, DNS, and authentication
   boundaries. Retry with the same input and receipt; keep both outside Git.
3. [Create accounts, assign permissions and scoped storage, and enroll each
   allocation](team-access.md#administrator-establish-local-accounts-and-policy).
   [Issue a personal key](team-access.md#issue-and-deliver-a-personal-key-file)
   and privately deliver its mode-0600 file, the HTTPS endpoint, workspace name,
   and cluster alias to each user.

## User or coding agent: connect, submit, inspect

After installing NPA, keep the delivered key in a private file and replace the
example endpoint, workspace, and cluster alias with the operator's values.

```bash
export NPA_TEAM_ENDPOINT=https://team.example.invalid
TEAM_KEY_FILE="$HOME/.config/npa/team.key"

npa workbench team whoami --token-file "$TEAM_KEY_FILE"
npa workbench team submit --spec workflow.yaml --workspace robotics \
  --cluster training --idempotency-key robotics-run-001 \
  --token-file "$TEAM_KEY_FILE"
```

Keep the returned run ID and reuse the same idempotency key when retrying the
same submission. A changed workflow needs a new key. Set `RUN_ID` to the returned
ID before inspecting the run:

```bash
npa workbench team run "$RUN_ID" --action status --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action logs --token-file "$TEAM_KEY_FILE"
npa workbench team run "$RUN_ID" --action artifacts --token-file "$TEAM_KEY_FILE"
```

The same personal identity applies through the
[HTTP API and Python SDK](team-access.md#user-submit-and-inspect-work-from-a-key-file).
Users can operate their own runs within the permissions granted by the operator.

## Where `npa configure` fits

`npa configure` handles operator project, credential, and storage configuration.
It does not install the shared service or create its LoadBalancer. The one-time
installation command is `npa workbench team setup`; everyday shared access uses
`npa workbench team` commands with the endpoint and personal key.

Ordinary `npa workbench workflow` commands retain the
[operator workflow](getting-started.md). Setting `NPA_TEAM_ENDPOINT` does not
automatically route those commands through the shared service.
