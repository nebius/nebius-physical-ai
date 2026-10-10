# Shared Workbench quickstart

An operator runs one persistent Workbench service on Nebius. Users and coding
agents send authenticated requests to its HTTPS endpoint; the service checks
permissions and runs workflows through its private SkyPilot scheduler.

| Role | Needs |
| --- | --- |
| Operator | Nebius and Kubernetes permissions, server configuration, durable storage, and TLS |
| User or coding agent | NPA client, shared HTTPS endpoint, personal key or linked Nebius identity, and an enrolled allocation |

Users do not need a Nebius account, Kubernetes credentials, VDI, a local proxy,
or port forwarding. Install the client on [macOS, Linux, or WSL2](../install.md).

The server can stay on Nebius while workflows run on other enrolled Kubernetes
clusters. See [existing clusters on other clouds](external-kubernetes.md) for
AWS EKS authentication, connection export, and destination checks.

## Operator: install once and grant access

1. Prepare the management namespace, private server configuration Secret,
   durable state PVC, and TLS Secret for the endpoint's DNS name. The selected
   cluster needs a CPU node pool and Nebius LoadBalancer support. Follow the
   [operator setup guide](team-access.md#deploy-and-qualify-the-service) to create
   a private setup file with the selected project, cluster, pinned server image,
   and resource references.
2. Run setup from a management host or CI with the required permissions:

   ```bash
   npa workbench access setup --input-path /private/npa-team/setup.yaml \
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

After installing NPA, sign in once with the endpoint and private key delivered
by your administrator:

```bash
npa login --endpoint https://team.example.invalid --token-file ./workbench.key
```

If your administrator linked your human Nebius identity instead, use
`npa login --endpoint https://team.example.invalid --nebius` and complete browser
sign-in. Running `npa login` alone guides you through missing choices.
Neither path needs a Workbench login page or Keycloak. See the
[identity guide](team-identity.md#sign-in-once-with-npa-login) for remote sign-in
and private certificates.

Both paths use the same Workbench commands for one person or a shared
installation. Add people and allocations to the existing installation as needed.
Login remembers placement when there is
one authorized workspace and cluster; otherwise select `--workspace` and
`--cluster` at login or submission.

`submit` starts workflow execution and returns a run ID. `run` takes that ID
and shows the existing run's status; `--action logs` or `--action artifacts`
retrieves its outputs. `runs` lists your runs. Set `RUN_ID` to the ID returned
by `submit`:

```bash
npa workbench whoami
# Start execution and retain the returned run ID.
npa workbench submit --spec workflow.yaml
npa workbench runs
# Inspect that same run; these commands do not start another workflow.
npa workbench run "$RUN_ID" --action status
npa workbench run "$RUN_ID" --action logs
npa workbench run "$RUN_ID" --action artifacts
```

Repeating the same submission safely recovers its original run. Add `--new-run`
when you intend another run after an acknowledged submission. `npa logout`
forgets your saved connection without cancelling jobs.

The same personal identity applies through the
[HTTP API and Python SDK](team-access.md#user-sign-in-submit-and-inspect-work).
Users can operate their own runs within the permissions granted by the operator.

## Where `npa configure` fits

`npa configure` handles operator project, credential, and storage configuration.
It does not install the shared service or create its LoadBalancer. The one-time
installation command is `npa workbench access setup`; everyday shared access uses
`npa login` followed by `npa workbench submit`, `runs`, `run`, and `whoami`.

Existing `npa workbench team` commands remain compatibility aliases. They use the
same accounts, saved connections, and submission receipts; no migration is needed.

Ordinary `npa workbench workflow` commands retain the
[operator workflow](getting-started.md). Setting `NPA_TEAM_ENDPOINT` does not
automatically route those commands through the shared service.
