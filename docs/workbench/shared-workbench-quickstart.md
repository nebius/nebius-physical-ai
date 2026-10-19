# Workbench setup and usage

Workbench gives people and coding agents one authenticated way to submit
workflows and retrieve their results. An operator hosts its HTTPS service and
private SkyPilot scheduler on Nebius Kubernetes. Users connect with the NPA
CLI or Python SDK; SkyPilot executes the workloads.

The same installation works for one person or a group. Adding colleagues means
adding accounts, permissions, and allocations to that installation.

Joining an existing service? Go to [sign in once](#user-sign-in-once).
Installing or administering the service? Start with [operator setup](#operator-prepare-one-installation).

| Role | Needs |
| --- | --- |
| Operator | Access to the hosting cluster and each destination, private server configuration, durable storage, and TLS |
| User or coding agent | NPA client, Workbench HTTPS endpoint, personal key or linked Nebius human identity, and workspace permission; an allocation is also needed to submit |

The personal-key path requires no end-user Nebius account. Install the client on
[macOS, Linux, or WSL2](../install.md). Users connect directly over HTTPS; they
need neither cluster credentials nor a VDI or local proxy.

## Operator: prepare one installation

Setup is operator work. `npa workbench access setup` deploys the server and
endpoint using prepared prerequisites; it does not create the cluster, PVC,
configuration, TLS certificate, personal accounts, or storage grants for you.

1. **Choose the host and destinations.** Use an existing Nebius Kubernetes
   cluster with CPU management nodes, or provision one through the
   [Kubernetes setup guide](kubernetes.md). Destinations may be in other
   projects or clouds. Ensure the server can reach each Kubernetes API, image
   registry, and allocated object store.
2. **Prepare durable identity and policy.** Prepare the state PVC and an
   operator environment that can mount it. Select one permanent
   `account_namespace`. Create local accounts, workspace grants, per-user GPU
   allocations, and scoped storage using the
   [account and policy guide](team-access.md#administrator-establish-local-accounts-and-policy).
   Account commands operate on the server's authoritative database: run them
   in an operator environment with that persistent state mounted. A separate
   database on a laptop will not create users in the deployed service.
3. **Enroll the destinations.** Review and apply the namespace, quota, admission,
   and worker-identity resources. Prepare restricted runtime credentials
   separately from enrollment credentials, then
   [export the server connections](external-kubernetes.md#administrator-setup).
   Keep the server's `team.yaml`, `sky-server.yaml`, kubeconfigs, and scoped
   storage credentials in its private configuration Secret.
4. **Prepare the remaining hosting resources.** Use the management namespace,
   state PVC, configuration Secret, and TLS Secret for the endpoint's DNS name.
   Preserve the account state prepared above. The server runs as one active
   replica with a private SkyPilot sidecar; it does not require another VM.
5. **Deploy and qualify the endpoint.** Follow the
   [operator setup contract](team-access.md#deploy-and-qualify-the-service) to
   prepare a private setup file with the selected project, cluster, reviewed
   server image, and resource references, then run:

   ```bash
   npa workbench access setup --input-path /private/workbench/setup.yaml \
     --output-path /private/workbench/installation.json
   ```

   Setup deploys or adopts the service, creates or reuses its HTTPS
   LoadBalancer, and records the retained address. Point DNS to that address.
   If setup reports `endpoint-awaiting-dns` or `endpoint-awaiting-connectivity`,
   correct the reported condition and retry with the same input and receipt.
   `ready` means the endpoint checks passed; verify the intended person's
   workload and artifact access before onboarding them.
6. **Deliver access.** [Issue a personal key](team-access.md#issue-and-deliver-a-personal-key-file)
   or [link a Nebius human identity](team-identity.md#native-nebius-human-identity).
   Privately give the user the HTTPS endpoint, their key if applicable,
   workspace and cluster aliases, and an approved workflow. For a private CA,
   provide its certificate too.

The server copies its configuration Secret at startup. After adding allocations
or destinations, update and re-export the policy, regenerate `sky-server.yaml`
with `access render`, update the existing Secret, and arrange a controlled
restart. Keep the account namespace and state PVC. Membership/key operations
on the authoritative account database are checked on subsequent requests.

## User: sign in once

Choose the method your administrator prepared:

### Personal key

This path needs no end-user Nebius account or SSO provider:

```bash
npa login --endpoint https://workbench.example.invalid --token-file ./workbench.key
```

### Nebius human sign-in

Use this after your administrator links your verified Nebius identity to the
Workbench account. Complete the official browser sign-in:

```bash
npa login --endpoint https://workbench.example.invalid --nebius
```

Running `npa login` alone guides you through missing choices. For a private
certificate, add the administrator-provided `--ca-file`. Neither method needs
a Workbench login page or Keycloak. See the
[identity guide](team-identity.md#sign-in-once-with-npa-login) for remote sign-in.

Verify what you can use:

```bash
npa workbench whoami
```

Login remembers placement when there is one authorized workspace and cluster.
If there are several, select `--workspace` and `--cluster` at login or submission.
For multiple installations, save a separate `--profile` at login and select that
profile on subsequent commands. Each installation controls its own membership.

## User: submit, list, and inspect

These are the current command names. `run` refers to one existing execution;
`runs` is the list of executions.

| Task | Command | Result |
| --- | --- | --- |
| Start workflow execution | `npa workbench submit --spec workflow.yaml` | A run ID; execution continues on the server |
| List your executions | `npa workbench runs` | Your visible runs in the selected workspace |
| Check one execution | `npa workbench run "$RUN_ID" --action status` | That run's current status |
| Read its logs | `npa workbench run "$RUN_ID" --action logs` | Recorded workload logs |
| Find its outputs | `npa workbench run "$RUN_ID" --action artifacts` | Artifact index |
| Request cancellation | `npa workbench run "$RUN_ID" --action cancel` | Cancellation state; check until terminal |

Submit the workflow supplied by your administrator, or author an approved
[`npa.workflow` specification](npa-workflow-guide.md):

```bash
npa workbench submit --spec workflow.yaml
```

Select inputs without editing the supplied YAML by repeating `--var KEY=VALUE`:

```bash
npa workbench submit --spec paidf-cosmos3.yaml \
  --var input_kind=lerobot \
  --var lerobot_dataset_uri="$DATASET_URI" \
  --var input_camera=observation.images.top --var input_episode=19
```

Overrides are strings in the submitted workflow's `config`. The YAML stays
unchanged. Different inputs have separate retry receipts; repeating the same
inputs recovers the same run. The server still assigns output storage and
placement from your allocation. See [team PAIDF](team-paidf.md) for the operator
prerequisites, camera selection, and a recoverable 20-episode SDK example.

Save the returned `id` as `RUN_ID`, then inspect that same execution:

```bash
npa workbench runs
npa workbench run "$RUN_ID" --action status
npa workbench run "$RUN_ID" --action logs
npa workbench run "$RUN_ID" --action artifacts
```

`run "$RUN_ID"` without `--action` also reports status. Listing, status, logs,
and artifact inspection do not submit another workflow. The artifact command
lists outputs; [SDK download](team-access.md#user-sign-in-submit-and-inspect-work)
retrieves an artifact's bytes.

Repeating an unchanged submission recovers its original run, including after a
lost response. Add `--new-run` only when you intend another execution after an
acknowledged submission. `resume` reconciles a `recovery_required` run; it is
not a general rerun command. See [run recovery](team-access.md#user-sign-in-submit-and-inspect-work).

`npa logout` forgets the saved connection. It does not cancel running jobs or
revoke the server-side key.

## Coding agents use the same access

Give a coding agent the person's authorized saved connection, or issue a
separate local account and personal key for unattended automation. It needs no
operator kubeconfig, cloud administrator role, or private SkyPilot endpoint.
The CLI and SDK enforce the same ownership, namespace, quota, and storage rules.

```python
from pathlib import Path
import yaml
from npa.sdk.workbench.team import open_connection, submit_workflow

client, session = open_connection(profile="research")
try:
    workflow = yaml.safe_load(Path("workflow.yaml").read_text())
    run = submit_workflow(client, session, workflow)
    print(client.run(run["id"], "status"))
finally:
    client.close()
```

Create the `research` connection with `npa login --profile research` first.
CLI and SDK submissions share durable retry receipts. Agents should retain
run IDs and request identities, inspect recorded state after a disconnect,
and retrieve artifacts through the authenticated API. The server is the policy
and ownership boundary; SkyPilot remains the workload orchestrator.

## Existing installations, groups, and multiple clouds

- **Start alone, add people later:** keep the endpoint, account namespace, and
  durable state. Add each person to local grants and explicitly allocate their
  cluster GPU cap and storage. Group membership alone does not supply capacity.
- **Use existing Kubernetes clusters:** enroll Workbench-managed personal
  namespaces alongside existing workloads. Existing jobs and native SkyPilot
  clients do not automatically acquire Workbench's ownership or quota rules.
- **Use multiple Nebius projects:** the operator supplies access to each
  destination. A user needs Workbench grants and allocations for the selected
  workspace/cluster, not membership in every hosting cloud project.
- **Keep the server on Nebius and use another cloud:** follow
  [external Kubernetes setup](external-kubernetes.md). EKS access is implemented;
  AWS live execution has not been qualified. Direct cloud VMs are outside this
  shared execution path.
- **Add Nebius login later:** explicitly link the verified human identity to
  the existing local account. Local ownership stays the same. Nebius groups and
  roles are not imported as Workbench permissions.

## Where `npa configure` and native SkyPilot fit

`npa configure` prepares operator project and credential configuration. Shared
execution uses `npa login` followed by the commands above. Existing
`npa workbench team` commands remain compatibility aliases with the same accounts
and saved receipts. The Python SDK retains its `npa.sdk.workbench.team` module.

Ordinary `npa workbench workflow` commands retain the
[operator workflow](getting-started.md); setting `NPA_TEAM_ENDPOINT` does not
route them through the shared service. Native `sky` clients use their own
SkyPilot API server, not this Workbench API. Keep Workbench's scheduler private.
