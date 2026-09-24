# Team operation and access boundaries

[Workbench docs](README.md) · [Project configuration](../configuration.md) ·
[Identity/RBAC design](identity-rbac-audit.md)

## Short answer

An infrastructure administrator can prepare a Nebius project, storage, and a
Kubernetes cluster for another person to use. That split is assembled from
Nebius IAM, Kubernetes RBAC, and storage credentials; it is **not** an NPA team
or role feature.

NPA currently has no team membership, no `viewer` / `operator` / `admin` role
enforcement, no per-user run attribution, and no central control-plane audit
log. Each person runs the CLI with their own Nebius profile and owner-only local
NPA configuration. The browser agent has one Basic Auth account for the entire
deployment, so it cannot give one person read-only access while reserving
administration for another.

This distinction matters:

- **Admin prepares infrastructure; trusted worker uses the CLI:** workable with
  external IAM/RBAC and deliberately issued credentials.
- **Admin and restricted worker share one browser workbench:** not supported.
- **Read-only teammate:** share selected artifacts or grant read-only storage
  access; do not share the agent login and call it viewer access.
- **Audited multi-user operation:** not supported yet. The proposed model is in
  [Identity, RBAC, and Audit Trail](identity-rbac-audit.md).

## Where access is enforced today

| Surface | Current authorization boundary | Per-person today? | Important limitation |
| --- | --- | --- | --- |
| Project and cloud resources | The active Nebius CLI principal and Nebius IAM | Yes, when every person uses their own profile | NPA does not translate cloud roles into NPA roles or pre-authorize individual commands. |
| Initial `npa configure` storage provisioning | Nebius project `admin`, because setup manages a project-scoped group, membership, and access permit | Yes at the cloud call | Provisioning credentials and ownership records are stored only in the administrator's local `~/.npa/`. |
| Kubernetes workloads | The selected kubeconfig and Kubernetes RBAC | Yes, when every person receives their own context/credentials | NPA does not create team Roles or RoleBindings. The user needs the verbs checked by `kubectl auth can-i`. |
| Workflow artifact storage | S3 access key/secret and bucket policy | Only with a distinct credential per person or workload | NPA's automatically provisioned storage credential is project-scoped, not a human identity. Sharing it removes per-person attribution. |
| Workflow submission and cancellation | Possession of the required Nebius, Kubernetes, SkyPilot, and S3 capabilities | No NPA role check | A person who can reach the mutation path is not classified as an NPA operator or admin. |
| Browser agent | One deployment-wide Basic Auth username/password | No | Every authenticated browser session reaches the same UI/API and acts through the same attached `npa-agent` service account, which has project `editor`. |
| Run records | S3 run manifest plus an owner-only local submission receipt | No | `npa.workflow.run.v1` has no submitting principal, and the local receipt is not a team audit log. |
| Artifact sharing | Storage policy or a tool-specific share URL | Sometimes | Sharing an artifact does not grant a constrained Workbench role. |

The owner-only files are a security boundary for one operating-system account,
not a team database. Do not copy an administrator's `~/.npa/`, Nebius OAuth
cache, kubeconfig, or private keys to onboard another person.

## Safe admin/worker pattern

Use this pattern only when cloud IAM, Kubernetes RBAC, and storage policy are
the intended enforcement points.

### Administrator

1. Create or select the project and perform the initial `npa configure`
   provisioning. This is the part that requires project `admin`.
2. Provision or register the shared Kubernetes cluster and retain the local
   Terraform, ownership, and teardown records.
3. Grant the worker's own Nebius principal only the project permissions needed
   for their workload. A cloud `viewer` role does not permit workload
   submission; avoid granting project `admin` merely to make setup convenient.
4. Create Kubernetes RBAC for the worker's own cluster identity. Workflow
   submission normally needs pod access in `default`; deployed services use
   `workbench`. Scope destructive and secret-management verbs separately.
5. Issue a dedicated, scoped storage identity for the worker or workload. Do
   not distribute the administrator's local credential store. If the team
   instead shares the NPA-provisioned S3 key, record that it is a shared
   workload identity and that individual attribution is unavailable.
6. Keep infrastructure destruction and credential rotation in the
   administrator's runbook. NPA does not enforce that division.

### Worker

1. Authenticate the Nebius CLI as your own principal. Do not reuse the
   administrator's OAuth cache or service-account key.
2. Save the project coordinates locally without attempting to reprovision
   storage:

   ```bash
   npa configure --no-interactive --no-provision \
     --tenant-id "<tenant-id>" --project-id "<project-id>" \
     --region "<region>" --project-alias "<project-alias>"
   ```

3. Put only your assigned workload credentials in your owner-only NPA store,
   and use the kubeconfig issued for your identity.
4. Verify the effective capabilities before submitting:

   ```bash
   npa workbench health preflight --project "<project-alias>" \
     --checks nebius,s3 --json
   kubectl auth can-i create pods -n default
   kubectl auth can-i list pods -n default
   kubectl auth can-i list nodes
   ```

5. Submit and inspect workloads with a unique run ID. Until run attribution is
   implemented, use a team naming convention only as an operational hint; it
   is self-asserted and is not an audit identity.

Passing these checks proves that the selected credentials can perform those
operations. It does not mean NPA assigned or verified a worker role.

## Browser agent sharing

Treat the browser agent login as a credential for a fully trusted operator. The
deployment writes one `AGENT_USER` / `AGENT_PASSWORD` pair, nginx applies it to
the general UI and API, and backend work uses one attached project-editor
service account. There is no separate account registry, read-only session,
per-user revocation, or actor field on a submitted run.

If two people know that login, NPA cannot tell them apart. Deploying separate
agent VMs gives them different browser passwords, but current agent deployments
still reuse the named `npa-agent` cloud identity and do not create an NPA role
boundary.

For an observer, prefer a narrowly scoped artifact path such as an S3 read-only
grant or a Rerun/Foxglove export. Review each sharing mechanism separately;
artifact access and Workbench control-plane access are different permissions.

## Remaining product gap

First-class team operation requires all of the following together:

- a verified principal for each human, service account, and delegated agent;
- project-scoped team membership and enforced roles;
- `submitted_by` attribution in the durable run manifest;
- authorization for submit, cancel, configure, credential issuance, and
  destroy, including ownership rules for another person's run;
- a centralized append-only audit trail; and
- per-user browser authentication rather than one deployment password.

The phased proposal and threat model are tracked in
[Identity, RBAC, and Audit Trail](identity-rbac-audit.md). Until server-side
authorization exists, Nebius IAM, Kubernetes RBAC, and storage policy remain
the actual security boundaries; CLI checks alone would only improve UX.
