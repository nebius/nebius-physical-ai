# Keep the Workbench server on Nebius and run on existing Kubernetes clusters

The server's location and a workflow's destination are independent. One
Workbench installation on Nebius can use explicitly enrolled Kubernetes
clusters on Nebius, AWS EKS, or another provider. Each destination has its own
connection, namespace boundaries, GPU allocations, and storage permissions.
There is no requirement for a destination cluster to have a Nebius project ID.

This path supports existing Kubernetes clusters. Direct AWS EC2 and other
cloud-VM provisioning remain outside the shared execution contract. Adding
the AWS authentication helper to the server image does not enable VM launches;
the scheduler and workflow policy still permit only enrolled Kubernetes targets.

## Administrator setup

1. Keep the existing Workbench HTTPS endpoint and persistent state on Nebius.
   Follow the [shared-access setup](shared-workbench-quickstart.md) only if an
   installation does not already exist.
2. Add the destination's explicitly selected `context` and private `kubeconfig`
   under `clusters` in the [access configuration](team-access.md). Add that
   cluster alias to the intended workspace and personal allocations. Do not
   copy the hosting cluster's Nebius project or cluster ID into this connection.
3. Apply enrollment with operator credentials against the selected destination:

   ```bash
   npa workbench team enroll --config /private/workbench/operator.yaml \
     --workspace research --cluster external-training --subject "$WORKBENCH_USER_ID"
   ```

4. Prepare a separate runtime connection with the scheduler's required
   Kubernetes permissions. The existing `scheduler_access_manifests` Python
   function renders read-only cluster discovery plus execution grants in the
   explicitly allocated namespaces. Set `subject_kind="Group"` and pass the
   exact EKS access-entry Kubernetes group as `account` for EKS IAM
   authentication; the same renderer grants identical restricted permissions.
   Do not grant cluster-admin or RBAC writes.
   Keep enrollment's operator credential separate from this runtime connection.
5. Export the runtime policy's connections for both server containers:

   ```bash
   npa workbench team export-connections \
     --config /private/workbench/runtime.yaml \
     --output-dir /private/workbench/connections
   ```

   This writes private `team.yaml`, `sky-kubeconfig.yaml`, and one minified
   kubeconfig per explicitly configured destination, including future
   destinations without personal allocations. Only allocated destinations get
   scheduler contexts. The export preserves workspace budgets and allocations,
   embeds certificate files through kubectl's flatten operation, preserves
   separate cluster contexts, and rewrites the gateway's connection paths to
   `/etc/npa-team`. `--server-directory` changes that mount location. Other
   policy paths, including state and workload storage, must already refer to
   the intended server paths. Existing output directories are never overwritten.
6. Add those files to the installation's existing configuration Secret,
   preserving its other files, account ownership namespace, and state PVC.
   Restart the server deployment after updating the Secret: its private copy
   is created at startup. Check that no active work requires an uninterrupted
   gateway before a planned restart.
7. From the gateway container or its equivalent server environment, run:

   ```bash
   npa workbench team check-clusters --config /etc/npa-team/team.yaml
   ```

   This verifies connection access and every configured allocation's quota,
   admission policy, workload identity and RBAC denials. It does not provision
   cloud resources. Then submit a real workload to the selected destination and
   verify its actual Pod namespace, logs and artifact before claiming live support.

## AWS EKS authentication

The server image includes the official AWS CLI through SkyPilot's AWS extra.
An exported `aws eks get-token` connection runs `/opt/sky/bin/aws` in both the
gateway and scheduler containers, even if the source used a laptop-specific
executable path. It preserves cluster, region and role arguments, selects an
explicit AWS profile, and references these private Secret files:

- `/etc/npa-team/aws-config`: AWS profile and optional assume-role configuration.
- `/etc/npa-team/aws-credentials`: the selected profile's source credentials.

Supply these files separately; the exporter does not copy your laptop's AWS
credentials. Inline AWS keys in the kubeconfig plugin environment are removed.
Instance-metadata fallback is disabled. Use a dedicated runtime principal with
the destination's EKS access entry and Kubernetes grants. AWS CLI obtains fresh
EKS tokens when the Kubernetes clients refresh authentication; short-lived
source credentials must themselves have an operator-managed refresh mechanism.
The default image's AWS CLI v1 is suitable for keys and assume-role profiles;
it does not supply AWS CLI v2 browser SSO or transfer a laptop's SSO session.
Other plugin environment settings fail export with a configuration error rather
than being silently dropped. Move role or credential-process configuration into
the server AWS profile and supply any referenced files or executables there.

The destination API must be reachable from the Nebius server. A public EKS API
can allow the server's outbound address; a private-only API requires private
routing such as a VPN. Being on another cloud does not itself require a private
link. EKS must support the required admission APIs and have network-policy
enforcement enabled in its CNI. Workloads also need permitted access to their
configured object storage and images. Credentials remain on the server;
end users need only their Workbench login.

Other Kubernetes providers can use self-contained certificate or token
credentials, or a server-installed credential plugin. A `tokenFile` reference
is preserved and must be mounted and refreshed on the server. Provider-specific
plugins beyond AWS are not bundled automatically.

## User workflow and existing SkyPilot clients

Select an authorized destination without changing the server endpoint:

```bash
npa workbench team submit --spec workflow.yaml \
  --workspace research --cluster external-training
```

The server assigns namespace and workload identity; a workflow cannot override
those bindings. Kubernetes placement uses `cloud: kubernetes`, including EKS.
It does not use `cloud: aws`, which means direct AWS VM provisioning in SkyPilot.

An existing native SkyPilot API server can independently configure external
Kubernetes credentials using [SkyPilot's deployment guide](https://docs.skypilot.ai/en/stable/reference/api-server/api-server-admin-deploy.html#optional-configure-cloud-accounts).
Its users can keep their existing `sky` clients. The Workbench HTTPS API is not
a native SkyPilot protocol proxy, and its private scheduler must not be exposed
to bypass Workbench authorization. Native SkyPilot access does not automatically
inherit Workbench's local-account, quota or storage guarantees.

## Validation scope

Hermetic tests cover portable AWS exec configuration, separate cluster routing,
credential handling, and failure behavior. The opt-in live connection gate is
`test_real_exported_connections_preserve_cluster_access` in
`npa/tests/e2e/test_team_access_live.py`; it uses the operator's explicitly selected
clusters and checks the exported files against the real Kubernetes API.
Live results must name the provider actually exercised. A Nebius workload or a
locally generated EKS authentication token does not prove an AWS workload ran.
