# LanceDB Deploy Runbook

This runbook covers the OSS LanceDB Workbench path. LanceDB is CPU-only; do
not request GPUs for this service.

## Runtime Modes

| Runtime | Purpose | Notes |
| --- | --- | --- |
| `container` | Local development and smoke validation | Runs Docker on the operator machine. |
| `vm` | Production OSS path on a Nebius CPU VM | Uses an addressable service backed by S3-compatible storage. |
| `byovm` | Existing SSH-accessible VM | Useful when infrastructure is pre-provisioned. |
| `cloud` | Existing LanceDB Cloud or Enterprise endpoint | Connection-only; no provisioning. |

`serverless` is intentionally not a deploy runtime because LanceDB is a
persistent service, not a batch Job.

## Local Container Smoke

Build:

```bash
npa/docker/workbench/lancedb/build.sh
```

The pushed first-party default is
`cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3`.

Deploy:

```bash
npa workbench lancedb deploy \
  --runtime container \
  --storage-path /tmp/npa-lancedb \
  --port 8686 \
  --auth-mode none \
  --replace \
  --image cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3
```

Check status:

```bash
npa workbench lancedb status --endpoint http://localhost:8686
```

Remove the smoke container:

```bash
npa workbench lancedb deploy \
  --runtime container \
  --storage-path /tmp/npa-lancedb \
  --port 8686 \
  --destroy
```

## VM Deployment Checklist

1. Choose an S3-compatible storage prefix, for example
   `s3://robot-data/lancedb/`.
2. Confirm `~/.npa/credentials.yaml` or the environment contains object
   storage credentials.
3. Set a token for the service:

```bash
export LANCEDB_TOKEN=...
```

4. Deploy with CPU settings:

```bash
npa workbench lancedb deploy \
  --runtime vm \
  --storage-path s3://robot-data/lancedb/ \
  --project-id project-... \
  --tenant-id tenant-... \
  --region eu-north1 \
  --cpu-preset 4vcpu-16gb \
  --port 8686 \
  --auth-mode token
```

5. Verify health:

```bash
npa workbench lancedb status \
  --endpoint http://<vm-ip>:8686 \
  --token-env LANCEDB_TOKEN
```

## Network And Auth

Use token auth for non-local deployments. `--auth-mode none` is intended only
for local development or private throwaway smoke runs.

Production operators should restrict inbound access to trusted networks and
avoid exposing the service broadly on the public internet.

## Storage Requirements

For S3-backed storage, pass these values through environment or credentials:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ENDPOINT_URL`
- `AWS_REGION`

Use a dedicated prefix for each LanceDB service to avoid accidental table
mixing across demos or customers.

## Backup And Restore

The v1 backup model is storage-level:

- Snapshot or replicate the S3 prefix used by `--storage-path`.
- For local container storage, stop the container and archive the local storage
  directory.
- Restore by pointing a new LanceDB service at the copied prefix or directory.

Dedicated `backup` and `restore` commands are deferred to v2.

## Teardown

For local containers:

```bash
npa workbench lancedb deploy \
  --runtime container \
  --storage-path /tmp/npa-lancedb \
  --port 8686 \
  --destroy
```

For VM deployments, use the same project/name context and pass `--destroy`.
Confirm no other workload is using the same storage prefix before deleting or
archiving data.

## Known Limitation In This Build Run

`npa workbench lancedb` is wired into the parent CLI. The remaining live-service
gap is managed VM registration: the `container` and `cloud` paths are usable for
local smoke and existing endpoints, while the `vm`/`byovm` app deploy path still
requires Workbench parent registration work before it is a one-command
production service deploy.
