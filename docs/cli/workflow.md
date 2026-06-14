# `npa workbench workflow`

## Command Tree

```text
Usage: npa workbench workflow [OPTIONS] COMMAND [ARGS]...

Multi-stage training workflow orchestration.

Options
--help  Show this message and exit.
Commands
submit  Submit a SkyPilot workflow YAML through the NPA controller convention.
run  Run a named workflow end-to-end.
list  List durable S3 workflow runs.
status  Check the status of a workflow run.
logs  Show logs for a specific stage of a workflow run.
artifacts  List durable S3 artifact URIs for a workflow run.
cancel  Cancel a managed workflow job and explicitly tear down its cluster.
teardown  Destroy both VMs from a distill workflow run.
distill  Run expert distillation: L40S (Genesis) + H100 (LeRobot).
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `submit` | Submit a SkyPilot workflow YAML through the NPA controller convention. |
| `run` | Run a named workflow end-to-end. |
| `list` | List durable S3 workflow runs. |
| `status` | Check the status of a workflow run. |
| `logs` | Show logs for a specific stage of a workflow run. |
| `artifacts` | List durable S3 artifact URIs for a workflow run. |
| `cancel` | Cancel a managed workflow job and explicitly tear down its cluster. |
| `teardown` | Destroy both VMs from a distill workflow run. |
| `distill` | Run expert distillation: L40S (Genesis) + H100 (LeRobot). |

## Examples

```bash
npa workbench workflow --help
npa workbench workflow submit --help
```

## Durable S3 Monitoring

`submit --durable-s3` instruments a SkyPilot workflow with a writable S3
MOUNT-mode `file_mount`, redacted per-stage log teeing, and small JSON state
files. The storage location is resolved from `--workflow-s3-uri`,
`--workflow-s3-prefix`, `--s3-bucket`, project storage, or
`~/.npa/credentials.yaml`. S3-compatible endpoint and credentials come from the
same NPA storage config used by BYO S3 workflows; users do not need `aws
configure`, and they do not need to call `sky` to inspect completed runs.

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/<workflow>.yaml \
  --run-id "$RUN_ID" \
  --durable-s3 \
  --workflow-s3-uri "s3://<bucket>/workflows/$RUN_ID/" \
  --s3-endpoint "https://storage.eu-north1.nebius.cloud" \
  --infra "k8s/<context>"
```

The run prefix is the source of truth:

```text
s3://<bucket>/<run-id>/
  manifest.json
  logs/<stage>/run.log
  logs/<stage>/status.json
  artifacts/<stage>/...
```

`manifest.json` maps the run to stages, SkyPilot job IDs, status URIs, log
URIs, and artifact URIs. Each stage `status.json` includes `state`, `tier`,
`start`, `end`, `sky_job_id`, `artifact_uri`, `log_uri`, and `error_summary`.
Logs are scrubbed for common access-key and token patterns before they are
written to S3.

Monitor commands read the durable S3 state:

```bash
npa workbench workflow list --workflow-s3-uri "s3://<bucket>/workflows/"
npa workbench workflow status "s3://<bucket>/workflows/$RUN_ID/" --watch
npa workbench workflow logs "s3://<bucket>/workflows/$RUN_ID/" --stage train
npa workbench workflow artifacts "s3://<bucket>/workflows/$RUN_ID/"
npa workbench workflow cancel "s3://<bucket>/workflows/$RUN_ID/"
```

`status` opportunistically checks `sky jobs queue` when a SkyPilot binary and
job ID are available, but completed-run status, logs, and artifacts come from
S3. `logs --follow` tails live SkyPilot logs while the job is still running and
falls back to the durable S3 log when the live stream is unavailable.

Durable workflow state has no TTL in v1. Configure an S3 lifecycle policy on
the workflow prefix when retention needs are known, for example deleting
`logs/` and `artifacts/` objects after 30 or 90 days while keeping
`manifest.json` longer for audit history.

## `submit` Materialization

`submit` can replace `${VAR}` placeholders with repeated `--var KEY=VALUE`
arguments before calling SkyPilot. For SONIC YAMLs, `--var` also overrides
matching `envs` keys, and the command materializes the first-party image, S3
profile, endpoint, bucket, and prefix into the submitted YAML so SkyPilot does
not need to interpolate values inside `envs`.

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-train-standalone.yaml \
  --run-id sonic-smoke-$(date -u +%Y%m%dT%H%M%SZ) \
  --registry cr.eu-north1.nebius.cloud/<registry-id> \
  --gpu-target l40s \
  --region eu-north1 \
  --aws-profile nebius \
  --s3-endpoint https://storage.eu-north1.nebius.cloud \
  --s3-bucket <bucket> \
  --s3-prefix sonic-workflow-proof/<run-id> \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

For Nebius Container Registry VM pulls, the SONIC materializer adds SkyPilot's
Docker login envs to the submitted YAML:

```yaml
envs:
  SKYPILOT_DOCKER_USERNAME: iam
  SKYPILOT_DOCKER_PASSWORD: <fresh-nebius-iam-token>
  SKYPILOT_DOCKER_SERVER: cr.eu-north1.nebius.cloud
```

By default the password is minted at submit time with
`nebius iam get-access-token`, matching Nebius Container Registry's
short-lived-token login flow. BYO private registries can override the three
values with:

```bash
npa workbench workflow submit ... \
  --registry registry.example/workbench \
  --registry-server registry.example \
  --registry-username <username> \
  --registry-password <token>
```

Prefer `NPA_REGISTRY_USERNAME`, `NPA_REGISTRY_PASSWORD`, and
`NPA_REGISTRY_SERVER` when you do not want the token in shell history. Use
`--no-registry-auth` only for public images or environments that preconfigure
Docker auth outside SkyPilot. In `SONIC_PAYLOAD_MODE=docker`, the standalone
SONIC task uses the same envs for an in-task `docker login` before `docker pull`.

For the SONIC G1 fine-tune to MuJoCo MVP, submit
`npa/workflows/workbench/skypilot/sonic-locomotion-finetuning.yaml` with
`--gpu-target h100 --region eu-north1 --use-spot --require-controller-up` and
`--var SONIC_PAYLOAD_MODE=docker`. The materializer selects
`npa-sonic-mujoco:0.1.3-mvp`, writes `region: eu-north1` into Nebius VM GPU
tasks, and rejects `me-west1`.

For RTX PRO 6000 Kubernetes targets, use the same command with
`--gpu-target gpu-rtx6000` and an accelerator string accepted by your SkyPilot
Kubernetes GPU catalog, for example
`--accelerators RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`. The SONIC
materializer resolves `gpu-rtx6000` to `npa-sonic:0.1.2-k8s-runtime`; L40S resolves to
`npa-sonic:0.1.2`.

When a Kubernetes target pulls from a private registry, provide a SkyPilot config
through `--config-path` that adds the registry pull secret to worker pods:

```yaml
kubernetes:
  pod_config:
    spec:
      imagePullSecrets:
        - name: <registry-pull-secret>
```

Only set `serviceAccountName` in that pod config if the account also has the
cluster-level permissions SkyPilot needs for node and pod discovery.

Regenerate this page with `bash scripts/build_docs.sh` after changing `workflow`.
