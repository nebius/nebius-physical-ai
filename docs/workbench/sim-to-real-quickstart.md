# Sim-To-Real H100 Quickstart

> **Production loop:** The maintained 13-stage VLM→RL pipeline is documented under
> [guides/sim2real-workflow.md](guides/sim2real-workflow.md) with data types in
> [guides/sim2real-data-contracts.md](guides/sim2real-data-contracts.md).
> This page covers the **legacy** `sim_to_real` H100 proof path only.

Run one small, real sim-to-real loop on an H100 and get a tangible result in S3:
a trained LeRobot policy checkpoint, a task-success metric, a JSON report, and a
Rerun artifact.

This guide uses the maintained sim-to-real stack:

- SkyPilot YAML: `npa/workflows/workbench/skypilot/sim-to-real-pipeline.yaml`
- CLI wrapper: `npa/scripts/run_sim_to_real_pipeline.py`
- Quickstart wrapper: `npa/scripts/run_sim_to_real_quickstart.py`
- Runtime module: `npa.workflows.sim_to_real real-loop`

The quickstart wrapper only fills defaults, submits the existing YAML, prints
the result, and tears the run down with `sky down` plus a status poll.

## Prerequisites

Complete the platform [quickstart](../quickstart.md) and keep storage credentials
in `~/.npa/credentials.yaml`:

```yaml
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.eu-north1.nebius.cloud
  bucket: s3://<your-bucket>/
```

The command also accepts the equivalent environment variables:

```bash
export S3_BUCKET=<your-bucket>
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export AWS_ACCESS_KEY_ID=<your-s3-access-key-id>
export AWS_SECRET_ACCESS_KEY=<your-s3-secret-access-key>
```

SkyPilot is installed outside the NPA virtualenv. The quickstart tries to reuse
`NPA_SKYPILOT_BIN` or the `skypilot.sky_bin` entry in `~/.npa/config.yaml`; if
neither exists, it bootstraps the pinned SkyPilot runtime before launch.

## One Command

From the repository root:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

The default run:

- provisions one `H100:1` worker through SkyPilot;
- downloads the pinned public LeRobot PushT dataset if the staged S3 copy is not
  already present;
- runs a tiny real LeRobot train/eval feedback loop;
- writes checkpoint weights, eval output, training signal, report, and Rerun
  artifact to `s3://<bucket>/sim-to-real/<run-id>/`;
- prints the task-success score and artifact URIs;
- runs `sky down` for the run-scoped cluster and polls until it is absent.

The final lines look like:

```text
sim-to-real quickstart result
run_id: s2r-quickstart-...
workflow_status: SUCCEEDED
wall_clock_seconds: ...
metric: task_success_score=...
checkpoint_uri: s3://<bucket>/sim-to-real/<run-id>/checkpoints/policy/
report_uri: s3://<bucket>/sim-to-real/<run-id>/reports/sim-to-real-report.json
rrd_uri: s3://<bucket>/sim-to-real/<run-id>/viz/<run-id>.rrd
teardown: cluster_absent=True
```

## Runtime

The quickstart is sized for a small proof run, not a converged policy. Warm runs
are expected to be in the 5-6 minute range when SkyPilot and package/image
caches are already warm. Cold runs can take longer because they
include H100 provisioning, source checkout, Python/runtime bootstrap, package
installation, and dataset staging. Treat the printed wall-clock as the source
of truth for each run.

## Override Points

Use flags or environment variables when you want to bring your own storage,
policy image, or dataset:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py \
  --bucket <your-bucket> \
  --s3-endpoint https://<your-s3-compatible-endpoint> \
  --source-ref <branch-or-tag> \
  --policy-image <registry>/<image>:<tag> \
  --input-data-uri s3://<your-bucket>/datasets/<your-lerobot-dataset>/
```

Common overrides:

| Setting | Default | Override |
| --- | --- | --- |
| Storage bucket | `storage.bucket` in `~/.npa/credentials.yaml` | `--bucket` or `S3_BUCKET` |
| S3 endpoint | `storage.endpoint_url` or `https://storage.eu-north1.nebius.cloud` | `--s3-endpoint`, `AWS_ENDPOINT_URL`, or `NEBIUS_S3_ENDPOINT` |
| Run prefix | `sim-to-real/<run-id>` | `--s3-prefix` |
| Source checkout | public repo `main` | `--source-repo` and `--source-ref` |
| Policy image | `npa-lerobot-policy:0.1.0` | `--policy-image` or `POLICY_IMAGE` |
| Dataset | pinned public LeRobot PushT staged under the run bucket | `--input-data-uri` |
| GPU | `H100:1` | `--gpu` |
| Step budget | 20 steps, one train/eval iteration | `--train-steps`, `--train-step-budget`, `--max-training-iterations` |

Keep `--gpu H100:1` for the quickstart proof path. Use other accelerators only
for explicit experiments.

## Cleanup Guarantees

The wrapper installs `SIGINT` and `SIGTERM` handlers. On success, failure, or
Ctrl-C it runs run-scoped teardown through `sky down --yes <pattern>` and polls
`sky status --refresh --output json` until no matching cluster remains. It does
not rely on unsupported teardown flags.

If teardown cannot prove absence, the command exits non-zero and reports
`cluster_absent=False`.

## Deeper Paths

Use the [sim-to-real pipeline cookbook](cookbooks/sim-to-real-pipeline.md) for
the equivalent raw SkyPilot, CLI wrapper, and SDK entry points. Those paths share
the same YAML, artifact layout, S3 handoff, eval backends, and BYO policy image
contract as this quickstart.
