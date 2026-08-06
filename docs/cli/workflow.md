# `npa workbench workflow`

## Command Tree

```text
Usage: npa workbench workflow [OPTIONS] COMMAND [ARGS]...

Multi-stage training workflow orchestration.

Options
--help  Show this message and exit.
Commands
prepare-run  Prepare a project/workflow-scoped fresh or explicit-resume run ID.
submit  Submit a SkyPilot or npa.workflow/v0.0.1 YAML through the NPA controller.
run  Run a named workflow end-to-end.
status  Check the status of a workflow run.
logs  Show logs for a specific stage of a workflow run.
artifacts  List durable S3 artifact URIs for a workflow run.
load-artifact  Retry only the final artifact load; never relaunch workflow stages.
list  List durable S3 workflow runs.
cancel  Cancel a launched run; never-launched and terminal runs are repeat-safe no-ops.
teardown  Destroy both VMs from a distill workflow run.
distill  Run expert distillation: L40S (Genesis) + H100 (LeRobot).
stage-src  Upload the local npa package to S3 for image-less workflow steps.
validate-spec  Validate an NPA workflow specification file.
plan-spec  Expand an NPA workflow spec into an execution plan (dry-run).
run-spec  Run or plan an NPA workflow spec.
preflight-images  Prove every image this spec pulls is pullable, with the run's own credentials.
gpus  Print the accelerator names this cluster advertises to SkyPilot.
trigger  Watch S3-compatible data prefixes and retrigger Workbench workflows.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `prepare-run` | Prepare a project/workflow-scoped fresh or explicit-resume run ID. |
| `submit` | Submit a SkyPilot or npa.workflow/v0.0.1 YAML through the NPA controller. |
| `run` | Run a named workflow end-to-end. |
| `status` | Check the status of a workflow run. |
| `logs` | Show logs for a specific stage of a workflow run. |
| `artifacts` | List durable S3 artifact URIs for a workflow run. |
| `load-artifact` | Retry only the final artifact load; never relaunch workflow stages. |
| `list` | List durable S3 workflow runs. |
| `cancel` | Cancel a launched run; never-launched and terminal runs are repeat-safe no-ops. |
| `teardown` | Destroy both VMs from a distill workflow run. |
| `distill` | Run expert distillation: L40S (Genesis) + H100 (LeRobot). |
| `stage-src` | Upload the local npa package to S3 for image-less workflow steps. |
| `validate-spec` | Validate an NPA workflow specification file. |
| `plan-spec` | Expand an NPA workflow spec into an execution plan (dry-run). |
| `run-spec` | Run or plan an NPA workflow spec. |
| `preflight-images` | Prove every image this spec pulls is pullable, with the run's own credentials. |
| `gpus` | Print the accelerator names this cluster advertises to SkyPilot. |
| `trigger` | Watch S3-compatible data prefixes and retrigger Workbench workflows. |

## Examples

```bash
npa workbench workflow --help
npa workbench workflow prepare-run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workflow`.
