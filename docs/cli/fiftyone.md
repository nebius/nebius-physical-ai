# `npa workbench fiftyone`

## Command Tree

```text
Usage: npa workbench fiftyone [OPTIONS] COMMAND [ARGS]...

Voxel51 FiftyOne dataset curation and visualization workbench.

Options
--project  -p  <str>  Project alias from ~/.npa/config.yaml.
--name  -n  <str>  Workbench instance name within the project.
--help  Show this message and exit.
Commands
ensure-ingress  Ensure public ingress for the saved FiftyOne BYOVM alias.
register-byovm  Register an existing VM as a FiftyOne BYOVM alias and ensure ingress.
list  List configured FiftyOne workbenches.
cleanup-partial  Clean up orphaned Terraform resources from an interrupted FiftyOne deploy.
deploy  Deploy or destroy a FiftyOne dataset curation VM.
launch  Start the FiftyOne app over SSH and print the browser URL.
curate  Curate a dataset and export a LeRobotDataset on Nebius Serverless.
eval  Evaluate checkpoint outputs and write FiftyOne curation metrics.
load-dataset  Load a dataset into FiftyOne on the VM.
restart  Restart the FiftyOne app or container without redeploying.
open  Port-forward the FiftyOne App to localhost and open it in the browser.
status  Check whether the FiftyOne app responds on its web port.
system-info  Collect and display system hardware information from the FiftyOne VM.
datasets  Inspect datasets through the FiftyOne GraphQL API.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  <str>  Project alias from ~/.npa/config.yaml. |
| `--name` | -n  <str>  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `ensure-ingress` | Ensure public ingress for the saved FiftyOne BYOVM alias. |
| `register-byovm` | Register an existing VM as a FiftyOne BYOVM alias and ensure ingress. |
| `list` | List configured FiftyOne workbenches. |
| `cleanup-partial` | Clean up orphaned Terraform resources from an interrupted FiftyOne deploy. |
| `deploy` | Deploy or destroy a FiftyOne dataset curation VM. |
| `launch` | Start the FiftyOne app over SSH and print the browser URL. |
| `curate` | Curate a dataset and export a LeRobotDataset on Nebius Serverless. |
| `eval` | Evaluate checkpoint outputs and write FiftyOne curation metrics. |
| `load-dataset` | Load a dataset into FiftyOne on the VM. |
| `restart` | Restart the FiftyOne app or container without redeploying. |
| `open` | Port-forward the FiftyOne App to localhost and open it in the browser. |
| `status` | Check whether the FiftyOne app responds on its web port. |
| `system-info` | Collect and display system hardware information from the FiftyOne VM. |
| `datasets` | Inspect datasets through the FiftyOne GraphQL API. |

## Examples

```bash
npa workbench fiftyone --help
npa workbench fiftyone ensure-ingress --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `fiftyone`.
