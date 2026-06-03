# `npa workbench cosmos`

## Command Tree

```text
Usage: npa workbench cosmos [OPTIONS] COMMAND [ARGS]...

NVIDIA Cosmos world model serving and inference endpoints.

Options
--project  -p  TEXT  Project alias from ~/.npa/config.yaml.
--name  -n  TEXT  Workbench instance name within the project.
--help  Show this message and exit.
Commands
check  Check Cosmos3 source and HF checkpoint access without downloading weights.
fetch  Clone source and download the HF checkpoint into ephemeral runtime cache.
ensure-ingress  Ensure public ingress for the saved Cosmos BYOVM alias.
register-byovm  Register an existing VM as a Cosmos BYOVM alias and ensure ingress.
autoscale  Configure Cosmos serverless endpoint autoscaling.
list  List configured Cosmos workbenches.
cleanup-partial  Clean up orphaned Terraform resources from an interrupted Cosmos deploy.
deploy  Deploy or destroy a Cosmos model serving backend.
teardown  Delete a Cosmos serverless endpoint and remove its local alias.
reload-env  Propagate local shared credentials into the running Cosmos service env without redeploying.
serve  Start or pre-warm the saved Cosmos model server.
finetune  Roadmap placeholder for LoRA or full fine-tuning of Cosmos models on custom datasets.
train  Submit a Cosmos training job.
optimize  Roadmap placeholder for TensorRT compilation and quantization of Cosmos models.
infer  Submit a Cosmos inference job, poll until completion, then download the output.
status  Check the Cosmos endpoint health.
system-info  Collect and display system hardware information from the Cosmos VM.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  TEXT  Project alias from ~/.npa/config.yaml. |
| `--name` | -n  TEXT  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `check` | Check Cosmos3 source and HF checkpoint access without downloading weights. |
| `fetch` | Clone source and download the HF checkpoint into ephemeral runtime cache. |
| `ensure-ingress` | Ensure public ingress for the saved Cosmos BYOVM alias. |
| `register-byovm` | Register an existing VM as a Cosmos BYOVM alias and ensure ingress. |
| `autoscale` | Configure Cosmos serverless endpoint autoscaling. |
| `list` | List configured Cosmos workbenches. |
| `cleanup-partial` | Clean up orphaned Terraform resources from an interrupted Cosmos deploy. |
| `deploy` | Deploy or destroy a Cosmos model serving backend. |
| `teardown` | Delete a Cosmos serverless endpoint and remove its local alias. |
| `reload-env` | Propagate local shared credentials into the running Cosmos service env without redeploying. |
| `serve` | Start or pre-warm the saved Cosmos model server. |
| `finetune` | Roadmap placeholder for LoRA or full fine-tuning of Cosmos models on custom datasets. |
| `train` | Submit a Cosmos training job. |
| `optimize` | Roadmap placeholder for TensorRT compilation and quantization of Cosmos models. |
| `infer` | Submit a Cosmos inference job, poll until completion, then download the output. |
| `status` | Check the Cosmos endpoint health. |
| `system-info` | Collect and display system hardware information from the Cosmos VM. |

## Examples

```bash
npa workbench cosmos --help
npa workbench cosmos ensure-ingress --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cosmos`.
