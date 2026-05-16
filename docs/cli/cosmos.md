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
ensure-ingress  Ensure public ingress for the saved Cosmos BYOVM alias.
register-byovm  Register an existing VM as a Cosmos BYOVM alias and ensure ingress.
list  List configured Cosmos workbenches.
cleanup-partial  Clean up orphaned Terraform resources from an interrupted Cosmos deploy.
deploy  Deploy or destroy a Cosmos model serving backend.
teardown  Delete a Cosmos serverless endpoint and remove its local alias.
reload-env  Propagate local shared credentials into the running Cosmos service env without redeploying.
serve  Start or pre-warm the saved Cosmos model server.
finetune  Roadmap placeholder for LoRA or full fine-tuning of Cosmos models on custom datasets.
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
| `ensure-ingress` | Ensure public ingress for the saved Cosmos BYOVM alias. |
| `register-byovm` | Register an existing VM as a Cosmos BYOVM alias and ensure ingress. |
| `list` | List configured Cosmos workbenches. |
| `cleanup-partial` | Clean up orphaned Terraform resources from an interrupted Cosmos deploy. |
| `deploy` | Deploy or destroy a Cosmos model serving backend. |
| `teardown` | Delete a Cosmos serverless endpoint and remove its local alias. |
| `reload-env` | Propagate local shared credentials into the running Cosmos service env without redeploying. |
| `serve` | Start or pre-warm the saved Cosmos model server. |
| `finetune` | Roadmap placeholder for LoRA or full fine-tuning of Cosmos models on custom datasets. |
| `optimize` | Roadmap placeholder for TensorRT compilation and quantization of Cosmos models. |
| `infer` | Submit a Cosmos inference job, poll until completion, then download the output. |
| `status` | Check the Cosmos endpoint health. |
| `system-info` | Collect and display system hardware information from the Cosmos VM. |

## Examples

```bash
npa workbench cosmos --help
npa workbench cosmos deploy --help
npa workbench cosmos teardown --help
npa workbench cosmos -p eu-north1 -n cosmos-sl deploy \
  --runtime serverless \
  --project-id project-... \
  --image cr.eu-north1.nebius.cloud/npa/cosmos:cuda12 \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --server-port 8080 \
  --subnet-id vpcsubnet-... \
  --wait
npa workbench cosmos -p eu-north1 -n cosmos-sl serve
npa workbench cosmos -p eu-north1 -n cosmos-sl infer --prompt "A robot arm stacks colored cubes"
npa workbench cosmos -p eu-north1 -n cosmos-sl teardown --yes
```

For `--runtime serverless`, `deploy` creates the Nebius Serverless AI Endpoint.
`serve` is an optional pre-warm/health call against the saved endpoint URL;
changing the model, image, platform, preset, env, or volumes requires a redeploy.
When the Nebius project has multiple subnets, include `--subnet-id`.

Regenerate this page with `bash scripts/build_docs.sh` after changing `cosmos`.
