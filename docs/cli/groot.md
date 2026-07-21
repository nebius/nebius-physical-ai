# `npa workbench groot`

## Command Tree

```text
Usage: npa workbench groot [OPTIONS] COMMAND [ARGS]...

NVIDIA Isaac GR00T humanoid foundation-model workbench.

Options
--project  -p  <str>  Project alias from ~/.npa/config.yaml.
--name  -n  <str>  Workbench instance name within the project.
--help  Show this message and exit.
Commands
ensure-ingress  Ensure public ingress for the saved GR00T BYOVM alias.
register-byovm  Register an existing VM as a GR00T BYOVM alias and ensure ingress.
list  List configured GR00T workbenches.
cleanup-partial  Clean up orphaned Terraform resources from an interrupted GR00T deploy.
deploy  Deploy or destroy a GR00T runtime VM with Isaac Lab available for sim evaluation.
download  Download GR00T model weights to the workbench VM or shared S3 storage.
reload-env  Propagate local shared credentials into the running GR00T service env without redeploying.
finetune  Fine-tune a GR00T action head on demonstration data with PyTorch.
eval  Evaluate a fine-tuned GR00T policy offline or through the S3 Isaac Lab data bus.
serve  Load a GR00T checkpoint and serve synchronous policy inference.
infer  Run GR00T policy inference over evaluation episodes and save predicted actions.
convert  Convert datasets between standard LeRobot and GR00T LeRobot layout.
status  Check the GR00T endpoint health.
system-info  Collect system information and GR00T runtime status from the VM.
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
| `ensure-ingress` | Ensure public ingress for the saved GR00T BYOVM alias. |
| `register-byovm` | Register an existing VM as a GR00T BYOVM alias and ensure ingress. |
| `list` | List configured GR00T workbenches. |
| `cleanup-partial` | Clean up orphaned Terraform resources from an interrupted GR00T deploy. |
| `deploy` | Deploy or destroy a GR00T runtime VM with Isaac Lab available for sim evaluation. |
| `download` | Download GR00T model weights to the workbench VM or shared S3 storage. |
| `reload-env` | Propagate local shared credentials into the running GR00T service env without redeploying. |
| `finetune` | Fine-tune a GR00T action head on demonstration data with PyTorch. |
| `eval` | Evaluate a fine-tuned GR00T policy offline or through the S3 Isaac Lab data bus. |
| `serve` | Load a GR00T checkpoint and serve synchronous policy inference. |
| `infer` | Run GR00T policy inference over evaluation episodes and save predicted actions. |
| `convert` | Convert datasets between standard LeRobot and GR00T LeRobot layout. |
| `status` | Check the GR00T endpoint health. |
| `system-info` | Collect system information and GR00T runtime status from the VM. |

## Examples

```bash
npa workbench groot --help
npa workbench groot ensure-ingress --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `groot`.
