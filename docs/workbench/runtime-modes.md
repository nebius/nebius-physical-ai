# Choose a Workbench runtime

[Workbench docs](README.md) · [CLI reference](../cli/workbench.md)

For pipelines, start with [Workbench setup](getting-started.md) and submit a
[workflow spec](npa-workflow-guide.md). Use direct tool deployment when you need
a persistent workbench or an existing VM. Check `npa workbench <tool> deploy --help`;
runtime support varies by tool.

| Runtime | What NPA manages | Prerequisites |
| --- | --- | --- |
| `vm` | Terraform VM and tool installation over SSH | Nebius project, Terraform, SSH key |
| `container` | Terraform VM and tool container over SSH | Same, plus the selected container's access requirements |
| `byovm` | Tool deployment on an existing VM | Reachable host, SSH key, compatible GPU and driver |
| `serverless` | AI Job or Endpoint, depending on the command | Nebius project permissions, subnet, supported image/platform |

Complete [configuration](../configuration.md) and the tool's credential and
model-access checks before deploying. Public NPA images resolve from GHCR;
[image compatibility](image-gpu-compatibility-matrix.md) depends on the exact tool
and GPU. Local engine extras are unnecessary when the engine runs remotely.

## Managed VM or container

For a LeRobot workbench, supply your configured project alias and choose a
workbench name. These parent selectors precede the subcommand:

```bash
npa workbench lerobot -p <project-alias> -n <workbench-name> deploy \
  --runtime container \
  --project-id <project-id> --tenant-id <tenant-id> --region <region>
npa workbench lerobot -p <project-alias> -n <workbench-name> status
npa workbench lerobot -p <project-alias> -n <workbench-name> system-info
```

Use the tool's deploy help to select GPU sizing and the matching image. A
successful deploy establishes the workbench; training and evaluation are
separate commands with their own dataset and checkpoint requirements.

## Bring your own VM (BYOVM)

Deploy to a VM you already operate:

```bash
npa workbench lerobot -p <project-alias> -n <workbench-name> deploy \
  --runtime byovm --host <ssh-host> \
  --ssh-user ubuntu --ssh-key ~/.ssh/id_ed25519
npa workbench lerobot -p <project-alias> -n <workbench-name> system-info
```

NPA probes `nvidia-smi` and saves the detected GPU names and count. If needed,
`--gpu-count <N>` restricts visible devices. Host defaults can also come from
`ssh.host`, `ssh.user`, and `ssh.key_path` in the credential store.

BYOVM does not create, resize, stop, or destroy the VM. A BYOVM `--destroy`
removes the local workbench entry; retire the actual VM through its owner.

## Serverless

The command determines whether serverless means a batch job or a serving
endpoint. For Cosmos, `deploy --runtime serverless` creates an endpoint;
`serve` checks or prewarms it. Changing its image or model requires redeployment.
See the [Cosmos serverless SDK](../sdk/cosmos-serverless.md).

Use `--subnet-id` explicitly when the project has multiple subnets. Pass secrets
through the private environment or credential store. Authentication success
does not establish permission to create AI Jobs or Endpoints in that project.

The older Cosmos serverless endpoint has no supported generated-media export
to S3. For an image or video artifact, use the
[Cosmos 3 generation workflow](cosmos3-generate.md). A serverless training smoke
that writes a status manifest does not prove model training.

## After a run

Inspect the workload's checkpoint, media, or report as well as its status.
Use the [teardown guide](../teardown.md) to remove owned resources in order;
local cleanup does not stop cloud compute.
