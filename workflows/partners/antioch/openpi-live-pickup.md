# OpenPI π0.5 live pickup in Antioch

This flow runs a Franka cube pickup using a pretrained π0.5 policy. Antioch
supplies current exterior/wrist images and robot state; the OpenPI service on
Nebius returns action chunks that the simulator validates and executes. Model
weights remain fixed throughout the run.

The [simulator project](../../../npa/examples/antioch-openpi-live/README.md)
contains the scene, cameras, policy protocol, physical checks, and native robot
recording. The [Workbench integration guide](../../../docs/workbench/antioch.md#continuing-openpi-live-demonstration)
describes the authenticated connection and deployment prerequisites.

## Execution path

| Step | Component | Evidence |
| --- | --- | --- |
| Serve the reviewed π0.5 checkpoint | OpenPI GPU service on Nebius | Authenticated readiness and model contract |
| Start the owned simulator and policy connection | Antioch project plus Kubernetes adapter | Matching project/session ownership and bridge readiness |
| Run the finite pickup scenario | `openpi_franka_pickup_v3` | Advancing camera observations, finite `[15,8]` replies, and applied targets |
| Verify the result and recording | Persisted Antioch scenario record | Physical checks, termination reason, camera inputs, and native HD frames |
| Stop the owned scenario and adapter | Workbench cleanup | Confirmed stop before owned resource removal |

This is an operator-managed live workflow using the existing deployment commands.
It is not a `npa.workflow` training YAML and is not submitted with
`npa workbench workflow submit`. The separate
[ACT training YAML](antioch-offline-policy-train.yaml) consumes a saved dataset.

## Prepare and run

Follow the integration guide to prepare the policy service, an owned Antioch
project, immutable images, supported GPUs, and a fresh private runtime config.
The [runtime config template](../../../npa/examples/antioch-openpi-live/mk8s-runtime-config.example.json)
describes the required fields. Replace placeholders only in an owner-readable
runtime file outside Git. Use the supported Antioch login and credential staging
procedure; tokens, certificates, and infrastructure coordinates stay out of
workflow YAML, project source, command arguments, and recordings.

After the prerequisites are ready, use that private config for the adapter:

```bash
npa workbench antioch live-k8s-deploy \
  --runtime-config /path/to/private/runtime.json --output json
npa workbench antioch live-k8s-status \
  --runtime-config /path/to/private/runtime.json --output json
```

Deploying the adapter starts its finite scenario. A successful deployment or two
policy replies alone do not establish a pickup. Require the durable scenario
record to finish `completed/passed`, with advancing nonblack exterior/wrist
frames, finite `[15,8]` responses, measured approach, bilateral finger contact,
and the required lift and hold. Exhaustion and timeout are failures. Review the
native robot recording together with the exact policy camera inputs.

Read the persisted checks independently through the supported Antioch CLI before
reporting success or sharing the authenticated console view. Keep scene size,
initial posture, camera rig, and checkpoint identity with the private evidence;
one successful pickup does not establish repeatability or real-robot transfer.

Stop the exact owned scenario and adapter through the same runtime config:

```bash
npa workbench antioch live-k8s-stop \
  --runtime-config /path/to/private/runtime.json --output json
```

The shared policy service and checkpoint cache have a separate lifecycle. Follow
their ownership-aware cleanup procedure when retiring that deployment.
