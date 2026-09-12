# Choose a Workbench workload

[Workbench docs](../README.md) · [Cookbooks](../cookbooks/README.md) · [Workflow catalog](../../../workflows/README.md)

Choose the result you want, then follow that guide from inputs to inspected
artifacts. Complete the [quickstart](../../quickstart.md) and
[GPU runtime setup](../getting-started.md) before a cloud run. A local preview
checks the declaration; each guide states what its live validation actually proves.

## Robot and reconstruction guides

| Goal | Guide and required input | Compute and result scope |
| --- | --- | --- |
| Train a Franka pick-and-place teacher | [Franka / Genesis](franka-pick-and-place-genesis.md); generated simulation tasks | H200 headless training was exercised; recorded held-out success was 0%. Read the rendering caveat. |
| Inspect the PushT SDK structure | [PushT smoke](pusht-sim-to-real.md); `lerobot/pusht` reference | Local structural smoke; follow the linked training path for learned weights. |
| Train a Reachy 2 policy | [Reachy 2 / LeRobot](reachy2-lerobot-policy.md); a dataset with matching observation/action schemas | GPU policy training; validate compatibility before substituting a dataset. |
| Train or evaluate Unitree G1 locomotion | [G1 / SONIC](g1-humanoid-walk-sonic.md); compatible motion/checkpoint inputs | RTX PRO 6000 training; B200 MuJoCo evaluation is a separate capability. |
| Train an ANYmal quadruped | [Quadruped / Isaac Lab](quadruped-isaac-lab.md); built-in simulator task | L40S or RTX PRO 6000; Isaac requires RT cores. |
| Reconstruct a scene capture | [NuRec / NRE](neural-reconstruction.md); compatible NCore capture | RTX PRO 6000 or L40S; inspect USDZ, renders, and Rerun outputs. |
| Reconstruct several living-lab zones | [Living-lab fan-out](living-lab-nurec-fanout.md); capture inputs per zone | Advanced parameterized workflow; default is 16 RTX PRO 6000 GPUs. |

For the candidate COLMAP ingestion path, see [NCore conversion](nurec-colmap-reconstruct.md);
it remains unvalidated end to end. For browser teleoperation measurements, see
[LeIsaac transport latency](leisaac-transport-latency.md).

## Generation and dataset production

| Goal | Start with |
| --- | --- |
| Generate images with Cosmos 3 | [Generation guide](../cosmos3-generate.md) and [access preflight](../cosmos3-access-preflight.md) |
| Augment your source video with Cosmos 3 | [PAIDF + Cosmos 3](paidf-cosmos3.md) and [setup/run procedure](../../../workflows/guides/paidf-cosmos3.md) |
| Produce a labeled dataset with Cosmos Transfer | [Data Factory deployment](physical-ai-data-factory-deploy.md); its [quickstart](physical-ai-data-factory-deploy.md#quick-start-copy-paste) can seed generated frames |
| Understand Data Factory stages and artifacts | [Component and S3 mapping](physical-ai-data-factory.md) |
| Reuse a verified dataset campaign | [Campaign reuse](paidf-campaign-reuse.md) |
| Caption or reason about existing artifacts | [Token Factory](../token-factory.md) |

<a id="physical-ai-data-factory-video-data-augmentation"></a>

Data Factory composes annotation, Cosmos augmentation, evaluation, curation, and
Rerun visualization on Nebius + SkyPilot. Select the actual workflow's inputs
and model access requirements; generation and input-conditioned augmentation
have different contracts.

## Sim-to-real: the full 14-stage loop

This path needs robot assets, compatible source data, prepared compute, immutable
images, and storage. Start with the operator runbook, then use the contract pages
when adapting it:

| Task | Guide |
| --- | --- |
| Configure, preflight, submit, and recover | [Sim2Real operator runbook](sim2real-workflow.md) |
| Prepare formats, schemas, and S3 layout | [Data contracts](sim2real-data-contracts.md) |
| Supply customer data and robot assets | [Customer assets](sim2real-customer-assets.md) · [RobotSpec](sim2real-robot-spec.md) |
| Understand stages, loops, and parallel execution | [Architecture](sim2real-architecture.md) |
| Recover after interruption | [Durable runtime behavior](sim2real-durable-controller.md) |
| Present a prepared result | [Demonstration script](sim2real-demo-script-10min.md) |

<a id="bring-your-own-everything"></a>

## Use your own data, policy, or robot

Match the selected tool's dataset format, observation/action schema, runtime,
and output contract. A new robot may also need simulator assets and action
mappings. Use [BYOF](../cookbooks/byof-isaac-lab/README.md) for an Isaac image or
training-entrypoint customization, and the [cookbooks](../cookbooks/README.md)
for other tool-specific recipes.

<a id="recorded-backend-checks"></a>

Earlier local, serverless, and capacity checks are retained in the
[historical backend record](../../archive/workbench-backend-checks.md).
For a new run, use the [current image/GPU matrix](../image-gpu-compatibility-matrix.md)
and inspect that run's actual artifacts.
