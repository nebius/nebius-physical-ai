# Historical Workbench backend checks

[Current workload guides](../workbench/guides/README.md)

This preserves the earlier guide-index evidence. It does not describe current
capacity or establish new end-to-end validation. Use the current guide for
your selected tool and GPU.

## Recorded backend checks

These entries record earlier checks against local and live backends. Each result
applies to the stated runtime and scope; the table does not establish that every
guide has completed end to end. Consult the selected tool's current guide and
skill for implementation changes since these checks.

| Path | Backend | Result |
| --- | --- | --- |
| `vlm-eval benchmark/run` (stub) | local, offline | works (`accuracy: 1.0`) |
| `lerobot train --runtime serverless --smoke` | Nebius AI Job (H200) | works — produced a real ACT checkpoint (`model.safetensors`) in S3 |
| `genesis train-teacher --runtime serverless` | Nebius AI Job (H100) | works, but is a **smoke** (import check + placeholder checkpoint); real Genesis training is local/VM |
| `sim_to_real.local_smoke` | local, no cluster | runs the spine; reports `blocked` unless `lerobot` is installed locally |
| `isaac-lab train --runtime serverless` | Nebius AI Job (`gpu-l40s-a`) | **capacity-blocked** — `NotEnoughResources` / VM schedule timeout |
| `isaac-lab train --runtime serverless` | Nebius AI Job (`gpu-l40s-d`) | job schedules and completes; minimal run produced no artifact yet (small step budget / `W9-isaac-lab-e2e-fix`) |
| `lerobot` / `fiftyone` deploy `--preemptible --dry-run` | Nebius Terraform VM path | CLI + dry-run OK; full apply needs IAM bootstrap on your project |
| Preemptible VM flags and resume | — | [preemptible-vms.md](../workbench/preemptible-vms.md) |

Isaac Lab needs RT cores, and serverless RT-core capacity varies by SKU: the
default `gpu-l40s-a` pool failed to schedule, while `gpu-l40s-d` had capacity and
ran to completion. `gpu-rtx6000` is **not** a serverless platform (use the
managed-Kubernetes path). For real Isaac Lab training prefer an RT-core VM /
managed-K8s + BYOF; for a serverless capacity retry use `--gpu-type gpu-l40s-d`.

These are historical records, not current capacity recommendations. Newer
[Genesis training evidence](../workbench/guides/franka-pick-and-place-genesis.md#go-bigger) records
real serverless PPO training and a held-out quality result. Current SONIC
capabilities and B200 MuJoCo evidence are in the [image catalog](../workbench/sonic-image-catalog.md).
