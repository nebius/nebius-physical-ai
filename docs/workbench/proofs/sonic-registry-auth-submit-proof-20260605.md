# SONIC registry-auth submit proof, 2026-06-05

This note records the live VM-GPU proof attempt for the SONIC `npa workbench
workflow submit` path after adding SkyPilot Docker registry auth materialization.
The proof used an isolated `/tmp` checkout on branch
`dev-fix-sonic-registry-auth-proof`; it did not modify `.github/workflows`, did
not touch the LeRobot cluster, and avoided the #47 overlap files.

## Registry auth

The SONIC materializer now emits SkyPilot Docker login environment variables for
VM image pulls:

- `SKYPILOT_DOCKER_USERNAME`
- `SKYPILOT_DOCKER_PASSWORD`
- `SKYPILOT_DOCKER_SERVER`

For first-party Nebius Container Registry images, the default username is `iam`
and the password is a fresh `nebius iam get-access-token` value minted at submit
time. This follows the Nebius Container Registry short-lived token flow:
`nebius iam get-access-token | docker login cr.<region>.nebius.cloud --username
iam --password-stdin`. Customer BYO registry credentials can override the
defaults through CLI flags, SDK parameters, or environment variables.

Local auth sanity check:

- `nebius iam get-access-token` succeeded.
- `docker login cr.eu-north1.nebius.cloud` with username `iam` succeeded.
- `docker manifest inspect cr.eu-north1.nebius.cloud/<registry-id>/npa-sonic:0.1.2`
  succeeded after login.
- S3 profile access to `s3://npa-sim2real-d87cf691/` succeeded with
  `AWS_PROFILE=nebius`.

Secrets were not written to this file or to PR text.

## Capacity oracle

The capacity oracle was run with:

```bash
nebius capacity resource-advice list \
  --parent-id <capacity-tenant-id> \
  --all \
  --format json \
| jq -r '
  .items[] |
  [
    .spec.region,
    .spec.fabric,
    .spec.compute_instance.platform,
    .spec.compute_instance.preset.name,
    .status.reserved.available,
    .status.ondemand.available,
    .status.preemptible.available
  ] | @tsv'
```

Available 1-GPU VM candidates were preemptible only:

| Priority | GPU | Region | Platform | Preset | Preemptible |
| --- | --- | --- | --- | --- | --- |
| 1 | H100 | `eu-north1` | `gpu-h100-sxm` | `1gpu-16vcpu-200gb` | 75 per listed fabric |
| 2 | H200 | `eu-north1` | `gpu-h200-sxm` | `1gpu-16vcpu-200gb` | 75 |
| 3 | L40S | `eu-north1` | `gpu-l40s-a` | `1gpu-16vcpu-64gb` | 9 |
| 4 | B200 | `us-central1` / `me-west1` | `gpu-b200-sxm*` | `1gpu-20vcpu-224gb` | 81 / 55 |

B200 remained last because it is an untested architecture for this SONIC image
and the available capacity was outside the launchable `eu-north1` project path.

## Live submit attempts

All attempts used:

- `NPA_SKYPILOT_BIN=/home/ubuntu/.npa/skypilot-venv/bin/sky`
- `npa/.venv/bin/npa workbench workflow submit`
- `--tool sonic`
- `--controller-backend nebius`
- `--registry cr.eu-north1.nebius.cloud/<registry-id>`
- `--use-spot`
- S3 proof prefixes under
  `s3://npa-sim2real-d87cf691/sonic-submit-proof/<run-id>/`
- scoped cleanup per attempt with `sky jobs cancel <id> --yes`, `sky down
  <cluster> --yes` when SkyPilot had a cluster handle, and Nebius instance
  polling for the run-specific name.

| Attempt | Managed job | Placement | Result |
| --- | --- | --- | --- |
| `sonic-proof-h100-20260605t220047z` | 1 | H100 | Failed SkyPilot precheck before launch because the previous H100 default used `mem=64`; this PR updates H100/H200 VM defaults to `mem=200`. |
| `sonic-proof-h100b-20260605t221200z` | 2 | `eu-north1`, `gpu-h100-sxm_1gpu-16vcpu-200gb[Spot]` | Instance came up and SkyPilot reported "Docker container is up", so the private-image pull path progressed past the prior registry-auth seam. SkyPilot then failed runtime setup with repeated `stdio forwarding failed` and `mkdir -p ~/.sky/.runtime_files` returning 255 before the SONIC entrypoint ran. The worker was cancelled and reached absence in Nebius. |
| `sonic-proof-h200-20260605t222900z` | 3 | `eu-north1`, `gpu-h200-sxm_1gpu-16vcpu-200gb[Spot]` | Instance came up but did not reach task execution or S3 output during the bounded proof window. The worker was cancelled and reached absence in Nebius. |
| `sonic-proof-l40s-20260605t223800z` | 4 | `eu-north1`, `gpu-l40s-a_1gpu-16vcpu-64gb[Spot]` | Instance came up but did not reach task execution or S3 output during the bounded proof window. The worker was cancelled and reached absence in Nebius. |
| `sonic-proof-b200-20260605t223959z` | 5 | SkyPilot selected `me-west1`, `gpu-b200-sxm-a_1gpu-20vcpu-224gb[Spot]` | Failed before provisioning because the configured SkyPilot/Nebius project has no project mapping for `me-west1`. Nebius showed no run-scoped B200 instance. SkyPilot's managed-job state settled at `FAILED_CONTROLLER`. |

## Tier

The registry-auth seam is fixed in the materialized VM YAML: H100 progressed to
SkyPilot's Docker-container-up stage with a private `cr.eu-north1.nebius.cloud`
image, whereas the previous live blocker was private-registry pre-pull auth. The
full SONIC entrypoint and S3 proof artifact did not complete on 2026-06-05
because each launchable GPU then hit SkyPilot VM runtime setup before task
execution; B200 additionally selected a non-launchable region for this project.

No successful S3 artifact exists under the attempted proof prefixes. The next
live-proof step is to resolve the SkyPilot VM runtime setup failure or constrain
B200 placement to a configured project region before rerunning the same submit
command. The SkyPilot-on-k8s image compatibility seam remains out of scope for
this VM/H100 MVP proof.
