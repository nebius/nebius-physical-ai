---
name: lerobot
description: Use when working on LeRobot workbench training, evaluation, serving, inference, dataset conversion, or robot policy workflows.
---

# LeRobot

LeRobot is the default robot policy training framework. It supports ACT, Diffusion Policy, and SmolVLA (and additional VLAs / world models in 0.6.0).

Use it as the data standard and policy interface layer, not as a managed-service competitor to Hugging Face.

## Supported versions

| Version | Role | Image tag | Notes |
| --- | --- | --- | --- |
| **0.5.1** | **Default** | `npa-lerobot:cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Accepted public default; the plain `0.5.1` alias is historical |
| **0.6.0** | Selectable alternative | `npa-lerobot:0.6.0-d6-extras-20260912` | Published immutable pin/digest; lean extras (`training,evaluation,pusht,libero,diffusion,smolvla`); `--eval_freq` → `--env_eval_freq` |

Select the package with `--lerobot-version`. Serverless training on 0.6.0
resolves the published image pin from the version manifest; `train --image`
remains available for a validated operator override. VM deployment installs the
selected package and has no `--image` option. The September 12 release supersedes
the anonymous September 5 audit that found no public 0.6.0 image. Its bare
`0.6.0` tag is a compatibility alias; use the manifest's immutable pin and digest.

Canonical manifest: `npa/src/npa/deploy/lerobot_version_manifest.json`.

Upstream release notes: https://huggingface.co/blog/lerobot-release-v060

## Interfaces

API:

- `POST /train`
- `POST /eval`
- `POST /serve`
- `POST /infer`
- `GET /list-checkpoints`

CLI:

```bash
npa workbench lerobot deploy
npa workbench lerobot deploy --runtime vm --lerobot-version 0.6.0
npa workbench lerobot train
npa workbench lerobot train --runtime serverless --lerobot-version 0.6.0 ...
npa workbench lerobot eval
npa workbench lerobot serve
npa workbench lerobot infer
npa workbench lerobot list-checkpoints
```

Build operator/BYOF image variants (official publication has separate gates):

```bash
npa/docker/workbench/lerobot/build.sh --registry '<operator-registry>' --all-versions
# or
npa/docker/workbench/lerobot/build.sh --registry '<operator-registry>' --version 0.6.0
```

The datacenter-Blackwell variant is `npa/docker/workbench/lerobot/Dockerfile.b300`.
LeRobot 0.5.1 requires Python 3.12, so this variant adds a dedicated
`/opt/lerobot/venv` with torch 2.9/cu130 and the matching torchcodec 0.8 line;
it does not reuse npa-base's Python 3.11 venv for the policy workload. The base
venv remains available at `/opt/npa/venv` for the baked architecture and kernel
validators. Keep both checks in hardware validation: validate the inherited
CUDA base, then run a real ACT training step from the LeRobot venv.

## Data Contract

Input format is `LeRobotDataset` in Hugging Face format. Use the SimToLeRobot adapter to convert Genesis or other simulation outputs.

Output is a policy checkpoint on S3.

## Validation

- 9/9 E2E serverless tests pass on Nebius (default 0.5.1 image).
- Tier 1 validated on B300.
- 0.6.0: the published digest passed `npa/scripts/validate_blackwell_image.sh`
  on a B200, constructed a 272,708-parameter `DiffusionPolicy` on CUDA, and
  passed all six environment checks on 2026-09-12. This proves the missing
  Diffusion Policy dependency is fixed; it is not a full training or SmolVLA
  benchmark. See the D3/D6 resolution in
  [the version audit](../../../docs/workbench/lerobot-version-support-audit-20260813.md#060-image-follow-up--2026-09-12)
  and merged [PR #462](https://github.com/nebius/nebius-physical-ai/pull/462).
