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
| **0.5.1** | **Default package compatibility** | `npa-lerobot:cuda13-b300-0.5.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z` | Historical public image is publication-quarantined; use only a separately validated immutable replacement |
| **0.6.0** | Selectable alternative | `npa-lerobot:0.6.0-d6-extras-20260912` | Historical immutable pin/digest; lean extras (`training,evaluation,pusht,libero,diffusion,smolvla`); `--eval_freq` → `--env_eval_freq`. It is not FastWAM-qualified. |

Select the package with `--lerobot-version`. `train --image` and container/BYOVM
`deploy --image` accept a validated immutable operator override. The default
0.5.1 public image is publication-quarantined because its OCI metadata exposes
operator-specific build provenance; naming the historical tag does not bypass
that gate. VM-native deployment installs the selected package without consuming
the container release. The September 12 release supersedes
the anonymous September 5 audit that found no public 0.6.0 image. Its bare
`0.6.0` tag is a compatibility alias; use the manifest's immutable pin and digest.
The optional 0.6.0 image is validated on B200 and does not replace the 0.5.1 default.

FastWAM requires a separately rebuilt immutable 0.6.0 image containing the
`fastwam` extra; do not attribute that capability to the historical digest.
Use `workflows/testing/fastwam-policy-qualification.yaml` for its native
five-stage prepare → train → direct-action rollout → CUDA evaluation → RRD/MP4
report path. FastWAM retains video world modeling during training but calls
`select_action` at inference; it is not the Cosmos3 FastWAM-K2 experiment.
The image must keep FastWAM, Wan, UMT5, data, checkpoints and outputs runtime
side, pin their revisions in run provenance, and record the dataset's own
terms. Do not add a FastWAM-specific EULA or acceptance environment variable.

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
npa workbench lerobot deploy --runtime container --image <reviewed-image@sha256:digest>
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

For newly simulated robot demonstration data, use the Token Factory robot SDG
path documented in `docs/workbench/token-factory-robot-sdg.md`. It runs real Fetch
pick-and-place in MuJoCo, records synchronized workspace/wrist RGB, joint states
and Cartesian/gripper actions, and exports physics-accepted episodes to LeRobot
v3. Native-reader validation must check language task labels as well as video
decoding and action alignment. The scripted teacher uses simulator state; this
does not prove physical robot transfer or learned-policy performance.

For a reproducible demonstration-first transfer experiment, use
`workflows/testing/lerobot-transfer.yaml` and
`docs/workbench/guides/lerobot-transfer.md`. The standard runtime owns four waves:
prepare, paired ACT training, paired native PushT evaluation, and reporting.
The recipe pins LeRobot 0.6.0 and public PushT data, excludes reserved episodes from training and
normalization, and disables affine image augmentation for absolute actions.
Select the arm using validation only; derive expert demonstration requests only
from validation failures. Report held-out success uncertainty without claiming
physical robot transfer. The default uses B200 for training and evaluation;
PushT renders on CPU. No new image or
per-blueprint CLI is required; stage the exact source with `submit --stage-src`.

Input format is `LeRobotDataset` in Hugging Face format. Use the SimToLeRobot adapter to convert Genesis or other simulation outputs.

For manual System 1 subtask labels, load LeRobot v3 into FiftyOne 1.22, create
complete non-overlapping `subtask:<label>` temporal tags, and run
`npa workbench fiftyone export-lerobot-subtasks`. The derived dataset retains
the episode task instruction and adds per-frame `subtask_index`,
`meta/subtasks.parquet`, and resumable annotation metadata. FiftyOne is the
review UI; LeRobot remains the durable training format. The source dataset is
immutable. Run `workflows/testing/lerobot-subtask-proof.yaml` after export when
the result needs a reproducible coverage gate and a row-level LeRobot proof.

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
- FastWAM: the rebuilt local 0.6.0 image constructs `FastWAMConfig`, exposes
  `policy.type=fastwam`, and has complete built-byte/no-baked-model evidence;
  this is not B200 or end-to-end workflow evidence. Require independent GPU,
  private pushed-digest, data/simulator, MP4/RRD, and measured-latency evidence
  before a FastWAM live-ready claim.
