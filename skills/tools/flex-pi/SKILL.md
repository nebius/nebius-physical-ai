---
name: flex-pi
description: Run, package, validate, or troubleshoot flex-pi world-action policy inference in NPA, including its runtime-fetch assets, RTX PRO 6000 workflow, and action artifacts.
---

# flex-pi

Run the genuine upstream 6B world-action checkpoint. Do not replace policy
inference with an import check, random action fixture, or manifest-only smoke.

## Boundaries

- Source: `geyan21/flex-pi@20c1b2b71ea35a415d5d47c39b04443cfadad7a1`,
  MIT, baked without the vendored simulators.
- Checkpoint: `flex-pi/flexpi-robotwin@87d3833ea3bd89c4922945631db81b346e780785`,
  MIT-labelled, runtime fetch.
- Observation: `flex-pi/robotwin_3d@bee164afe94041d8c3d7dd1203725c41163fc3f4`,
  episode 0 frame 0. Its dataset card declares no license. Use only under the
  operator's authorized scope, fetch at runtime, verify each byte hash, and do
  not publish the media.
- DINOv3 uses its separate model license. The pinned converted Wan VAE/T5 and
  tokenizer repositories are Apache-2.0. They are operator runtime fetches;
  immutable revisions and heavyweight-file hashes are enforced before loading.
- The public image must contain none of those weights, media, credentials, or
  populated caches. Run `npa/scripts/scan_image_flex_pi_payload.py` against the
  built image before any registry write.

## Choose the regime and GPU

Use the released action-only regime for policy-serving validation: video and
DINO are observed, pointmap is absent, and all future-stream joint-denoising
flags are false. It produces the same 32-step, 14D bimanual action contract as
upstream evaluation without pretending an isolated observation proves simulator
success. The upstream model card reports equal average RoboTwin success for
action-only and full-joint inference, while action-only is substantially faster.

Route the reference workflow to one RTX PRO 6000. It is headless and does not
need RT cores, but this target proves the required CUDA `sm_120` path. A B200 run
does not substitute for RTX evidence.

## Preflight and run

Before build, provisioning, or submission, prove Nebius and selected-project S3
credentials with `npa workbench health preflight --checks nebius,s3 --json`.
Probe the exact anonymous/public model and input objects before allocating a GPU.
Do not treat a generic Hugging Face token check as proof of those payloads.

Validate and plan the reference spec, then submit it with the selected project's
bucket and exact immutable image:

```bash
npa workbench workflow validate-spec workflows/testing/flex-pi-rtxpro-inference.yaml
npa workbench workflow plan-spec workflows/testing/flex-pi-rtxpro-inference.yaml
npa workbench workflow submit workflows/testing/flex-pi-rtxpro-inference.yaml \
  --infra "$CONFIGURED_TARGET" --var "bucket=$OPERATOR_BUCKET" \
  --secret-env HF_TOKEN
```

The cache may be run-owned and persistent across retries. Never place it in the
image build context or upload it as evidence. The checkpoint is public and the
token is not an access acceptance; forward an operator read token through this
secret-only channel to avoid anonymous rate limits during its multi-shard fetch.
The image defaults `MODELSCOPE_DOWNLOAD_PARALLELS=16`, the SDK's supported
maximum, because the converted UMT5 asset is one roughly 11 GB object. Retain
that default for normal cold starts; an operator may lower it for a constrained
network.

## Artifacts and acceptance

Require all of the following from the exact image digest:

- terminal workflow/job success on one RTX PRO 6000;
- `actions.json` with schema `npa.flex_pi.actions.v1`, action-only regime, and
  exactly 32 rows of 14 finite numbers;
- positive inference latency and peak allocated GPU memory;
- CUDA device name matching RTX PRO 6000 and compute capability 12.0;
- pinned source/checkpoint/dataset identity plus checkpoint, stats, and input
  SHA-256 provenance;
- non-empty `result.json` and read-after-write verification in operator storage.

Keep raw logs and object locations access-controlled. Repository and PR evidence
may report only sanitized status, metrics, hashes, and generic GPU class.

Cancel the exact run before removing only run-owned infrastructure. Never tear
down a shared or pre-existing cluster.

## Troubleshoot

- ModelScope VAE/T5 fetch failures: this repository exposes only `master` to the
  SDK. Do not pass its raw commit as `revision`; require `master` to match the
  pinned Git commit, then enforce the two maintained SHA-256 file digests.
- Missing DINO weights: confirm access to the timm DINOv3 repository and its
  model terms before retrying.
- CUDA OOM: verify action-only flags and absence of competing workloads. Do not
  silently reduce cameras, horizon, action dimensions, or checkpoint fidelity.
- GPU mismatch: inspect scheduler labels and the artifact's device name. Do not
  infer RTX support from a B200 result.
- Invalid actions: retain the failed artifact privately and diagnose upstream;
  never coerce, clip, or fabricate values to pass validation.

## Verify changes

```bash
npa/.venv/bin/python /home/ubuntu/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/tools/flex-pi
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_flex_pi.py \
  npa/tests/docker/test_flex_pi_image_contract.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/guardrails/test_three_tier_contract.py \
  npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py -q
```
