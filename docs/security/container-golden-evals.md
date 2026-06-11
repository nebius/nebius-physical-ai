# Container safety review & golden evals

This document is the safety + Physical AI usefulness review for every Workbench
container image, and the contract for each container's **golden eval** — the
minimal "does this container actually work" / "hello world" tested rerun.

The machine-readable source of truth is
[`npa/src/npa/smoke/golden_evals.yaml`](../../npa/src/npa/smoke/golden_evals.yaml).
This doc is the human-readable companion; if the two disagree, the manifest
(which is test-enforced) wins.

## How it is enforced

- **Completeness/consistency gate** —
  `npa/tests/smoke/test_golden_eval_manifest.py` runs in the standard unit suite
  and fails CI if any container in `npa.deploy.images.CONTAINER_IMAGE_NAMES` is
  missing an entry, references a missing Dockerfile, an unimportable smoke
  module, or omits a safety / Physical AI field.
- **Nightly run** — the workflow at `docs/ci/golden-evals-nightly.yml` runs at
  04:00 UTC once installed to `.github/workflows/` (it ships under `docs/ci/`
  because the author credential lacked the GitHub `workflow` scope). The
  `validate-manifest` and `cpu-evals` jobs run on GitHub-hosted runners; the GPU
  golden evals run on a self-hosted GPU runner via `workflow_dispatch`
  (`run_gpu_evals: true`).
- **Image CVE / config scanning** — handled separately by the weekly
  `image-security-scan.yml` (Trivy config scan + base-image CVE matrix).

## CLI

```bash
npa workbench golden-eval list              # table of every container + eval
npa workbench golden-eval show lerobot      # full safety + eval record (JSON)
npa workbench golden-eval validate          # offline completeness/consistency
npa workbench golden-eval run cosmos        # print the eval command (dry run)
npa workbench golden-eval run cosmos --execute   # run it (needs the runtime)
```

The same logic is available as a script for CI:
`python npa/scripts/run_golden_evals.py {validate,list,run}`.

## Golden-eval kinds

| kind | meaning |
| --- | --- |
| `container-smoke` | env + functional smoke module run inside the built image |
| `server-smoke` | start the FastAPI service, poll `/health`, do one real op |
| `entrypoint-smoke` | container ENTRYPOINT mode that self-reports a result artifact |
| `workflow-smoke` | workflow/CLI entrypoint contract (help/parse) proof |
| `build-import` | import/compile proof that the heavy deps resolve |

`status` is one of `ready` (runs on a normal runner), `gpu-gated` (needs a GPU
host with the image), or `blocked-on-upstream` (B300/CUDA13 family).

## Summary: safety + Physical AI usefulness

All shipped containers are assessed as **useful for Physical AI**; each is a
distinct stage of the robotics / simulation / perception / synthetic-data
pipeline. Key safety notes are condensed below.

| container | Physical AI role | golden eval | gpu | status |
| --- | --- | --- | --- | --- |
| `base-cuda13-b300` | CUDA13/PyTorch foundation for B300 derivatives | `build-import` | required | blocked-on-upstream |
| `groot` | Isaac-GR00T foundation-model deploy/inference | `container-smoke` | required | gpu-gated |
| `lerobot` | LeRobot policy train/eval/serve | `container-smoke` | required | gpu-gated |
| `lerobot-policy` | sim-to-real policy stage (serve/train/eval) | `build-import` | optional | gpu-gated |
| `lerobot-vlm-rl` | VLM-reward RL step for sim-to-real | `workflow-smoke` | optional | gpu-gated |
| `genesis` | Genesis physics sim + RL teacher + demos | `container-smoke` | required | gpu-gated |
| `isaac-lab` | Isaac Lab RL sim (headless train/eval) | `container-smoke` | required | gpu-gated |
| `cosmos` | Cosmos world-model serving (text2world) | `container-smoke` | required | gpu-gated |
| `cosmos2-transfer` | Cosmos-Transfer2 video-to-video for synthetic data | `build-import` | required | gpu-gated |
| `cosmos3-reason` | Cosmos-Reason1 VLM reasoning stage | `workflow-smoke` | optional | blocked-on-upstream |
| `sonic` | SONIC whole-body humanoid locomotion | `entrypoint-smoke` | required | gpu-gated |
| `retargeting` | CPU motion retargeting for SONIC locomotion | `build-import` | none | ready |
| `fiftyone` | dataset curation/visualization (CPU) | `container-smoke` | none | ready |
| `lancedb` | vector store for AV/perception data | `server-smoke` | optional | ready |
| `detection-training` | object-detection train/eval service | `server-smoke` | optional | ready |
| `sim2real-envgen` | randomized Genesis env generation | `workflow-smoke` | optional | gpu-gated |
| `sim2real-reference-policy` | reference policy contract | `workflow-smoke` | optional | gpu-gated |
| `sim2real-eval` | sim-to-real full-loop evaluation | `workflow-smoke` | optional | gpu-gated |

## Safety review highlights

- **Runtime user** — npa-built images (`groot`, `lerobot*`, `genesis`, `cosmos`,
  `cosmos3-reason`, `fiftyone`, `sim2real-*`) run as the unprivileged `ubuntu`
  user. `isaac-lab` and `sonic` inherit `root` from the `nvcr.io/nvidia/isaac-lab`
  base; `lancedb` and `detection-training` run as `root` from the PyTorch base.
  These are candidates for a non-root hardening pass.
- **Network exposure** — services that open ports (`lerobot` :8080, `cosmos`
  :8080, `lancedb` :8686, `detection-training` :8790, `fiftyone` :5151) must be
  deployed in the `workbench` namespace behind controlled access, never bound to
  public ingress without auth. `lancedb` and `detection-training` ship a token
  auth mode and warn loudly when started with `auth_mode=none` (the golden eval
  uses `none` against a throwaway store/port only).
- **Content safety** — `cosmos` ships a content-safety guardrail.
  `COSMOS_DISABLE_SAFETY` must remain `"0"` in production; the functional smoke
  keeps safety enabled by default.
- **External fetches** — `isaac-lab` and `sonic` pull from `nvcr.io` (NGC auth
  required); `groot`/`sonic` clone pinned Git refs; several images fetch from
  Hugging Face. Base images are digest-pinned and tracked by the weekly Trivy
  CVE scan.
- **B300 / CUDA13 family** — `base-cuda13-b300` and its derivatives
  (`cosmos3-reason`) are `blocked-on-upstream` (Taichi sm_103, flash-attn
  Blackwell wheels, CUDA 13 host driver >= 580); their golden evals are defined
  but expected to remain gated until upstream lands.

## Per-container golden eval commands

Run these inside the corresponding built image (or via
`npa workbench golden-eval run <name> --execute` on a host with the runtime):

- `groot` — `python -m npa.smoke.test_groot_functional` (env: `test_groot_env`)
- `lerobot` — `python -m npa.smoke.test_lerobot_functional` (env: `test_lerobot_env`)
- `lerobot-policy` — `python -m npa.workbench.lerobot.policy_container check-import`
- `lerobot-vlm-rl` — `python -m npa.workbench.lerobot.policy_container vlm-signal-step --help`
- `genesis` — `python -m npa.smoke.test_genesis_functional` (env: `test_genesis_env`)
- `isaac-lab` — `python -m npa.smoke.test_isaac_lab_functional` (env: `test_isaac_lab_env`)
- `cosmos` — `python -m npa.smoke.test_cosmos_functional` (env: `test_cosmos_env`)
- `cosmos2-transfer` — `python -c "import npa.workbench.cosmos2"` (image built outside this repo)
- `cosmos3-reason` — `python -m npa.workflows.sim2real_loop inner-loop --help`
- `sonic` — `/entrypoint.sh smoke` (artifact: `sonic_smoke_result.json`)
- `retargeting` — `python -c "import npa.workbench.retargeting"`
- `fiftyone` — `python -m npa.smoke.test_fiftyone_functional` (env: `test_fiftyone_env`)
- `lancedb` — `python -m npa.smoke.test_lancedb_functional`
- `detection-training` — `python -m npa.smoke.test_detection_training_functional`
- `sim2real-envgen` — `python -m npa.workflows.sim2real_envgen --help`
- `sim2real-reference-policy` — `python -m npa.workflows.sim2real_envgen policy-contract --help`
- `sim2real-eval` — `python -m npa.workflows.sim2real_loop full-loop --help`
- `base-cuda13-b300` — `python -c "import torch; assert torch.cuda.is_available(); import flash_attn"`

## Adding a new container

1. Add the Dockerfile under `npa/docker/workbench/<tool>/`.
2. Register the image in `npa.deploy.images.CONTAINER_IMAGE_NAMES` and pin the
   version in `pyproject.toml [tool.npa.supported-tools]`.
3. Add a `golden_evals.yaml` entry with `physical_ai`, `safety`, and
   `golden_eval` blocks. The unit gate will fail until it is present and valid.
4. Provide the smoke entrypoint the golden eval references (a
   `npa.smoke.test_<tool>_functional` module, a server smoke, or an entrypoint
   mode).
