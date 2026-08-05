---
name: wan2-2
description: Use when packaging, running, reviewing, or extending the Alibaba Wan 2.2 TI2V-5B BYOF solution, its video artifacts, or the Bellboy real-robot workflow boundary.
---

# Wan 2.2 BYOF and Bellboy workflow

Use this skill for the public Wan 2.2 registry candidate and the Bellboy-shaped
episode/video workflow. Read these files before changing behavior:

- `npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml`
- `npa/workflows/workbench/npa-workflows/bellboy-wan2.2-e2e.yaml`
- `npa/src/npa/workflows/bellboy_wan.py`
- `docs/workbench/wan2.2-bellboy.md`
- `docs/workbench/bellboy-episode-manifest-v1.schema.json`

Also load `byof-onboard`, `oss-solution-registry-onboard`,
`author-npa-workflow`, `real-components`, `solution-licensing`, `gpu-selection`,
`nebius-infra`, `testing-conventions`, and `agent-visual-feedback` when their
surfaces are involved.

## Ground truth

- Official source: `https://github.com/Wan-Video/Wan2.2.git`, pinned to
  `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.
- Official model: `Wan-AI/Wan2.2-TI2V-5B`, pinned to
  `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`.
- TI2V-5B is a stock generative-video model supporting text and image inputs.
- The accepted-candidate hard gate is text-to-video plus decoded MP4
  validation. Run `byof-wan22-e2e-20260805T191659Z` satisfied it on a real RTX
  PRO 6000 Blackwell (`sm_120`).
- I2V, A14B, speech-to-video, Animate, and training are separate capabilities.
  Do not infer acceptance from the T2V smoke.
- Stock Wan does not predict robot actions. Bellboy's action head and
  action-conditioned model are a private extension, not an upstream claim.
- The Cosmos3 image's run-time Wan VAE download is not a full Wan integration.

For changing facts, use only the official Wan repository, official Wan-AI model
cards, and primary framework documentation.

## Packaging contract

Use `workbench.byof.repo` and `solution-smoke`; do not add a fake Wan toolRef.
Keep the repo and all model inputs immutable. The image may contain pinned
source and dependencies but no checkpoint weights, credentials, customer data,
or private code. The final runtime must remain non-root, with `/opt/byof` and
its venv world-readable/executable.

Route the checked-in baseline to one RTX PRO 6000 Blackwell Server Edition
(`sm_120`). It uses the official PyTorch 2.7.1 CUDA 12.8 wheel line, asserts
that `torch.cuda.get_arch_list()` contains `sm_120`, and executes the upstream
PyTorch SDPA fallback rather than FlashAttention. Record the observed device,
compute capability, driver, CUDA, torch version, arch list, and a finite SDPA
probe in both runtime evidence artifacts.

The smoke must call the real native `wan.WanTI2V` generator and write:

- `wan2_2_ti2v_5b.mp4`
- `wan2_2_ti2v_5b_text_to_video.json`
- `wan2_2_runtime_inventory.json`

It must decode every frame and fail on invalid dimensions/count/fps, a corrupt
or empty container, an implausibly small file, or uniform/blank content. Keep
`capabilities_exercised` exact and `deferred` empty for the hard-gated T2V run.
If `context_image_uri` is set, also change the declared capability, named smoke
artifact, and workflow output URI to their image-to-video values. The smoke
must fail closed when the input mode and declarations disagree.

All files under `$NPA_SMOKE_OUTPUT_DIR` are uploaded by the existing BYOF
runner. The agent artifact browser renders the `.mp4` in its video viewer; do
not add an RRD unless there are synchronized comparison streams.

## Bellboy contract

`npa.bellboy.episode_manifest.v1` references customer S3 objects for:

- gripper-mounted wide-angle RGB video/frames and timestamps;
- executed actions and timestamps under an explicit action schema;
- joint state and timestamps;
- task, timing, outcome, failure, and corrective-retry lineage;
- train/validation/heldout split.

The public workflow may record the manifest as lineage and use an optional
context image for stock Wan. It must not claim that stock Wan consumes action
sequences, trains on episodes, or evaluates robot task success.

The held-out boundary report intentionally leaves the action release gate
unsatisfied. A customer extension requires a private repo URL and immutable
ref, real entrypoint/smoke, exact action schema, authorized data, immutable
checkpoint URI, action-prediction artifact, and held-out real-episode evaluator.

## Capability status

| Capability | Status |
| --- | --- |
| `wan2.2_ti2v_5b_text_to_video` | accepted; live validated on RTX PRO 6000 Blackwell by `byof-wan22-e2e-20260805T191659Z` |
| `wan2.2_decoded_mp4_validation` | accepted; same run decoded 17 1280x704 frames at 24 fps and passed non-uniform-content gates |
| `wan2.2_ti2v_5b_image_to_video` | deferred |
| A14B / S2V / Animate | deferred |
| official TI2V fine-tuning | deferred; no pinned-source training entrypoint |
| stock Wan action prediction | rejected as an upstream capability |
| Bellboy private action prediction | deferred customer extension |

The accepted T2V/MP4 capabilities met the pushed-image
NPA/SkyPilot/Kubernetes gate and have their named JSON, runtime inventory, and
decoded MP4 evidence in S3. Do not infer acceptance for any deferred capability
or treat capability acceptance alone as authorization for public publication.

## Licensing

Track four layers separately: official source, baked OS/dependencies, run-time
model/tokenizer, and customer data/private checkpoints. Apache-2.0 declarations
for source/model do not classify the built image. Before publication, inspect
the S3 runtime inventory emitted from inside the pulled image, classify the
actual installed package set, and scan for unexpected restricted payloads. The
inventory's large-checkpoint scan must be empty. Do not add this dynamic BYOF
candidate to the first-class packaging contract before image promotion.

## Validation

Use the repository venv, never bare Python:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml --run-id wan22-plan
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/bellboy-wan2.2-e2e.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/bellboy-wan2.2-e2e.yaml --run-id bellboy-wan22-plan
npa/.venv/bin/python -m pytest npa/tests/workflows/test_bellboy_wan.py -q
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q
```

The live test is `npa/tests/e2e/test_byof_wan22_live_e2e.py`. Run it only through
its explicit operator gate. The recorded acceptance run is
`byof-wan22-e2e-20260805T191659Z`; future compatibility changes require fresh
live evidence rather than inference from that run.
