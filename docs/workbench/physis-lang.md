# Physis-Lang physical prompting experiment

[Workbench](README.md) · [Workflow](../../workflows/testing/physis-lang.yaml)

Run a paired physical-prompting experiment on two Nebius RTX PRO 6000 GPUs.
The workflow prepares descriptions through Token Factory, generates real Wan
2.1 14B videos, checks every decoded frame, and publishes a comparison gallery
with blinded vision-model judgments.

This is an **independent NPA implementation inspired by the Physis-Lang paper**.
The [official repository](https://github.com/Physis-Intelligence/Physis-Lang)
at `294121a06fa20bbca6bdb644bbae7229d5c88160` contains the project description
and paper assets, but no implementation. The workflow implements an inference
prompting experiment related to [the paper](https://arxiv.org/abs/2609.40358).
It does not implement PhysThinker, PhysCapBench, self-evolving guidelines,
retrieval, or supervised fine-tuning, and it does not reproduce paper scores.

| Capability | Implementation and runtime | Evidence / status |
| --- | --- | --- |
| Paired physical descriptions and negative conditioning | Original NPA `prepare` → native Diffusers Wan 2.1 generation; two RTX PRO 6000 GPUs | Frozen recipe, native generation receipts, and original MP4s; qualification pending |
| Blinded physical-assertion comparison | Original NPA `evaluate`; CPU decoding plus hosted vision inference | Exact paired coverage, per-assertion verdicts, hashes, and gallery; qualification pending |
| PhysThinker and self-evolving physical guidelines | Upstream implementation/checkpoints are unreleased at the recorded revision | Deferred; no executable upstream API to onboard |
| PhysCapBench, retrieval, and fine-tuning | Upstream benchmark/training implementation and artifacts are unreleased at the recorded revision | Deferred; no benchmark or training claim |

## Experiment

Six original scenarios cover domino contact, ramp rolling, water pouring, cloth
draping, block pushing, and sponge compression. Each has three assertions fixed
before generation. Two seeds run each scenario through three arms:

| Arm | Positive conditioning | Negative conditioning |
| --- | --- | --- |
| `baseline` | Original scenario | Empty |
| `physics` | Original scenario plus generated physical description | Empty |
| `physics-negative` | Same positive prompt as `physics` | Generated physical errors to avoid |

The native Wan pipeline uses BF16 weights, its FP32 VAE, UniPC with flow shift
5, guidance 5, 50 steps, 81 frames at 16 fps, and 1280×720 output. Each GPU
loads one resident model and executes 18 requests in a seeded shuffled order.
All arms for a scenario share the same generator seed. Oversized text fails
before generation instead of silently truncating at the text encoder boundary.

The evaluator verifies the sealed recipe, exact comparison coverage, native
conditioning and generation settings, video hashes, dimensions, frame count,
frame rate, and nonblank moving content. It then sends 16 chronological frames
per video to the vision model. The judge sees only the original scenario and
assertions; it receives neither the arm label nor the expanded prompt.
Unknown assertions count as unproven, not passing. Negative experimental results
are preserved. These sampled-frame judgments cannot establish unobserved contact
or full physical correctness; review the original videos in the gallery.

## Run

Use a Linux operator host with the supported isolated SkyPilot runtime. Select
the intended project and one exact kubeconfig, then follow
[SkyPilot setup](../orchestration/skypilot-setup.md). Confirm storage and hosted
model credentials with `npa workbench health preflight --project PROJECT
--checks nebius,s3,token_factory --json`.

The public checkpoint is `Wan-AI/Wan2.1-T2V-14B-Diffusers`, pinned to
`38ec498cb3208fb688890f8cc7e94ede2cbd7f68`. Verify access to actual weight
payloads at that revision before allocating GPUs. Weights and the hash-locked
CUDA runtime are fetched at execution time. The workflow reuses the accepted
`npa-diffusers` digest in its YAML; it creates no new container image.

```bash
npa workbench workflow validate-spec workflows/testing/physis-lang.yaml
npa workbench workflow plan-spec workflows/testing/physis-lang.yaml \
  --run-id physis-comparison --check-render --json
npa workbench workflow submit workflows/testing/physis-lang.yaml \
  --run-id physis-comparison --project PROJECT --infra k8s/CONTEXT \
  --isolated-config-dir "$HOME/.npa/physis-comparison" \
  --var bucket=YOUR_BUCKET --stage-src --runtime --max-wait-seconds 0 \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`KUBECONFIG` selects the verified kubeconfig. `NPA_SKYPILOT_BIN` selects the
pinned SkyPilot executable. Keep credentials in supported private configuration,
never in workflow YAML. The source overlay carries the checked-out NPA code;
the image's `/opt/wan-base/bin/python` supplies the reviewed CPU dependencies,
and `model-runtime ensure` prepares the native GPU interpreter.

| Configuration | Default / requirement |
| --- | --- |
| `bucket` | Required operator-owned output bucket; shipped value is a placeholder |
| `prefix` | `physis-lang/{{run.id}}`; use a fresh run prefix |
| `source_overlay` | `true`; required for this source implementation |
| `seed_a`, `seed_b` | `0`, `1`; distinct integers in `[0, 2**63)` |
| `prompt_model` | `nvidia/Nemotron-3_5-Lightning` |
| `judge_model` | `MiniMaxAI/MiniMax-M3`; must support images |
| `runtime_image` | Immutable accepted Diffusers image in the YAML |
| `prepared_uri`, `shard_a_uri`, `shard_b_uri`, `report_uri` | Derived run-scoped S3 directories |

Changing a model requires checking its actual endpoint access and response
contract. A completed response must identify the requested model and contain
valid JSON; malformed responses are not repaired into success.

## Artifacts and scope

`prepared/` stores the frozen recipe and original model responses. `seed-a/`
and `seed-b/` store every original MP4, native generation receipts, and logs.
`report/` contains `report.json`, `judgments.json`, raw judge responses,
`index.html`, and all 36 original videos. Each stage has a SHA-256 manifest;
publication is verified by full storage readback. Failed stages use a separate
`-failed/` prefix and return a failing status.

No Physis-Lang source, paper figures, benchmark data, or trained checkpoint is
redistributed. The scenario text and adapter code are original NPA work. Wan
and Diffusers retain their upstream Apache-2.0 terms; runtime delivery retains
the existing [Diffusers redistribution contract](../../npa/docker/workbench/diffusers/REDISTRIBUTION.md).
GPU generation validity, VLM assertion scores, and upstream reproduction are
separate claims. See the [readiness record](../../workflows/testing/physis-lang.readiness.json)
for checked prerequisites and current validation evidence.
