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
| Paired physical descriptions and negative conditioning | Original NPA `prepare` → native Diffusers Wan 2.1 generation; two RTX PRO 6000 GPUs | [Qualified](physis-lang-validation.md): all 36 full native videos and 2,916 decoded frames verified |
| Blinded physical-assertion comparison | Original NPA `evaluate`; CPU decoding plus hosted vision inference | [Qualified](physis-lang-validation.md): 108 verdicts, raw responses, paired scores, hashes, and original-video gallery verified |
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

The [full GPU qualification](physis-lang-validation.md) measured assertion pass
rates of 72.2% for baseline, 77.8% for physical descriptions, and 72.2% with
negative guidance. Keep the method experimental; this small comparison does not
establish a reliable physics improvement or reproduce the paper's scores.

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
`index.html`, and all 36 original videos. The HTML embeds every original MP4,
so downloading only `index.html` is sufficient for offline playback; no server,
network, or sibling files are needed. It groups the three prompt arms by scenario
and matched seed, with measured scores, assertion evidence, prompts, native GPU
settings, and SHA-256 receipts. Embedding verifies each clip against both its
generation and evaluation receipt. Separate MP4s remain available for analysis.
Each stage has a SHA-256 manifest;
publication is verified by full storage readback. Completed stages record their
invocation in `stage.json`; retrying the same stage verifies and reuses its
completed destination before running a model. Changed requests, conflicting
bytes, and incomplete destinations fail before expensive work.

Publication reserves the expected checksum manifest and creates objects with
conditional writes. Repeating publication can fill in missing objects with
identical bytes, but cannot replace another result. Failed stages preserve
`failure.json` and the original `output/` under a separate
`-failed/<attempt>/` prefix and return the original failing status. Rejected
judge responses are saved before response validation.

If storage also fails, the worker retains its private working directory and
prints its location. Copy it before deleting the worker; worker-local retention
does not survive pod or disk deletion. Once storage is available, publish a
completed stage's retained `output/` without rerunning generation or judging:

```bash
npa/.venv/bin/python -m npa.workflows.physis_lang publish \
  --input-path "$RETAINED_OUTPUT" --output-path "$ORIGINAL_OUTPUT_URI"
```

`RETAINED_OUTPUT` can be the local `failure/output/` directory or the S3
`-failed/<attempt>/output/` prefix printed in the failure report. Recovery requires
a verified checksum manifest and the original completed `stage.json`; partial
model execution cannot be published as completed work. The destination must
match the original stage request. Then resume the workflow through its normal
runtime command. Local retained directories can be removed after verified
publication. Failed evidence is retained separately for inspection.

If a worker or its disk is lost during publication and no complete retained
copy survives, start a new workflow with a fresh run ID and output prefix.
Keep the original partial prefix as failure evidence. Its checksum manifest
reserves that result; removing the manifest cannot recover missing video bytes.

No Physis-Lang source, paper figures, benchmark data, or trained checkpoint is
redistributed. The scenario text and adapter code are original NPA work. Wan
and Diffusers retain their upstream Apache-2.0 terms; runtime delivery retains
the existing [Diffusers redistribution contract](../../npa/docker/workbench/diffusers/REDISTRIBUTION.md).
GPU generation validity, VLM assertion scores, and upstream reproduction are
separate claims. See the [readiness record](../../workflows/testing/physis-lang.readiness.json)
for checked prerequisites and current validation evidence.
