# Cosmos3-Super serving on the workbench (`npa-cosmos3-serving`)

`npa workbench cosmos3 generate` runs Cosmos 3 generation as a batch job: one
invocation, one artifact, and a full model load every time. For the 64B
`Cosmos3-Super` checkpoint that load costs minutes, so a synthetic-data
workload that wants many clips pays it over and over.

This image is the serving half. It loads `nvidia/Cosmos3-Super` once and serves
an OpenAI-compatible endpoint against resident weights, on one 8-GPU node.

| Piece | Path |
| --- | --- |
| Image | `npa/docker/workbench/cosmos3-serving/Dockerfile` (`npa-cosmos3-serving`) |
| Entrypoint | `npa/docker/workbench/cosmos3-serving/entrypoint.sh` |
| Build gate | `npa/docker/workbench/cosmos3-serving/verify_env.py` |
| Build script | `npa/docker/workbench/cosmos3-serving/build.sh` |
| Image contract tests | `npa/tests/docker/test_cosmos3_serving_image_contract.py` |

The image wraps NVIDIA's own `vllm/vllm-omni` serving runtime, pinned by digest,
and adds a preflight entrypoint and a build gate.

## What this is not, yet

The image is **not registered as a deployable workbench tool**. There is no
`npa workbench cosmos3 serve` CLI verb, no SDK surface, no workflow integration,
and no entry in the packaging contract, the datacenter Blackwell verdict
manifest, `CONTAINER_IMAGE_NAMES`, or the golden-eval manifest.

That is a deliberate stopping point rather than an omission, because those
registrations are one chain rather than four independent edits: a packaging
contract entry obliges a Blackwell verdict, which obliges a
`CONTAINER_IMAGE_NAMES` key and a supported-tools version pin, which obliges a
golden-eval manifest entry backed by a real capability smoke that needs an
8-GPU node to run. Build it as one deliberate change, with a version scheme and
a smoke tier chosen on purpose, or not at all.

## Weights are never in the image

The base runtime carries no checkpoint bytes and this layer adds none. The
`Cosmos3-Super` checkpoint, the guardrail models, and every other gated artifact
download **at run time** with the operator's own Hugging Face token, under that
operator's own license acceptance. That is the same posture `npa-cosmos3` holds,
and the reason the built image carries no gated weight bytes to redistribute.
Redistribution of the base runtime itself follows its own upstream license; this
layer adds only the entrypoint, the build gate, and configuration.

`verify_env.py` runs at build time and fails the build if a weight file larger
than 50 MB landed in any layer, the same guarantee `npa-cosmos3` enforces.

## Access preflight

Two access gates fail confusingly on a clean host. Both are covered in full by
`docs/workbench/cosmos3-access-preflight.md`, which is added by
nebius/nebius-physical-ai#264.

**The serving path pulls a different gated guardrail repo than the batch path.**
`npa workbench cosmos3 generate` pulls `nvidia/Cosmos-Guardrail1`. This image
pulls `nvidia/Cosmos-1.0-Guardrail`. They are separate Hugging Face repos with
separate license acceptances, and clearing one does not clear the other. The
entrypoint refuses to start when guardrails are on and no token is present,
because without that check the fetch goes out anonymous and dies with a `401`
several minutes into startup, which reads like a bad token rather than a missing
one. If a token **is** set and the download still fails, use an authenticated
probe: authenticated `401` means the supplied token is missing, invalid, or
revoked; authenticated `403` means the identity is valid but unauthorized
because of repo approval/license, fine-grained scope, or organization policy.
An anonymous `401` is not a token discriminator.

**The base image's pinned Hugging Face client pair breaks the guardrail
download.** `HF_HUB_DISABLE_XET=1` is set in the image, which is a departure
from `npa workbench cosmos3 generate`, where the same condition only produces a
warning. The difference is ownership of the pins. That path runs in whatever
environment the operator built, so setting the variable on their behalf would
silently disable a faster download path for environments that do not need it.
This Dockerfile chose its own base image, and that base pins `hf-xet` 1.5.1 with
`huggingface_hub` 1.23.0, the exact pair
[huggingface/xet-core#895](https://github.com/huggingface/xet-core/issues/895)
reproduces on. `verify_env.py` re-reads the installed pair at build time and
fails the build once a base bump moves off it, naming the line to delete, so the
workaround cannot outlive the defect it works around.

## Build

No GPU is needed to build.

```bash
bash npa/docker/workbench/cosmos3-serving/build.sh --registry <your-registry>
bash npa/docker/workbench/cosmos3-serving/build.sh --registry <your-registry> --push
```

The build runs `verify_env.py` inside the image, which checks four things: the
vLLM-Omni serving stack imports, the xet workaround still matches the installed
pins, the entrypoint assembles the serve command it advertises, and no weight
file is present in a layer. The third check executes the real entrypoint in
dry-run mode rather than restating its expected argv, so a change to the script
that breaks the serve command fails the build rather than the first 8-GPU boot.

## Run

```bash
docker run -d --name cosmos3-serving --gpus all --ipc=host --shm-size 32g \
  --ulimit nofile=1048576:1048576 \
  -v <hf-cache-dir>:/opt/npa-cosmos3-serving/hf-cache \
  -e HF_TOKEN=<your-token> \
  -p 8000:8000 \
  <your-registry>/npa-cosmos3-serving:<tag>
```

The mounted cache must be writable by uid 1000: the image runs as a non-root
user, and a cache mounted from a root-owned host directory is readable but not
writable. The entrypoint checks this before any download starts, because
otherwise the failure surfaces as a lock error partway through fetching 17 GB of
guardrail weights.

Budget the cache before first start: the `Cosmos3-Super` checkpoint is about
124 GB on disk plus 17 GB of guardrail weights, so plan roughly 145 GB. The
first-ever download is its own cost, separate from and larger than the
readiness figures below, which were measured with weights already on disk.

`--ipc=host` and the `nofile` ulimit are the vendor runtime's requirements, not
this layer's. To run the vendor runtime bare, without this image's preflights,
the serve command the Dockerfile assembles works directly against
`vllm/vllm-omni:cosmos3`; the measured behavior on this page applies to both,
because they run the same server.

### Configuration surface

| Variable | Default | Effect |
| --- | --- | --- |
| `NPA_COSMOS3_SERVE_MODEL` | `nvidia/Cosmos3-Super` | Model to serve |
| `NPA_COSMOS3_SERVE_HOST` | `0.0.0.0` | Bind address |
| `NPA_COSMOS3_SERVE_PORT` | `8000` | Bind port |
| `NPA_COSMOS3_SERVE_GUARDRAILS` | `on` | `off` passes `--no-guardrails` and drops the token requirement |
| `NPA_COSMOS3_SERVE_INIT_TIMEOUT` | `1800` | Passed through as `--init-timeout` |
| `NPA_COSMOS3_SERVE_GPUS` | `8` | GPU count the preflight requires |
| `NPA_COSMOS3_SERVE_EXTRA_ARGS` | empty | Appended verbatim to the serve command |
| `NPA_COSMOS3_SERVE_DRY_RUN` | `0` | Print the serve command and exit |
| `NPA_COSMOS3_SERVE_SKIP_GPU_CHECK` | `0` | Bypass the GPU-count preflight |

The parallel configuration is **pinned**, not exposed:
`--cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8`. That
is the configuration the model card recommends for 8x H200, H100, or A100, and
measurement on 8x H200 backed it: it was the fastest of five strategies tried,
about 9% faster than plain `--tensor-parallel-size 8` at both 35 and 50 steps.

| Configuration | 35 steps | 50 steps |
| --- | --- | --- |
| Pinned config above | ~121 s | ~166 s |
| `--tensor-parallel-size 8` | ~131 s | ~181 s |

Server-reported generation times at 1280x720, 189 frames, guardrails off;
client wall time adds roughly 3 to 4 seconds. Every measured cell fits
`latency ~= 14.4s + steps x per_step` to within 0.3%: the ~14.4 s outside the
denoise loop is the same across every strategy tried, and the entire difference
between configurations is denoise-step rate (3.03 to 3.06 s/step for the pinned
config, 3.33 to 3.36 for tensor parallelism). The full measured block is in
[vllm-project/vllm-omni#5909](https://github.com/vllm-project/vllm-omni/pull/5909).
`NPA_COSMOS3_SERVE_EXTRA_ARGS` is the escape hatch for anyone who wants a
different strategy, and `NPA_COSMOS3_SERVE_GPUS` has to be set to match.

Generation parameters are **not** part of this surface. Steps, frame count,
resolution, seed, and per-request guardrail posture are request fields on
`/v1/videos`, not server flags, so they are chosen per call rather than per
server.

### Generate a clip

```bash
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  --form-string "prompt=<your prompt>" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "max_sequence_length=4096" \
  --form-string "seed=17" \
  -o clip.mp4
```

## Readiness takes minutes

This is the operational fact most likely to be got wrong. Measured on 8x H200,
HSDP-sharded configurations reached `Application startup complete` in roughly
280 to 290 seconds with the model weights already in the host page cache, and
about 320 seconds longer than that on a cold cache. `--init-timeout 1800` was
never approached. The 280 to 290 second figure is a property of HSDP-sharded
weight loading: the one non-HSDP strategy measured, `--tensor-parallel-size 8`
on a single boot, reached readiness in about 90 seconds warm, so anyone
overriding the parallel configuration through `NPA_COSMOS3_SERVE_EXTRA_ARGS`
should re-measure readiness rather than inherit these numbers.

This image's own validation run reached readiness in 592 seconds on a cold cache,
which is the number to plan against for a node that has just been started.

The image's `HEALTHCHECK` therefore carries a 20 minute `start-period`. A
readiness probe tuned for an ordinary web service reports a healthy boot as a
failure and restarts it into another one, which is a loop that never converges.
A probe tuned even for the warm figure would have failed the 592 second boot
above. Orchestrators driving this image need the same allowance.

Memory during that load, with `--hsdp-shard-size 8`: 17.3 GiB per GPU at model
load, and roughly 43 GiB per GPU resident while serving.

## Guardrails

Guardrails are **on** by default, matching the model card. Turning them off is
per server, through `NPA_COSMOS3_SERVE_GUARDRAILS=off`, and the request API also
accepts a per-request posture.

Two things worth knowing before choosing a default for a shared server.
Guardrails add roughly 20 seconds of one-time initialization and 17 GB of extra
weights on disk. Their per-request cost is content-dependent rather than a fixed
tax: measured overhead ranged from a few seconds on short randomized prompts to
about 15% on a richly described one at the same shape, so any single overhead
figure is scoped to the prompt it came from.

Guardrails also change the output bytes rather than only gating requests, so a
clip generated with them on is not the same clip generated with them off.

## Determinism is per server instance

Within one running server, output is bitwise reproducible at a fixed seed.
Across a restart it is not: the same seed and configuration produce a different
clip, at a magnitude comparable to the difference between two adjacent frames of
the same video (median 28 dB PSNR against the pre-restart clip, versus 27.9 dB
between adjacent frames within one clip). Same-seed comparisons are only valid
inside one server instance, which matters for any evaluation harness that
restarts servers between arms.

## Concurrency

Cosmos 3 models do not support step-level batching in vLLM-Omni
([vllm-project/vllm-omni#4340](https://github.com/vllm-project/vllm-omni/issues/4340)
lists Step Execution as unsupported for them). Concurrent requests queue and
serialize rather than sharing GPU work, so client-side queue depth and timeouts
are the design surface, not server-side batch tuning.

Measured at 1280x720, 189 frames, 35 steps, guardrails on:

| Concurrency | Throughput | Mean latency | Median latency | Peak memory |
| --- | --- | --- | --- | --- |
| 1 | 28.21 clips/hr | 127.6 s | 127.6 s | 38,562 MB |
| 2 | 30.43 clips/hr | 208.2 s | 230.1 s | 38,562 MB |
| 4 | 30.78 clips/hr | 382.0 s | 461.4 s | 38,564 MB |
| 8 | 30.96 clips/hr | 729.0 s | 923.3 s | 38,564 MB |

Peak memory holding flat within 2 MB across an eightfold increase in in-flight
work is the hard evidence for the no-batching design. Throughput gains from 1
to 8 concurrent requests are under 10% and nearly all collected at concurrency
2, while median latency doubles at each step. There is little reason to run
this server above concurrency 2 unless the marginal 9.8% of throughput is worth
a 7.2x median-latency cost.

## Validated on real GPUs

Built and run on 8x H200 SXM (143,771 MiB each, driver 580.126.09), guardrails on,
against base digest `sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587`.

| Stage | Result |
| --- | --- |
| Build with no GPU | Succeeded; the build gate passed all four checks inside the image |
| Startup to `Application startup complete` | 592 s, cold page cache |
| One t2v request (1280x720, 189 frames, 24 fps, 35 steps, seed 17) | HTTP 200, server-side `stage_gen_time_ms` 138,721 |
| Clip verification | h264 1280x720 yuv420p, 24 fps, 189 frames read, 7.875 s, full decode clean, not blank |
| Memory while serving | 43.8 to 44.0 GiB per GPU on ranks 1 to 7; 50.3 GiB on rank 0, which also hosts the API server and the guardrail models |

The generation figure sits inside the band already measured for this
configuration and prompt on the same hardware without the container wrapper, so
the wrapper adds no measurable cost.

The base pin is not currently load-bearing for that cell: re-checked on the
vendor's `v0.26.0` image, the same flags produced 124 s against 123 s on the
pinned image (guardrails off, one run per image), with output parity inside the
same-configuration restart band.

All three preflight refusals were exercised against real conditions rather than
simulated ones: a missing token, a host cache genuinely owned by a different uid
than the container's runtime user, and four GPUs against the pinned 8-GPU
config. Each refused before the server started.

## Artifact handoff

The served endpoint returns bytes and keeps no record. `npa workbench cosmos3
generate` writes a `generate.json` manifest recording prompt, seed, guardrail
posture, and shape alongside each artifact; this server does not. A workload
that needs that provenance should record the request fields and the response's
sha256 on the client side and upload them alongside the clip.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `guardrails are on but no HF_TOKEN` | The preflight refusing to start a boot that would 401 several minutes in. Supply a token that has accepted `nvidia/Cosmos-1.0-Guardrail`, or set `NPA_COSMOS3_SERVE_GUARDRAILS=off`. |
| `Hugging Face cache ... is not writable` | The mounted cache is not writable by uid 1000. Chown it, or point `HF_HOME` somewhere writable. |
| `the pinned parallel config needs 8 GPUs, found N` | The pinned strategy is an 8-GPU decomposition. Override it through `NPA_COSMOS3_SERVE_EXTRA_ARGS` and set `NPA_COSMOS3_SERVE_GPUS` to match. |
| `Unable to parse string as hex hash value` | The xet defect the image already works around. Confirm `HF_HUB_DISABLE_XET=1` survived into the container environment. |
| Health check fails during startup | The probe is tighter than the real readiness window. See the readiness section above. |
| Authenticated `403` on a guardrail download | The token identity lacks authorization. Check acceptance for `nvidia/Cosmos-1.0-Guardrail` (accepting `nvidia/Cosmos-Guardrail1` does not clear it), fine-grained repo scope, and organization policy. |

## Related

- `docs/workbench/cosmos3-generate.md`: the batch generation tool this serves alongside.
- `docs/workbench/cosmos3-access-preflight.md`: the full credential and license checklist (added by nebius/nebius-physical-ai#264).
