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

The public image is a zero-payload bootstrap. It contains no vLLM, PyTorch,
CUDA Python runtime, NVIDIA container layer, model, guardrail, credential, or
accepted term. Runtime provisioning and model access happen only after the
operator supplies the relevant credential and explicit run-scoped acceptance.

## Packaging and deployment surface

The packaging contract records the bootstrap as `public`. Supported release
resolution is bound to the exact public development digest that passed its
layer/history scan and real guarded eight-GPU serving acceptance. A new digest
must earn the same evidence before promotion.

The old `vllm/vllm-omni` parent is prohibited because its built filesystem
contained the NVIDIA Deep Learning Container license. A source rebuild also
proved insufficient: its CUDA Python closure includes `cuda-bindings` under the
NVIDIA Software License (v. May 12, 2021), whose downstream-terms requirements
are not established by anonymous GHCR. The public image therefore carries only
the digest-pinned official Python base, bootstrap scripts, and a hash-locked
package manifest. The operator reviews those upstream terms and opts into the
runtime download with
`NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES`; NPA never bakes or persists
that value.

## Weights are never in the image

The bootstrap carries no checkpoint or serving-runtime bytes. The
`Cosmos3-Super` checkpoint, the guardrail models, and every other gated artifact
download **at run time** with the operator's own Hugging Face token, under that
operator's own license acceptance. That is the same posture `npa-cosmos3` holds,
and the reason the built image carries no gated weight bytes to redistribute.
Credentials are used only for authenticated access probes and runtime downloads.

`verify_env.py` proves the exact manifest checksum and absence of serving
modules in the final image. `scan_image_cosmos3_serving_payload.py` independently
walks every saved layer plus config/history and rejects the old vendor bases,
NLC markers, CUDA/vLLM/PyTorch payloads, models, tokens, or baked acceptance.

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

## Build

No GPU is needed to build.

```bash
bash npa/docker/workbench/cosmos3-serving/build.sh
```

Official development publication uses the guarded public-image workflow and an
immutable `dev-<full-git-sha>` tag. The build alone does not authorize a push or
release promotion.

## Run

```bash
docker run -d --name cosmos3-serving --gpus all --ipc=host --shm-size 32g \
  --ulimit nofile=1048576:1048576 \
  -v <runtime-dir>:/opt/npa-cosmos3-serving/runtime \
  -v <hf-cache-dir>:/opt/npa-cosmos3-serving/hf-cache \
  -e NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES \
  -e HF_TOKEN=<your-token> \
  -p 8000:8000 \
  ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-serving@sha256:<validated-digest>
```

Both mounted directories must be writable by uid 1000: the image runs as a non-root
user, and a cache mounted from a root-owned host directory is readable but not
writable. The runtime volume retains the hash-locked serving closure across pod
replacement; the Hugging Face volume retains operator-entitled models.

Budget the cache before first start: the `Cosmos3-Super` checkpoint is about
124 GB on disk plus 17 GB of guardrail weights, so plan roughly 145 GB. The
first-ever download is its own cost, separate from and larger than the
readiness figures below, which were measured with weights already on disk.

`--ipc=host` and the `nofile` ulimit are serving-runtime requirements. Guardrails
remain on by default, and authenticated access to both model repositories is
confirmed before distributed workers or checkpoint downloads start.

On Kubernetes, mount a memory-backed `emptyDir` with `sizeLimit: 32Gi` at
`/dev/shm`. The container runtime's default 64 MiB shared-memory filesystem is
not the documented topology: an eight-GPU process group can finish diffusion
and then lose a worker while returning the decoded response.

### Configuration surface

| Variable | Default | Effect |
| --- | --- | --- |
| `NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE` | unset | Exact `YES` opts this run into direct installation of the pinned CUDA Python serving closure; otherwise bootstrap refuses with exit 78 |
| `NPA_COSMOS3_SERVE_MODEL` | `nvidia/Cosmos3-Super` | Model to serve |
| `NPA_COSMOS3_SERVE_HOST` | `0.0.0.0` | Bind address |
| `NPA_COSMOS3_SERVE_PORT` | `8000` | Bind port |
| `NPA_COSMOS3_SERVE_GUARDRAILS` | `on` | `off` passes `--no-guardrails` and skips only the separate guardrail-repository access probe |
| `NPA_COSMOS3_SERVE_INIT_TIMEOUT` | `1800` | Passed through as `--init-timeout` |
| `NPA_COSMOS3_SERVE_GPUS` | `8` | GPU count the preflight requires |
| `NPA_COSMOS3_SERVE_EXTRA_ARGS` | empty | Appended verbatim to the serve command |

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

The quarantined predecessor image's validation run reached readiness in 592
seconds on a cold cache. The exact accepted zero-payload image took about 1,325
seconds from vLLM startup to `Application startup complete` on 8x B200 with the
checkpoint on network storage. Its runtime bootstrap installs the pinned OSS
serving closure before vLLM starts, so total cold-container startup takes longer.

The image's `HEALTHCHECK` therefore carries a 30 minute `start-period`. A
readiness probe tuned for an ordinary web service reports a healthy boot as a
failure and restarts it into another one, which is a loop that never converges.
A probe tuned even for the historical cold-cache figure would have failed the
accepted image's 1,325-second vLLM boot above. Orchestrators driving this image
need the same allowance, plus time for the runtime closure bootstrap when that
cache is cold.

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

## Historical baseline (quarantined predecessor; not release evidence)

The measurements below describe the former vendor-based image only. That image
is prohibited from public publication and this section is not acceptance
evidence for the zero-payload release. New evidence must name the exact public
development digest built from this architecture.

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
