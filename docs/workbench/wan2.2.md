# Wan 2.2 TI2V-5B Workbench support

NPA packages the official Alibaba Wan 2.2 source as a BYOF solution and runs
real TI2V-5B generation on Nebius GPUs. The single-GPU workflow targets one RTX
PRO 6000 Blackwell; the distributed workflow runs one shared generation across
four B200s with NCCL, FULL_SHARD FSDP, and Ulysses sequence parallelism.

Wan is a generative video model. This integration does not represent upstream
Wan as an action-conditioned robotics simulator or an action-prediction model.

## Immutable upstream inputs

| Input | Revision | Packaging |
| --- | --- | --- |
| Official source | [`Wan-Video/Wan2.2` `42bf4cf…`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | cloned into the BYOF image |
| Official TI2V-5B model | [`Wan-AI/Wan2.2-TI2V-5B` `921dbaf…`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e) | fetched at run time; never baked |
| UMT5 tokenizer | [`google/umt5-xxl` `66cb9e7…`](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad) | fetched at run time |

The canonical `npa-wan2-2` image ships the pinned source, its Apache license,
and an OSS CPU dependency base. CUDA-enabled PyTorch and every `nvidia-*`
distribution are deliberately absent from all image layers. `wan-runtime`
installs the pinned runtime atomically from the upstream package indexes and
lands only in `/workspace/.cache/npa/wan2-2/runtime`.
Offline reuse succeeds only for a complete cache whose requirements/Python-ABI
stamp matches exactly; empty or deliberately stale caches fail closed with exit
69. Installation and use remain governed by the upstream package/container terms;
NPA adds no per-run consent variable. `HF_TOKEN` is optional for the public model
and may be forwarded through `--secret-env HF_TOKEN` for rate limits or overrides.
The image build validates the dependency union from two independent sources:
the metadata of every distribution actually installed in `/opt/wan-base`, and
PEP 658 core-metadata sidecars selected by the exact wheel hashes in
`runtime-requirements.txt`. It checks every applicable `Requires-Dist` edge and
allows only the declared Torch/CUDA family to be absent from the public image;
the Torch/NVIDIA wheel payloads are not downloaded or baked during that check.
The resulting hash-bound closure report is rechecked by the image smoke.

Before the private runtime overlay exists, importing full `wan` is intentionally
invalid because upstream imports Torch eagerly. The base-image smoke therefore
executes the input-contract and EasyDict compatibility modules, verifies Wan
package discovery and compiled source bytecode, confirms Torch is absent, and
rechecks the closure report. This proves the strongest honest pre-overlay
boundary; the real single- and multi-GPU workflows prove the full import and
inference path only after runtime provisioning.

The hard gate generates 17 frames at the official TI2V-5B 1280×704 spatial
size and 24 fps with eight sampling steps. The shorter duration and sampling
count make this a capability smoke; they are not a production-quality claim.

## Workflow surfaces

- `byof-wan2.2.yaml` calls native `wan.WanTI2V.generate` on one RTX PRO 6000
  Blackwell (`sm_120`) using the upstream PyTorch SDPA fallback.
- `byof-wan2.2-multigpu.yaml` uses `python -m torch.distributed.run` to launch
  an instrumentation wrapper on exactly four B200 ranks; the wrapper executes
  pinned official `/opt/byof/generate.py` as `__main__` with `--dit_fsdp`,
  `--t5_fsdp`, and `--ulysses_size 4`.
- Both use the real `workbench.byof.repo` toolRef and the existing BYOF upload
  path. There is no synthetic or import-only Wan toolRef.

Validate and plan the checked-in specs:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml \
  --run-id wan22-plan

npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2-multigpu.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2-multigpu.yaml \
  --run-id wan22-multigpu-plan
```

The default declaration is text-to-video. The single-GPU smoke also has an
honest optional image input, but image-to-video remains deferred until its own
live input/output evidence is accepted.

## GPU and runtime gates

The single-GPU path requires exactly one compute-capability 12.0 device. Its
PyTorch 2.13.0 CUDA 13.0 wheel must report `sm_120`, FlashAttention must remain
absent, the patched official model binding must point at native PyTorch SDPA,
and a BF16 SDPA probe must be finite.

The distributed path requires exactly four ranks in one pod and fails unless:

- ranks and local ranks are exactly 0–3 on four unique hashed B200 UUIDs;
- every rank initializes NCCL and observes an all-reduce sum of 10/10;
- both `T5Encoder` and `WanModel` are wrapped with FULL_SHARD FSDP;
- WanModel sequence-parallel bindings are active before wrapping;
- every rank records live Ulysses distributed-attention and all-to-all calls;
- every rank crosses the upstream barrier and the observer terminal barrier;
- every selected device is compute capability 10.0 and the wheel has `sm_100`.

Both routes start as UID 1000 and fetch CUDA Python/model/tokenizer bytes into
writable volumes at run time. That UID and cache ownership prevent accidental
image-layer writes; they are not an isolation claim because the common SkyPilot
bootstrap image contract retains passwordless sudo. A hardened deployment must
apply and validate its own pod security policy. Both routes use the explicit
`-1` terminal-wait sentinel without imposing an artificial deadline. Generic
timeout zero still checks once.

## Output and Rerun contract

Every successful smoke fully decodes the MP4 and rejects invalid dimensions,
frame count, FPS, container size, spatial variation, pixel range, or temporal
variation. The distributed route also requires H.264 and records the MP4
SHA-256.

The single-GPU prefix contains:

```text
npa_byof_summary.json
npa_source_metadata.json
wan2_2_ti2v_5b_text_to_video.json
wan2_2_runtime_inventory.json
wan2_2_ti2v_5b.mp4
wan2_2_ti2v_5b.rrd
wan2_2_ti2v_5b_rrd_manifest.json
```

The distributed prefix contains:

```text
npa_byof_summary.json
npa_source_metadata.json
wan2_2_ti2v_5b_multigpu.json
wan2_2_multigpu_topology.json
wan2_2_multigpu_runtime_inventory.json
wan2_2_multigpu_rank_0.json ... wan2_2_multigpu_rank_3.json
wan2_2_ti2v_5b_multigpu.mp4
wan2_2_ti2v_5b_multigpu.rrd
wan2_2_ti2v_5b_multigpu_rrd_manifest.json
```

After the GPU job and initial BYOF upload succeed,
the closed BYOF postprocess registry calls
`npa.solutions.wan2_2.rerun`, which downloads the immutable evidence, validates the
source contract, builds the recording with Rerun SDK 0.31.4, parses it, runs
`rerun rrd verify` and `rerun rrd stats`, uploads it, downloads it again, and
repeats the structural and embedded-video checks. Any failure makes the BYOF
command fail.

The recording uses these stable entities:

| Entity | Meaning |
| --- | --- |
| `/wan2_2/video/asset` | exact source MP4 bytes as `AssetVideo` |
| `/wan2_2/video/frame` | one `VideoFrameReference` per decoded frame on `video_time` |
| `/wan2_2/summary/overview` | human-readable official source/model/run/output summary |
| `/wan2_2/summary/machine_readable` | sanitized JSON summary |
| `/wan2_2/evidence/validation` | decode, dimensions, variation, size, and SHA facts |
| `/wan2_2/evidence/execution` | accurate execution evidence for either single-GPU runtime or distributed topology |
| `/wan2_2/evidence/runtime` | runtime inventory |
| `/wan2_2/evidence/ranks/rank_0` … `rank_3` | exact sanitized rank evidence for distributed runs |
| `/wan2_2/evidence/teardown` | per-rank post-`destroy_process_group` markers and loaded-NCCL summary |
| `/wan2_2/metrics/*` | static scalar facts, never invented time series |

The default blueprint opens the video beside tabs for overview, validation,
execution, runtime, and machine-readable evidence. The timeline uses the
recorded output FPS, and every video-frame reference points to the embedded MP4
entity.

The manifest schema is `npa.workbench.wan2_2.rerun_manifest.v1`. It records
every source object URI, ETag, byte size, and SHA-256; the RRD URI, SHA-256,
size, Rerun version, entity paths, and row counts; embedded-video identity; and
local plus remote verification results. Only a verified manifest names the
`wan2.2_verified_rerun_recording` capability.

## Accepted evidence and limitations

| Capability | Status | Evidence |
| --- | --- | --- |
| `wan2.2_ti2v_5b_text_to_video` | accepted current evidence | the exact zero-payload public-dev digest used Torch 2.13.0/CUDA 13.0 to produce a real 1280×704 output on RTX PRO 6000 Blackwell |
| `wan2.2_decoded_mp4_validation` | accepted current evidence | the same exact-digest run decoded all 17 frames at 24 fps and passed non-uniform-content gates |
| `wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses` | accepted historical evidence | the prior runtime completed the official four-rank path on 4×B200 |
| `wan2.2_distributed_rank_topology_validation` | accepted historical evidence | the prior runtime proved unique ranks/devices, NCCL 2.27.7 transport and sum 10/10, T5/DiT FULL_SHARD, Ulysses calls, and process-group teardown; the current NCCL 2.29.7 gate requires new operator-accepted live qualification |
| `wan2.2_verified_rerun_recording` | accepted current single-GPU evidence | the exact MP4 and JSON evidence produced an uploaded RRD that passed local and remote identity and structural verification |
| `wan2.2_ti2v_5b_image_to_video` | deferred | optional real input path exists but lacks separately accepted live evidence |
| A14B, S2V-14B, Animate-14B | deferred | separate models and input/GPU contracts |
| official TI2V fine-tuning | deferred | pinned official source has no TI2V training entrypoint |
| stock Wan action prediction | rejected | not an upstream Wan 2.2 capability |

The generated RRD is an evidence visualization of official inference. It does
not turn Wan into a world model or add action conditioning.

For the current single-GPU qualification, Kubernetes independently reported the
accepted digest as the running container `imageID`. The immutable tuple of image
digest, runtime-requirements hash, source/model/tokenizer revisions, observed
image ID, and MP4/RRD proof hashes is recorded in
`npa/src/npa/deploy/wan2_2_image_manifest.json`.

The supported tag is `2.2-ti2v5b-rtfetch-cu130-20260817`. Its promotion gate
accepts only the exact validated public-dev digest and does not claim the
historical four-GPU result for the current runtime.

The materialized historical distributed recording is:

- `s3://<project-bucket>/oss-solutions/wan2.2-multigpu/<operator-run-id>/wan2_2_ti2v_5b_multigpu.rrd`
  (2,948,508 bytes; SHA-256
  `5a4f77461f0877e72c3543508df20edd3a12d0e1fb6bf3f9cfd7215b0dfe0606`).
- `s3://<project-bucket>/oss-solutions/wan2.2-multigpu/<operator-run-id>/wan2_2_ti2v_5b_multigpu_rrd_manifest.json`
  (10,801 bytes; SHA-256
  `1f256009e149100c5dfb06c200861537e41e89dc11d11a5923b4615fb14bc308`).

S3 HEAD/GET, local and downloaded `rerun rrd verify`, `rerun rrd stats`, entity
inspection, and embedded-video identity agreed on those exact distributed
bytes. The live agent Rerun blob and compositor separately agreed with and
visibly rendered the 3,045,269-byte single-GPU RRD
(`49a57f5b5233a83dbde4fb43aa4eeab8d289e045d09f3ce9971ff654c4dcfb10`).

## Licensing and publication

The pinned source, model, and tokenizer declare Apache-2.0. The shipped runtime
contains only the digest-pinned official Python/Debian base and audited OSS
dependencies; Debian copyright records and wheel metadata carry
their GPL/LGPL/BSD/MIT/Apache notices. CUDA Python distributions, model/tokenizer,
credentials, data, and caches are runtime-only. Public eligibility requires four
separate checks: scan the pushed digest and every individual layer/history entry
with `npa/scripts/scan_image_wan_payload.py`; recursively decompress ordinary
gzip, bzip2, and xz payloads as well as nested archives and fail closed on
unreadable or oversized content; inspect the BuildKit SPDX
attestation; bind the SLSA provenance to the exact platform manifest; and review
the license inventory. The scanner proves prohibited-byte absence only—it does
not generate or review the SBOM or make a legal determination. The publication
preflight rejects any digest other than the immutable GPU-accepted tuple, then
repeats the exact-digest scan and attestation binding before any copy.

The exact accepted candidate's Trivy report contains 27 CRITICAL OS-package
findings for which the installed distribution reports no fixed version: 16
`affected`, 9 `fix_deferred`, 1 `end_of_life`, and 1 `will_not_fix`. It contains
zero Trivy secret findings. These are disclosed residual risks under the
repository's `ignore-unfixed` policy; any CRITICAL finding with an available fix
still fails publication. Passing these gates remains an engineering
classification and still requires the organization's human publication/legal
approval.

## Validation

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_wan_rerun.py \
  npa/tests/workflows/test_byof_solution_smokes.py \
  npa/tests/e2e/test_byof_wan22_live_e2e.py \
  npa/tests/e2e/test_byof_wan22_multigpu_live_e2e.py -q
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q
```

The two live GPU cases retain explicit operator gates. Their always-on portions
validate and plan the exact checked-in workflows. Successful future live runs
must also publish and verify the named RRD and manifest.
