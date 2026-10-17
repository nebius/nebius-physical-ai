# OpenDM DM05 / LIBERO workflow

[`dm05-opendm.yaml`](../../workflows/testing/dm05-opendm.yaml) is an
operator-owned, private-BYOF workflow for the upstream OpenDM DM05 LIBERO
training and evaluation path. It is not a public NPA image, a hosted model
service, or original NPA research.

The successful path has five connected executable stages:

1. Fetch the exact licensed LeRobot LIBERO dataset revision, map its actual
   two-camera/8D-state/7D-action frames into OpenDM's JSONL/image contract,
   and compute genuine OpenDM normalization stats.
2. Fetch the exact DM05 base checkpoint and run upstream LIBERO full SFT.
3. Serve the resulting checkpoint through upstream HTTP `/v1/infer` and record
   an actual action chunk from a prepared observation.
4. Start that same checkpoint-derived server and run the pinned upstream
   Dexbotic closed-loop LIBERO evaluator, preserving `results.json` and its
   rollout MP4s.
5. Create factual Rerun metrics/provenance plus a copy of an evaluator-written
   MP4. It does not turn a simulation result into a physical-robot claim.

## Pinned upstream inputs and credit

| Component | Pin and attribution | Terms / delivery decision |
| --- | --- | --- |
| Training and HTTP-server code | [Dexmal OpenDM](https://github.com/dexmal/opendm) commit `7d52f1591437332cb0157be3303c1c46da811344`; Apache-2.0, Copyright 2026 Dexmal | Apache-2.0 source may be retained in the private BYOF image with its LICENSE and source metadata. Modifications are only the NPA wrapper; the upstream code remains attributed to Dexmal. |
| Base checkpoint | [Dexmal/DM05](https://huggingface.co/Dexmal/DM05) revision `5cd18734814abb075a9ccfd9ad6d16777b5cf10e`; model card cites the Dexmal Team's July 2026 DM0.5 work | The card declares the [Gemma terms](https://ai.google.dev/gemma/terms). The checkpoint is runtime-fetched to the operator's run; it is neither copied into the image nor published or served as an NPA hosted service. Public, anonymous payload availability is operational access, not a redistribution finding. |
| Training data | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) revision `affa19c0de0f6bce2a7edd26dddef8a532e7e6f6`; Hugging Face VLA's LeRobot-v2.1 conversion of LIBERO, with the original LIBERO citation | The immutable card declares CC-BY-4.0. It is runtime-fetched, run-scoped, and not included in a public image or redistribution package. The adapter maps its embedded `image`/`image2`, 8D state, and 7D action fields to OpenDM's `images_1`/`images_2` JSONL contract with no action padding, inversion, or convention change. Preserve the CC-BY attribution and the original LIBERO citation in provenance. |
| Closed-loop evaluator | [Dexmal Dexbotic Benchmark](https://github.com/dexmal/dexbotic-benchmark) commit `789b87f50d9fadc7663d2e8bac057941221aab81`; MIT, Copyright 2025 Dexmal | Its LIBERO gitlink is [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) `8f1084e3132a39270c3a13ebe37270a43ece2a01`, MIT, Copyright 2023 Lifelong Robot Learning. Keep both notices when constructing the private image. |
| CUDA/PyTorch/FlashAttention, Python and evaluator dependencies | Exact package/base identities are captured by the actual private image build receipt and its dependency inventory | They are separate runtime dependencies with their own terms. No conclusion about public redistribution follows from source access or from a successful image build. |

The report artifact includes the upstream source pins, model/data revisions,
normalization settings, input artifact URIs, numeric evaluator metrics, and
the actual MP4 checksum. It also retains the exact manifests from the four
prior stages without copying data or model weights again.

The separate [Dexmal/libero](https://huggingface.co/datasets/Dexmal/libero)
card was inspected at revision `f15a66b3975f8cd210c746991f80adde5ab05ca4`.
It labels the dataset only as `cc` and provides no root license file, so this
integration does not treat it as a specific use or redistribution grant and
does not select it. This is a source-selection decision, not a statement about
Dexmal's authorship of its data or models.

Use this citation for the base model, without relabeling it as NPA work:

```bibtex
@misc{dm05,
  title={{DM0.5}: An Open-World Foundation Model for General-Purpose Embodied Intelligence},
  author={{Dexmal Team}},
  month={July},
  year={2026},
  url={https://www.dexmal.com/blog/dm0.5/index_en.html}
}
```

## Normalization and service boundary

The workflow rejects a mismatched checkpoint or observation before inference:

| Field | Exact LIBERO DM05 contract |
| --- | --- |
| Dataset registration | `libero_pi0_all` |
| Cameras | `images_1`, `images_2`, in that order; prompts `Head`, `Left wrist` |
| State | 8 floating-point values: six Franka joints followed by two gripper values |
| Action | 7 values, absolute-action experiment configuration |
| Action chunk | 10, consistent across stats, training, service, and evaluator |
| Checkpoint stats | Training-generated `norm_stats.json` is copied into the checkpoint and inference requires it |

The evaluator consumes two cameras in this same order and an 8D state (`eef`
position, axis-angle orientation, and gripper state) through Dexbotic's
documented HTTP client. That state is **not** relabeled as the training state:
OpenDM's pinned `dm05_libero.py` sets `add_state=False`, so the HTTP server
still parses and shape-normalizes it, but does not tokenize it into model
conditioning. The report records this non-equivalence, and the evaluator
continues to make no state-conditioned performance claim. During evaluation,
the server uses one visible GPU and the simulator/evaluator another; no
cross-job localhost endpoint is assumed.

## Private image and workflow execution

Build an operator-private image before submitting. The image must contain the
two pinned source trees and compatible Python environments, but **must not**
contain DM05 weights, LIBERO data, Hugging Face caches, credentials, output
artifacts, an `ACCEPT_*` variable, or optional telemetry/privacy consent.
Do not reuse OpenPI's Gemma acceptance flag: OpenDM/DM05 has no corresponding
upstream switch.

Use an exact CUDA base and an exact Dexbotic source pin. The upstream Dexbotic
`latest` image currently inherits unrelated EULA/privacy environment settings,
so it is not used as a base for this solution. The private build instead
reconstructs only the required LIBERO evaluator dependency environment and
records the actual dependency inventory. The intended parent is
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85`.

The build command must:

- install the OpenDM Python 3.10 environment from `/opt/byof` at the pinned
  source revision, including upstream CUDA 12.8 PyTorch and FlashAttention;
- clone Dexbotic at its exact revision and initialize only its pinned LIBERO
  submodule, then create the evaluator's Python 3.8 environment from the
  upstream LIBERO requirements. Install the standard `cmake` build tool too:
  the upstream `egl_probe` dependency compiles with CMake;
- install `rerun-sdk`, `pyarrow`, and the NPA runtime dependencies used by
  the staged adapter; `pyarrow` reads the licensed LeRobot v2.1 Parquet
  records before the adapter writes OpenDM's native JSONL/image representation;
- install the BSD-2-Clause `imageio-ffmpeg` wrapper from its source
  distribution (`--no-binary imageio-ffmpeg`), not the wheel-bundled static
  executable. Set `IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg`, verify that the wrapper
  resolves to Ubuntu's dynamically packaged FFmpeg, and reject any
  `imageio_ffmpeg/binaries/ffmpeg*` path before the image can qualify; and
- emit source/dependency/license inventory into the private build evidence; and
- scan the complete image before the private push. It remains outside public
  registry admission unless a separate, complete redistribution review passes.

The generic BYOF builder makes the checked-out OpenDM tree owned by the runtime
user. When the root-owned build layer records its revision, use the scoped
`git -c safe.directory=/opt/byof -C /opt/byof rev-parse HEAD` form; do not
write a global Git configuration.

Run the image build through `npa workbench byof run` (or its
`npa/scripts/run_byof_repo.py` equivalent) with `--repo-url
https://github.com/dexmal/opendm.git`, `--repo-ref
7d52f1591437332cb0157be3303c1c46da811344`, the exact base above, and
`--skip-run --skip-push` for local validation. Inspect the completed image
before a push, and push it only to an operator-authorized **private** registry;
the generic public registry default is not an authorization to publish this
candidate. Retain the resolved `@sha256:` image identity only in the operator's
private evidence. Then submit the workflow with the run-scoped bucket and that
image identity passed as overrides:

```bash
: "${DM05_RUN_ID:?set a unique run id}"
: "${DM05_BUCKET:?set the operator-owned run-scoped bucket}"
: "${DM05_RUNTIME_IMAGE:?set the private digest-pinned runtime image}"
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/dm05-opendm.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/dm05-opendm.yaml \
  --run-id "$DM05_RUN_ID" --var "bucket=$DM05_BUCKET" \
  --var "runtime_image=$DM05_RUNTIME_IMAGE" --check-render --json
npa/.venv/bin/npa workbench workflow submit workflows/testing/dm05-opendm.yaml \
  --run-id "$DM05_RUN_ID" --runtime --stage-src \
  --var "bucket=$DM05_BUCKET" --var "runtime_image=$DM05_RUNTIME_IMAGE"
```

Inspect the `report/manifest.json`, `provenance.json`, `dm05-opendm.rrd`, the
upstream evaluator `results.json`, and a decoded evaluator MP4 independently
before marking a live run accepted. A short operational run is a smoke only;
the default 100,000-step/50-trials-per-task configuration is the configured
`libero_spatial` suite run, not the complete LIBERO benchmark and not
convergence evidence. Neither establishes a physical-robot success claim.

## Current acceptance state

The workflow, explicit artifact chain, normalization contract, and source/data
revision access probes are implemented and locally validated. It is not yet
registry-ready or publicly publishable. Live acceptance requires the actual
private immutable-image build, Kubernetes pull, all five workflow stages, and
independent final-artifact inspection. Any unavailable target remains reported
as unverified rather than inferred from a local check.
