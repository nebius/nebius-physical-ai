---
name: neural-reconstruction
description: Use when reconstructing real sensor captures into renderable 3D scenes with NVIDIA Omniverse NuRec / the Neural Reconstruction Engine (NRE) on Nebius — NCore V4 input, 3DGUT Gaussian training, renderable USDZ, novel-view rendering, and the Rerun recording the NPA agent displays. Also use when an NCore sequence will not load in NRE, when picking the GPU for a reconstruction, or when changing the nurec workbench tool, CLI, or SkyPilot workflow.
---

# Neural Reconstruction (NuRec / NRE)

## Source And Attribution

Adapted from the NVIDIA Omniverse NuRec agent skills at
<https://github.com/NVIDIA/nurec-skills> (`skills/nre`,
`skills/physical-ai-datasets`, `skills/ncore`) and the NVIDIA NCore data library
at <https://github.com/NVIDIA/ncore>.

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. Upstream licenses are
Apache-2.0 and CC-BY-4.0. Trademarks (NVIDIA, Omniverse, NuRec, NRE, Isaac Sim,
Cosmos) belong to NVIDIA. See `skills/NOTICE-NVIDIA-SKILLS`.

NPA does not redistribute NVIDIA source or model weights. The capability drives
the public NGC containers and public Hugging Face datasets from Nebius
infrastructure, orchestrated by SkyPilot.

## When To Use

Load this skill when the user wants to:

- turn a real sensor capture (photographs, multi-camera clips) into a renderable
  3D Gaussian scene and render **novel views** from it;
- run, modify, or debug `npa workbench nurec` or
  `npa/src/npa/workflows/skypilot/nurec-reconstruct.yaml`;
- work out why an NCore sequence fails to load in NRE;
- choose the GPU for a reconstruction or rendering job;
- make a reconstruction run show up in the NPA agent's Rerun panel.

**Do NOT use this skill for:** Cosmos video augmentation (use
`skills/workflows/physical-ai-data-factory/SKILL.md`), per-object asset
extraction from sparse views (upstream `asset-harvester`, not implemented here),
or generic Rerun visualization of a Sim2Real run
(`skills/workbench/sim2real-engine/SKILL.md`).

## GPU Routing (hard constraint)

Gaussian reconstruction and rasterization are **RT-core** work. Route these jobs
at:

- **RTX PRO 6000 Blackwell** — `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` on the
  RT-core Kubernetes context (what the shipped workflow uses), or
- **L40S** — `L40S:1`.

Never route the reconstruct/render path at **H100 / H200 / A100 / B200**: they
have no RT cores. `npa workbench nurec check` fails with an explicit error when
the visible GPU is one of those (`has_rt_cores`). NRE additionally requires
driver **R580+** on Blackwell and >= 24 GB VRAM (48 GB+ recommended). See
`skills/atomic/gpu-selection/SKILL.md`.

## Container Access (read this before debugging a pull failure)

NRE ships only as a closed-source NGC container, so the container *is* the
runtime — there is no source/pip path.

| Repository | With a standard `NGC_API_KEY` |
| --- | --- |
| `nvcr.io/nvidia/nre/nre` | **402 Payment Required** — needs an extra entitlement |
| `nvcr.io/nvidia/nre/nre-ga` | **pullable** — the General Availability channel |
| `nvcr.io/nvidia/nre/nre-tools-ga` | pullable (auxiliary seg/depth data only) |

Use the **`-ga`** repositories. `npa workbench nurec check` reports
`ngc_image: entitlement-required` rather than failing opaquely when a
non-GA reference is configured.

## Real Entrypoints

Every stage is a real command; nothing here is a manifest stub.

```bash
npa workbench nurec check       # NGC pullability + HF download rights + RT-core GPU
npa workbench nurec fetch       # real NCore V4 shards + derived rig pose edge
npa workbench nurec reconstruct # NRE 3DGUT training -> renderable USDZ + metrics
npa workbench nurec render      # `nre render` novel views (rig offset, not training views)
npa workbench nurec visualize   # reports/sim2real.rrd for the agent's Rerun panel
npa workbench nurec finalize    # reports/final.json aggregate
npa workbench nurec status      # what a run prefix holds, stage by stage
```

| Concern | Implementation |
| --- | --- |
| Pure logic + argv builders | `npa/src/npa/workbench/nurec/nurec.py` |
| NCore rig-pose derivation | `npa/src/npa/workbench/nurec/ncore_rig.py` |
| CLI | `npa/src/npa/cli/nurec/__init__.py` |
| SDK | `npa.workbench.nurec` (`check`, `fetch`, `reconstruct`, ...) |
| SkyPilot workflow | `npa/src/npa/workflows/skypilot/nurec-reconstruct.yaml` |
| Declarative twin | `npa/workflows/workbench/npa-workflows/nurec-reconstruct.yaml` |
| Rerun recording | `npa.workflows.data_factory_viz.build_run_rrd` |

## Input Data

Default: **`nvidia/PhysicalAI-NuRec-PPISP`** — ungated, CC-BY-4.0, real
photographic captures of four outdoor object-centric scenes shipped **already in
NCore V4**, which is what NRE consumes. Scene `struktur28`, variant `auto`, is
the small default (59 images across two cameras, ~1.1 GB archive).

The `nvidia/PhysicalAI-Autonomous-Vehicles*` family (raw clips, `-NCore`, and the
pre-built `-NuRec` USDZ scenes) is **gated**: the account that owns `HF_TOKEN`
must accept the NVIDIA AV Dataset License on each dataset page before any token
can pull a byte. `npa workbench nurec check` reports `hf_dataset: gated` for
those until it is accepted — it probes real download authorization, not just
metadata visibility, because a gated repo still answers 200 for
`/api/datasets/<id>`.

## The rig -> world Pose Edge (the thing that breaks first)

NRE's NCore data source requires a `("rig", "world")` pose-graph edge:

```text
# nre/datasets/ncore.py, nre-ga 26.04
# TODO: frame-pose only data might fail here as there are no rig poses ...
rig_world_edge = unpack_optional(
    sequence_loader.pose_graph.get_edge("rig", "world"),
    msg="Rig-to-world poses are currently required to determine scene extend")
```

Object-centric captures have no vehicle rig, and NVIDIA's own COLMAP -> NCore
converter stores per-camera `<camera> -> world` poses with **no rig node**
(`tools/data_converter/colmap/converter.py`: `reference_frame = "world"`). So
every COLMAP-derived NCore sequence — including PPISP's own export — fails to
load with:

```text
ValueError
Rig-to-world poses are currently required to determine scene extend
```

`dataset.frame_generic_data_pose_overwrite=true` does not help either; it needs a
`T_sensor_worlds` generic-data field these sequences do not carry.

`npa workbench nurec fetch` fixes this by default. For a single-camera capture
the rig **is** the camera, so `rig -> world` is exactly that camera's pose
trajectory. `ncore_rig.derive_rig_poses` writes one small extra component store
plus a new sequence meta-file that symlinks the original shards, so the source
data is never modified:

- the derived poses live in their own component instance **`npa_rig`**, not
  `default` — `open_component_readers` asserts instance names are unique across a
  sequence's stores, and re-using `default` raises
  `Component instance default encountered multiple times`;
- selecting a poses group **replaces** the pose set rather than merging, so the
  derived component carries a complete copy of the original edges plus the rig
  edge;
- `reconstruct` then passes `dataset.poses_component_group=npa_rig`.

Pass `--no-derive-rig` for AV-style sequences that already ship a rig edge (the
derivation short-circuits with `already_present: true` anyway), and
`--reference-camera <id>` to pin which camera becomes the rig.

## Recipe Selection

The container resolves `--config-name` against its own `configs/` tree. Pick by
capture shape:

| Capture | Recipe |
| --- | --- |
| Object-centric / static / camera-only (PPISP, COLMAP-style) | `configs/experimental/3dgut/3dgut_colmap.yaml` (**the default**) |
| Waymo Open Dataset | `configs/apps/AV/Waymo/3dgut_dynamic*.yaml` |
| PhysicalAI Autonomous Vehicles (Hyperion-8.1) | `configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml` |
| PandaSet / NV / Tesla / Alpasim | `configs/apps/AV/{PandaSet,NV,Tesla}/...`, `configs/apps/Alpasim/...` |

The default recipe already composes `options/artifact: default` (which is what
sets `checkpoint.artifact.enabled`, i.e. the renderable USDZ), MCMC
densification, SfM-point-cloud initialization, and disables difix/mesh/ground.
Enumerate what a given release actually ships with:

```bash
find /app/run.runfiles/_main/configs -name '*.yaml' | sort
/app/run --help          # sub-command inventory
/app/run render --help   # authoritative flag surface
```

## Novel Views vs Training Views

`nre render` defaults to `--replicate-training-views`, which re-renders views the
model was trained on. That is **not** a novel view. The tool therefore emits
`--no-replicate-training-views` plus a rig offset by default:

- `--rig-translation-offset` and `--rig-rotation-offset` are `FLOAT...` (three
  values) upstream; the CLI accepts one `"x,y,z"` string and expands it;
- a zero offset with no custom trajectory is rejected rather than silently
  producing training views;
- `--renderer default` (the artifact's own trained renderer) is the default.
  `nrend` is faster but needs the nrend model dictionary embedded in the USDZ,
  which the object-centric recipe disables.

## Artifact Layout

One S3 run prefix per run, so the agent's artifact browser picks it up:

```text
s3://<bucket>/<prefix>/neural-reconstruction/<run_id>/
  ncore/manifest.json                    # dataset, scene, sensors, rig derivation
  input/camera_images/<camera>/*.jpg     # real capture frames (export-ncore-benchmark-gt)
  reconstruction/last.usdz               # renderable Gaussian scene
  reconstruction/parsed.yaml
  reconstruction/metrics.yaml            # test/psnr, test/ssim, test/lpips
  reconstruction/val/...                 # NRE validation renders + videos
  novel_views/<camera>/*.png             # rig-offset novel views
  novel_views/<camera>.mp4
  reports/final.json
  reports/sim2real.rrd                   # preferred artifact for the Rerun panel
```

`<run_id>` must be a single safe segment embedding the submit timestamp, e.g.
`neural-reconstruction-struktur28-20260731t050118z`, so the run picker dates the
run by when it **started** (`npa.workflows.artifacts._run_started_at`).

`.usdz` classifies as `download`, which is correct: no browser renders USDZ and
the agent has no USDZ viewer. Viewability comes from the `.rrd`, `.png`, `.mp4`
and `.json`.

## Running The Workflow

```bash
# 1. Confirm access and GPU suitability first (seconds, no image pull).
npa workbench nurec check --require-gpu --output json

# 2. Submit the real GPU workflow. The wrapper substitutes ${...}; SkyPilot 0.12.2
#    does not interpolate them itself.
RUN_ID="neural-reconstruction-struktur28-$(date -u +%Y%m%dt%H%M%S)z"
npa workbench workflow submit npa/src/npa/workflows/skypilot/nurec-reconstruct.yaml \
  --run-id "$RUN_ID" \
  --infra k8s/<rt-core-context> \
  --var NPA_NUREC_IMAGE=nvcr.io/nvidia/nre/nre-ga:26.04 \
  --var NPA_NUREC_RUN_ID="$RUN_ID" \
  --var NPA_NUREC_RUN_URI="s3://<bucket>/<prefix>/neural-reconstruction/$RUN_ID" \
  --var NPA_SRC_S3_URI="s3://<bucket>/npa-src/<tag>" \
  --var AWS_ENDPOINT_URL="<s3-endpoint>" \
  --var AWS_ACCESS_KEY_ID="<key>" --var AWS_SECRET_ACCESS_KEY="<secret>" \
  --var HF_TOKEN="<hf>" --var NGC_API_KEY="<ngc>"

# 3. Watch it.
npa workbench workflow status "$RUN_ID" --watch
npa workbench workflow logs "$RUN_ID" --follow

# 4. Inspect the run tree.
npa workbench nurec status --run-uri "s3://<bucket>/<prefix>/neural-reconstruction/$RUN_ID/" --output json
```

Budget roughly: image pull (~14 GB compressed) on a cold node, a few minutes of
setup, ~1 min fetch, ~20 min for 30k 3DGUT steps on one RTX PRO 6000, then
rendering, the `.rrd`, and the upload.

## Two Container Quirks The Workflow Already Handles

1. **No `sudo`.** SkyPilot's runtime setup calls bare `sudo` when it writes the
   nofile limits and `/etc/fuse.conf`
   (`sky/templates/kubernetes-ray.yml.j2`). The NRE image has no `sudo`, so that
   step exits 127 and the cluster never becomes usable even though provisioning
   otherwise succeeds. The workflow's `pod_config` runs an initContainer that
   drops a shim which execs its arguments into `/usr/local/sbin` (on the image's
   PATH, and empty) — equivalent to SkyPilot's own `alias sudo=""` root path.
2. **64 MB `/dev/shm`.** NRE wants tens of GB; the workflow mounts a 64 Gi
   `emptyDir{medium: Memory}` at `/dev/shm`.

Also: the image ships no `unzip`, `git`, or `ffmpeg`. Extraction uses stdlib
`zipfile`; `setup` installs `ffmpeg` for `render --export-video`.

## Artifact Layout Surprise

Upstream docs describe `<run>/usd-out/last.usdz`. nre-ga 26.04 actually writes
one artifact per checkpoint as `<run>/artifacts/<step>.usdz` plus
`<run>/artifacts/last.usdz`. `latest_usdz()` handles both and picks the newest,
because taking the first alphabetical match would ship the 1000-step preview
instead of the trained scene.

## Verify

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_nurec.py \
  npa/tests/workbench/test_nurec_access.py \
  npa/tests/workflows/test_nurec_viz.py \
  npa/tests/workflows/test_nurec_artifacts.py \
  npa/tests/orchestration/npa_workflow/test_real_components.py -q

npa/.venv/bin/python -m ruff check npa/src/npa/workbench/nurec npa/src/npa/cli/nurec
npa workbench nurec reconstruct --help
```

GREEN when: the workflow name equals its file stem, the accelerator is an RT-core
GPU with no H100/H200 reference, every stage in the YAML is a real
`npa workbench nurec` call, the `.rrd` carries the `novel_view` /
`reconstruction` / `gaussians` entities, and a synthetic run prefix is listed by
`list_runs` with `has_viewable=True` and `preferred == reports/sim2real.rrd`.

## Limitations

- **Linux x86_64 + NVIDIA RT-core GPU only.** aarch64 is unsupported upstream.
- **NGC entitlement.** Only the `-ga` repositories are pullable with a standard
  key.
- **Multi-GPU with `dataset.aux_data=false` is a known upstream crash.** The
  camera-only default therefore stays single-GPU; raise `--world-size` only with
  aux data enabled.
- **No gRPC sensor-sim path yet.** `serve-grpc` / `render-grpc` (LiDAR sweeps,
  actor editing, simulator loops) exist in the container but are not wired into
  a workbench verb.
- **`nre-tools` auxiliary data (segmentation, depth, DINOv2) is not wired in.**
  It is a second ~22 GB image and the object-centric default does not need it.
