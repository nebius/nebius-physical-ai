---
name: open3d
description: Use when registering overlapping point-cloud scans into one frame, optimizing a multiway pose graph, reconstructing a Poisson surface, or reviewing Open3D registration artifacts and their validators.
---

# Open3D point-cloud registration and reconstruction

Two scans of the same room do not share a coordinate frame. This tool recovers the
rigid transform between them with the real upstream pipeline —
`registration_ransac_based_on_feature_matching` over FPFH features, refined by
`registration_icp` — then optimizes the full pairwise pose graph with
`global_optimization`, fuses the result, and reconstructs a surface with
`create_from_point_cloud_poisson`. Nothing is recomputed or smoothed locally: every
fitness, RMSE and correspondence count is what Open3D's own `RegistrationResult`
returned.

CPU-only, and that is a design constraint rather than an unfinished port. Open3D's
`pipelines.registration` and geometry APIs have no CUDA path, so the image does not
derive from the CUDA base and no stage requests an accelerator. A stage that asked
for a GPU here would hold one it cannot use.

The image is `0.20.0-cpu-20260918`, a validation candidate quarantined from public
publication until exact-image byte scans are accepted. Build from committed inputs
with `npa/docker/workbench/open3d/build.sh`.

## Run it

```bash
npa workbench workflow validate-spec workflows/testing/open3d-registration.yaml
npa workbench workflow submit workflows/testing/open3d-registration.yaml \
  --var bucket=<your-bucket>
```

The shipped spec starts from `stage-demo`, which downloads the upstream
`open3d.data` `DemoICPPointClouds` scans (release `20220301-data`) — real captured
indoor scans, so the run needs no operator capture to produce a real result. To
register your own data, replace that stage with `prepare` and point
`--input-path` at a prefix holding at least two `.pcd`/`.ply` files.

`voxel_size` is the single scale knob. The normal, FPFH and correspondence radii
are all multiples of it, exactly as the upstream Global-registration and
Multiway-registration tutorials derive them, so tuning one number rescales the
whole pipeline coherently. It is in the scans' own units; `0.05` suits
metre-scale indoor captures.

## Verbs and what each one proves

| Verb | Real upstream call | Publishes |
| --- | --- | --- |
| `stage-demo` | `open3d.data.DemoICPPointClouds` | staged scans + `manifest.json` |
| `prepare` | none (indexes a prefix) | digest-bound `manifest.json` |
| `register` | RANSAC/FPFH then ICP, consecutive pairs | `pairs.jsonl`, `result.json`, one aligned cloud per pair |
| `multiway` | full pairwise graph + `global_optimization` | `pose_graph.json`, `fused.ply` |
| `validate` | none | `validation.json`, re-derived from the artifacts alone |
| `reconstruct` | `create_from_point_cloud_poisson` | `mesh.ply` with manifold/watertight/area facts |
| `visualize` | Rerun | decode-verified `point_cloud.rrd` |

`validate` deliberately imports no Open3D: the published journal is
self-describing, so a reviewer can re-check a registration without the 400 MB
wheel. `register` writes each source cloud transformed into its target's frame,
because that is precisely what the published transform claims — a reviewer should
be able to open the thing being asserted.

## Exact limits, and what the numbers do not mean

- Fitness and inlier RMSE are Open3D's agreement measures between two clouds.
  They are **not** absolute pose error. The demo scans carry no ground-truth pose,
  so no accuracy claim can be made from a real run against them; the golden eval
  is where a ground-truth check exists, because it applies a known transform and
  asserts the pipeline recovered it.
- RANSAC is randomized. `random_seed` seeds Open3D's global generator so a rerun
  repeats, but a different seed can legitimately give a different transform on
  low-overlap fragments.
- A Poisson surface from partial indoor scans is normally **not** watertight and
  **not** edge-manifold. Those facts are reported, not asserted to be true. Do not
  treat an open surface as a failure, and do not treat a watertight one as proof of
  completeness.
- `reconstruct` crops the lowest-density Poisson vertices (the upstream tutorial's
  density quantile). Those vertices are extrapolation past the samples, so the
  count removed is recorded rather than hidden.
- The density quantile alone is nowhere near enough on a partial scan. Poisson
  returns a *closed* surface, so an open capture comes back wrapped in an
  extrapolated shell that renders as smooth opaque geometry indistinguishable
  from observed structure. On the upstream three-fragment demo scans that shell
  measured **0.527 of the total surface area**, reaching 0.905 m from the nearest
  sample, and it explained none of the observations: cropping all of it left
  sample coverage within one voxel unchanged at 0.99243 and moved
  sample-to-surface RMSE by 0.00005 m. `--support-distance-factor` (default
  `1.0`) discards surface with no observed sample within that many `voxel_size`
  units. One voxel is the sampling geometry, not a tuned constant — the cloud is
  voxel-downsampled, so a surface point interpolating between neighbouring
  samples sits at most about half a voxel diagonal from one. Pass `0` to publish
  the closed surface unchanged, which is right when the capture is already
  complete.
- **The default assumes sample spacing well below the voxel, and says so because
  it is measurably wrong otherwise.** A watertight mesh sampled uniformly at 300k
  points, then voxelized at its own median spacing, reports **0.2055 unsupported
  area** at factor `1.0` — yet that surface is correct: its median vertex sits
  0.02 voxels from the ground-truth mesh, p99 at 0.16, and *no* vertex is further
  than one voxel. The figure is discretization, not fabrication. It collapses to
  0.0075 at factor 1.5 and 0.000084 at 2.0, and the furthest vertex is only 2.21
  voxels out. Genuine fabrication does not collapse like that: on the demo scans
  the discarded shell reached 18 voxels. So when spacing approaches the voxel,
  raise the factor to 1.5–2.0; at `1.0` the crop removed 8.45% of *correct*
  vertices there. Coverage did not catch it and cannot — coverage asks whether
  observations are explained, not whether correct surface was discarded.
- **Read `crop_justification`, not the unsupported fraction.** The fraction alone
  cannot distinguish the two cases and measurably ranks them wrong: across three
  scenes the complete capture reported **0.2009** unsupported while a genuinely
  partial scan of a solid object reported **0.1249**, so thresholding the headline
  number calls the correct surface the worse one. What separates them is how far
  past the voxel the unsupported area lies — the share of it beyond three voxels
  was 0.7297 on the partial demo scans, 0.1012 on that solid-object scan, and
  **0.0000** on the complete capture. `reconstruct` reports those bands as
  `unsupported_area_beyond_1_5_voxels` and `unsupported_area_beyond_3_voxels`, and
  summarizes them as `crop_justification.removed_surface_reads_as`: either
  `extrapolated shell` or `near-threshold surface`, the latter meaning the crop may
  have taken correct geometry. Nothing gates on that reading and it should not be
  treated as a verdict — the solid-object scan sits on the 0.1 boundary, so any
  scene near it is undecided and the bands themselves are the evidence.
  Measurements: `evidence/open3d/band-profile-across-scenes.json`.
- **The bands hold from 0.5 to 5.4 voxels of sample spacing, and the fraction does
  not.** Swept against a fixed voxel on a watertight mesh, so fabrication could be
  measured against ground truth rather than inferred, there was **no case where a
  correct surface read as a shell**. Across that range the share past three voxels
  tracked the area genuinely further than one voxel from the true surface to within
  0.06 absolute. The headline fraction did not: at 0.708 voxels of spacing it
  reported **0.5659** unsupported where only **0.0027** of area was actually
  fabricated, overstating by roughly 200-fold. Measurements:
  `evidence/open3d/band-validity-sweep.json`.
- **The reverse error is open, and cannot be closed with a watertight reference.**
  A real shell reading as near-threshold is the error that matters more, since it
  would tell you to loosen a crop that was working. Measuring it needs ground
  truth, and that is self-defeating: Poisson closure over a hole in a *closed*
  object tracks the true surface — occluding 205947 of 300000 samples left only
  0.0003 of area beyond one voxel — so there is no invented surface to miss. The
  failure the crop exists for is a shell wrapped around an *open scene*, and an
  open scene has no watertight reference by definition. Don't repeat the sweep
  expecting a different answer; it is at `band_false_negative_sweep.py` with its
  result labelled inconclusive. This is why nothing gates on the reading.
- Both surfaces ship: `mesh.ply` is the cropped result and `mesh_uncropped.ply`
  is what Poisson returned, so the crop is a checkable claim rather than a
  deletion. `reconstruct` also publishes the support measurement before and
  after the crop plus coverage in the other direction (observed samples to the
  surface), and the stage fails if those disagree — cropping that reported more
  unsupported area than it started with, or surface left beyond the stated limit.
- `visualize` sends a blueprint: an elevated three-quarter camera with its
  distance solved so the whole scan fits, the scan and surface on independent
  toggles, and the cropped-away surface in red in its own tab. The up axis is
  inferred from the largest planar segment and then snapped to the nearest world
  axis so the horizon stays level; the raw normal, its inlier fraction, and the
  snap angle are all published, because it is an inference. A scan with no
  dominant plane falls back to `+Z` and says so.
- Multiway registration is O(n²) in pairwise registrations. At most 32 fragments
  are accepted in one run, so an over-broad prefix fails fast instead of becoming
  an unbounded job.

## Artifacts and diagnostics

Every artifact is published with a read-after-write digest check, and every
published summary is validated against the journal it claims to describe before the
stage reports success: a summary that inflates the pair count, a 4×4 matrix that is
scaled or sheared rather than rigid, a pose graph whose node 0 has drifted off the
identity anchor, or a mesh with vertices and zero surface area each fail the stage.
See `npa/tests/workbench/test_open3d.py` for the specific corruptions these
validators exist to catch.

When the runner subprocess fails, the error carries the tail of its log, so triage
does not need the pod back. The two failures worth recognizing:

- *"RANSAC feature matching found no correspondences"* — the fragments probably do
  not overlap at this `voxel_size`. Lower it before assuming the data is bad.
- *"voxel_size N removed every point"* — the voxel is larger than the cloud's own
  extent, usually a units mismatch (millimetres given a metre-scale voxel).

## Validate a change

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workbench/test_open3d.py npa/tests/cli/test_open3d_cli.py -q
npa workbench workflow validate-spec workflows/testing/open3d-registration.yaml
```

For a real capability check, run the golden eval inside the built image:

```bash
docker run --rm npa-open3d:<tag> python -m npa.smoke.test_open3d_functional
```
