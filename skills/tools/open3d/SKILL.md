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
  the discarded shell reached 18 voxels. **That 0.2055 does not transfer to a
  differently sampled capture** — the headline fraction's magnitude depends on the
  sampling scheme, not only on the reconstruction, because uniform random sampling
  leaves gaps well above the median nearest-neighbour spacing and pushes the
  per-triangle maximum further out. Scored against samples drawn from itself, a
  reconstruction with *zero* error measured 0.75126 unsupported, so at this
  threshold the fraction is not measuring fabrication at all. So when spacing approaches the voxel,
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
  summarizes them as `crop_justification.removed_surface_reads_as`, which has three
  states: `extrapolated shell`, `near-threshold surface` (the crop may have taken
  correct geometry), and **`undecided`** when the share falls in `[0.05, 0.2]`,
  roughly a factor of two either side of the reporting boundary. The middle state
  exists because the boundary is where the reading is least trustworthy, and without
  it the artifact made its most confident claim exactly there while the other branch
  was properly hedged — confidence inverted where it should be lowest. Both
  informative poles keep their verdict; the real solid-object scan at 0.1012 now
  returns `undecided` and points you at the two bands. Nothing gates on any of the
  three. Measurements: `evidence/open3d/band-profile-across-scenes.json`.
- **Choose `--voxel-size` at or above twice your median sample spacing.** This one
  ratio sets both of the caveats above, and it is the only input the caller controls.
  Sweeping it on a fixed scene: at a voxel equal to the median nearest-neighbour
  spacing, a *zero-error* reconstruction reports `unsupported_area_fraction` 0.8780
  and the smallest detectable fabrication is 4.7% of surface area; at three times the
  spacing, the same reconstruction reports 0.0055 and the floor falls to 0.75%. A
  tight voxel costs both ways at once — it inflates the headline fraction on correct
  geometry *and* blinds the reading to smaller invented regions. Mesh resolution does
  not matter by comparison: the floor showed no trend above sampling noise across a
  32-fold range of triangle-edge-to-voxel ratio. Measurements:
  `evidence/open3d/floor-vs-voxel-multiple.json`,
  `evidence/open3d/floor-vs-resolution.json`.
- **The bands hold from 0.5 to 5.4 voxels of sample spacing, and the fraction does
  not.** Swept against a fixed voxel on a watertight mesh, so fabrication could be
  measured against ground truth rather than inferred, there was **no case where a
  correct surface read as a shell**. Across that range the share past three voxels
  tracked the area genuinely further than one voxel from the true surface to within
  0.06 absolute. The headline fraction did not: at 0.708 voxels of spacing it
  reported **0.5659** unsupported where only **0.0027** of area was actually
  fabricated, overstating by roughly 200-fold. Measurements:
  `evidence/open3d/band-validity-sweep.json`.
- **A low reading is a lower bound, not a clean bill of health.** The reading is a
  detector with a sensitivity floor, and below that floor it is *confidently wrong*
  rather than undecided — genuine fabrication reads as `near-threshold surface`.
  The error is one-sided: the reading only ever under-calls fabrication, never
  over-calls it, which is the good direction for an advisory field but means a low
  value tells you fabrication is *below the floor* and nothing more.
  **Don't quote the floor as a number.** It moves with geometry, sampling and
  resolution: 0.011 to 0.037 invented area across two geometries × two sampling
  schemes, against 0.09 in an independent setup at a different resolution ratio —
  nearly an order of magnitude. What held in every case, and what to rely on, is
  monotonicity in the invented fraction and no false shell reading on a perfect
  reconstruction. The 0.1012 real scan sits near where the floor falls rather than
  awkwardly on a line. Nothing gates on the reading while the floor is this
  setup-dependent. Measurements:
  `evidence/open3d/floor-across-geometry-and-sampling.json`.
- **Sample evenly and the headline fraction becomes usable again.** Its failure is
  not inherent to the metric, it is what uneven sampling does to a nearest-sample
  distance. A reconstruction with *zero* error measured between 0.1393 and 0.7513
  unsupported depending only on how its samples were drawn — and **exactly 0.0**
  under Poisson-disk sampling, in both geometries tested. Uniform random sampling
  leaves gaps well above the median nearest-neighbour spacing and the per-triangle
  maximum lands in them. If you control the capture, sample evenly; if you don't,
  read the bands and treat the headline fraction as uninformative.
- **Measuring that floor needs a sphere scored against itself, not an occluded
  capture.** Occluding a watertight mesh does not work, and it is worth knowing
  why before trying: Poisson closure over a hole in a *closed* object tracks the
  true surface, so removing 205947 of 300000 samples left only 0.0003 of area
  beyond one voxel and there was nothing invented to miss
  (`band_false_negative_sweep.py`, retained and labelled inconclusive). What works
  is deleting observations from a polar cap while leaving the mesh covering it, so
  the invented fraction is known exactly from the cap angle.
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
