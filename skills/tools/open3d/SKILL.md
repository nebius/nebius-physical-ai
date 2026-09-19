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
