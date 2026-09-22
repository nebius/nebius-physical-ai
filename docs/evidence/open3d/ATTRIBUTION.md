# Attribution for the data behind this evidence

Every scan, registration, mesh and rendered view in this directory derives from one
upstream dataset. `viewer-ui/ordinary-native-replay.png` is a published rendering of it,
so the attribution below travels with it.

Open3D's own code is MIT. **That is the software licence and it does not cover this
data.** Open3D declares a separate, dataset-specific grant for each dataset it ships, and
for `DemoICPPointClouds` that grant is **CC BY 3.0** — see the `DemoICPPointClouds`
declaration in
[`cpp/open3d/data/Dataset.h`](https://github.com/isl-org/Open3D/blob/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/cpp/open3d/data/Dataset.h).

## What the data actually is

`DemoICPPointClouds` is the `living-room1` sequence from Redwood's **augmented ICL-NUIM**
indoor benchmark. ICL-NUIM is **synthetic**: the scene is rendered from a modelled living
room, not captured with a physical depth sensor. Nothing here is a real-world sensor
recording, and no claim in this evidence set should be read as one.

## Required credit

- **Dataset (Redwood augmented ICL-NUIM):** S. Choi, Q.-Y. Zhou, and V. Koltun,
  "Robust Reconstruction of Indoor Scenes", *IEEE Conference on Computer Vision and
  Pattern Recognition (CVPR)*, 2015.
- **Original benchmark (ICL-NUIM):** A. Handa, T. Whelan, J. B. McDonald, and
  A. J. Davison, "A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM",
  *IEEE International Conference on Robotics and Automation (ICRA)*, 2014.
  Original release: <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>
- **Packaging and distribution:** Open3D, <https://www.open3d.org/>

## Licence

Creative Commons Attribution 3.0 Unported (CC BY 3.0):
<https://creativecommons.org/licenses/by/3.0/>

## Notice of changes

CC BY 3.0 requires that modifications be indicated. The material was changed. From the
three `cloud_bin_*.pcd` fragments the workflow voxel-downsampled at 0.05 m, estimated
normals and FPFH features, registered the fragments pairwise with RANSAC and refined with
point-to-plane ICP, optimized the resulting multiway pose graph, fused the posed
fragments, reconstructed a Poisson surface, cropped surface unsupported by observation,
and rendered views of the result. The published image is a rendering of those derived
products, not of the original data.

## No endorsement

Neither the authors above nor Open3D endorse this work or its use here. The attribution
is required by the licence; it does not imply any relationship.

## Provenance of the exact bytes used

The run read Open3D's `DemoICPPointClouds` release `20220301-data`.

| | |
| --- | --- |
| Release archive | `DemoICPPointClouds.zip`, release `20220301-data` |
| Archive SHA256 | `b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5` |

The three point clouds it contains, as hashed in this run:

| Fragment | SHA256 |
| --- | --- |
| `cloud_bin_0.pcd` | `e1e100802c29ef454c6b523084668ee0e2f365ec52eaeebe79ae804c20447b15` |
| `cloud_bin_1.pcd` | `a4c3dc0ad7b1279736491b9b2638991d4c808605997be4f9ab174c24a9fa6e52` |
| `cloud_bin_2.pcd` | `1e68e194ebc1941f0f29764e4daf89340e69b224d2b80db5efbc1373a17f8b4a` |

Those three hashes also appear in `replay-b18d-public-report.json` under
`prepared/manifest.json`, which is what ties the published rendering to this release
rather than to some other copy of the same sequence.
