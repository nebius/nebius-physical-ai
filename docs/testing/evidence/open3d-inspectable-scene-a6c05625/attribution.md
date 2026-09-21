# Dataset attribution and changes

This recording and its native screenshot derive from Open3D **DemoICPPointClouds**, release `20220301-data`: three fragments of the Redwood augmented ICL-NUIM indoor benchmark. The living-room scene is synthetic, with simulated sensor noise. It is not a physical sensor capture.

The dataset-specific license is [Creative Commons Attribution 3.0 Unported (CC BY 3.0)](https://creativecommons.org/licenses/by/3.0/). Open3D's separate MIT software license is not the data license. The authoritative [DemoICPPointClouds declaration in fixed Open3D source](https://github.com/isl-org/Open3D/blob/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/cpp/open3d/data/Dataset.h) explicitly identifies CC BY 3.0. The [original ICL-NUIM release](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) independently states the same license. Both sources were retained and rechecked on 2026-09-21. The exact archive has three PCD files and `init.log`; it does not contain a separate license file.

Credit:

- S. Choi, Q.-Y. Zhou, and V. Koltun, “Robust Reconstruction of Indoor Scenes,” CVPR, 2015. Dataset: Redwood augmented ICL-NUIM.
- A. Handa, T. Whelan, J. B. McDonald, and A. J. Davison, “A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM,” ICRA, 2014. Original ICL-NUIM benchmark.
- [Open3D](https://www.open3d.org/) packaged and distributed these fragments.

No endorsement or relationship with the authors or Open3D is implied.

Changes: the workflow downsampled the three fragments at 0.05 m, estimated normals/features, registered and refined poses, optimized a three-node pose graph, fused samples, reconstructed a Poisson surface at depth 9 and cropped triangles using the declared observation-support criterion. The recording preserves those output geometry arrays, replaces operational metadata with a positive safe set, and reproduces the default viewer layout/cameras. The screenshot is a rendering of this derivative. The blank-control image is a separate diagnostic recording containing no geometry.

The support criterion does not establish that removed triangles were fabricated; sparse but correct surfaces can fail it. The partial mesh is not watertight or manifold, and no physical sensor accuracy, complete scene reconstruction or simulation-ready geometry is claimed.

Archive SHA256: `b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5`.

Fragment SHA256:

- `cloud_bin_0.pcd`: `e1e100802c29ef454c6b523084668ee0e2f365ec52eaeebe79ae804c20447b15`
- `cloud_bin_1.pcd`: `a4c3dc0ad7b1279736491b9b2638991d4c808605997be4f9ab174c24a9fa6e52`
- `cloud_bin_2.pcd`: `1e68e194ebc1941f0f29764e4daf89340e69b224d2b80db5efbc1373a17f8b4a`

Retained authoritative-source SHA256:

- [Dataset.h](https://raw.githubusercontent.com/isl-org/Open3D/b6c5e196384ad71e75b6e6f9c5da22d046221f1d/cpp/open3d/data/Dataset.h): `41df8058df510393ac4d2feba62c1042ddb5d38e331dbaabbe4d8ce94639d729`
- [iclnuim.html](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html): `2ba6466fac964071fa7d4632f738a19249cd3ed4da437a22f65d3602ab66ed05`
