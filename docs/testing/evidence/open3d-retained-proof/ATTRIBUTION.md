# Dataset attribution and image provenance

The input is the public **Open3D DemoICPPointClouds** dataset, distributed in the Open3D `20220301-data` release. Open3D identifies these three fragments as the **living-room1 scene of the Redwood RGB-D dataset**. Credit the **Open3D contributors**, the **Redwood indoor reconstruction dataset contributors**, and the underlying **ICL-NUIM scene contributors**. Related Redwood work: **Sungjoon Choi, Qian-Yi Zhou, and Vladlen Koltun, “Robust Reconstruction of Indoor Scenes,” CVPR 2015**.

The dataset-specific declaration in [Open3D v0.19.0 Dataset.h, line 419](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/data/Dataset.h#L419) identifies **Creative Commons Attribution 3.0 (CC BY 3.0)**. Retain this attribution and the [CC BY 3.0 license link](https://creativecommons.org/licenses/by/3.0/) with the images. The software's MIT license is separate from the dataset license. No endorsement by the data creators or Open3D is implied.

Changes from the source data: the point-cloud fragments were registered and fused, surfaces were reconstructed, and the candidate surface was cropped by an observed-scan support rule. Control C3 deliberately changes alignment; control C4 omits cropping. These results were rendered with a CPU z-buffer evidence renderer. The PNG files in this pack are unchanged copies of those retained renders; they are not Rerun UI screenshots.

Sources:

- [Open3D dataset documentation: DemoICPPointClouds](https://www.open3d.org/docs/latest/tutorial/data/index.html#demoicppointclouds).
- [Original public download](https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip).
- [Versioned Open3D download definition and MD5](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/data/dataset/DemoICPPointClouds.cpp).
- [Dataset-specific CC BY 3.0 declaration](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/data/Dataset.h#L419).
- [Redwood indoor reconstruction project](https://redwood-data.org/indoor/).
- [CVPR 2015 paper and author attribution](https://openaccess.thecvf.com/content_cvpr_2015/html/Choi_Robust_Reconstruction_of_2015_CVPR_paper.html).

The original public ZIP was downloaded independently during review. All three PCD files matched the retained run-input SHA256 values byte-for-byte; the ZIP MD5 also matched the versioned Open3D download definition. The full input hashes appear in `manifest.json`. The Redwood website itself could not be fetched during this audit, so its additional current attribution wording was not independently rechecked.
