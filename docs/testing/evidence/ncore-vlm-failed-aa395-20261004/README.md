# NCore visual diagnostic: failed calibration, final not called

The one frozen hosted diagnostic at source `aa3958d9d63de78a127e55ad1759311dd9848374` made four actual requests. It **failed calibration: 2 TP, 1 TN, 1 FP, 0 FN** at the unchanged 0.8 threshold. The conditional final request was **not called**; there is no final-render VLM score. No retries, label changes, threshold changes, model switches or prompt changes were made.

| Frozen control | Expected | Score | Outcome |
| --- | --- | --- | --- |
| [Local displaced duplicate patch](controls/case-9b0f5fde653b0b74/frame-000.png) | Negative | 1.0 | False positive |
| [Source camera 2](controls/case-28fc2744fc42167c/frame-000.png) | Positive | 1.0 | True positive |
| [Global block permutation](controls/case-ce11af7af01cfa49/frame-000.png) | Negative | 0.2 | True negative |
| [Source camera 1](controls/case-d4bff0fc8d9d7504/frame-000.png) | Positive | 1.0 | True positive |

Every control contains four ordered frames. All 16 actual control PNGs and the four frozen-but-not-sent final PNGs are included byte-for-byte. The negative controls are deliberately altered diagnostic images, not undocumented native renderer defects. The false-positive control visibly contains a rectangular displaced duplicate patch; the model nevertheless described it as having no significant defects. Its complete returned rationale, all four actual scores, request/response/output hashes, token usage, and the entire 20-image hash population are in [summary.json](summary.json).

The model was `openbmb/MiniCPM-V-4_5`. The exact [rubric](vlm-rubric-v2.txt), [control task](vlm-calibration-task-v2.txt), and unused [final task](vlm-final-task-v2.txt) are retained. Four HTTP 200 responses were parsed and hash-bound; four conditional one-shot markers were read back byte-for-byte. The current source's final gate rejected the failed calibration before its provider call. Usage was 10,296 prompt and 456 completion tokens (10,752 total); billed cost was not independently measured.

This does not erase the separately successful one-run native photometry result: the 518 fitted-frame means were PSNR 23.871788, SSIM 0.674246, and LPIPS 0.493915, and all four workflow stages succeeded at source `10b3c34dd6033a1de8798d0013b062848c73a3a0`. Complete unchanged source/output lineage and the earlier failures are in the [photometry evidence](https://github.com/nebius/nebius-physical-ai/blob/8e5c4a1d07f6f04e7339ad72615b45f32afc5345/docs/testing/evidence/ncore-photometry-10b-20261004/README.md). These are fitted-population metrics, not held-out accuracy, generalization, physical safety, or independent novel-view quality acceptance. Earlier geometry/photometry failures and historical VLM disagreements remain intact.

Independent AI review had accepted the exact source, whole output population, 20-pixel protocol and current protected CI before these four calls. Review of this terminal failure was pending when this proof was prepared; this page is not approval or merge readiness. The visual qualification gate is unmet. Raw operational observations remain private. Image source-readiness and public-image release are separate decisions; this publishes evidence media only, not a container, native model or vendor payload.

The underlying photographs are from NVIDIA and contributors' PhysicalAI-NuRec-PPISP dataset, revision `2521064a3af6ab1c1caa2ba1b01ddde7eecded69`, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Rendered and synthetic-control derivatives are explicitly identified above. SHA-256 values in the machine-readable summary use uppercase uniformly; this lossless encoding preserves all digest bits. Original raw artifacts were not rewritten.
