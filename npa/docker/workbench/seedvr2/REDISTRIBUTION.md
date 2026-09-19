# SeedVR2 redistribution record

Status: `public` eligibility, publication quarantined pending built-image and
real-capability acceptance.

This record is engineering classification, not legal advice.

## Six artifact boundaries

1. Source: the image bakes ByteDance-Seed/SeedVR at
   `e4de8c24441a67e1b7df56abea10645059bb1185`. Upstream's repository license
   and file headers are Apache-2.0.
2. Baked runtime: the image uses a digest-pinned NVIDIA CUDA 13.0 development
   base, current PyTorch/CUDA runtime wheels, pinned Python dependencies,
   FlashAttention source, and NVIDIA Apex source. Their license notices are
   retained in installed distributions and `THIRD_PARTY_NOTICES.md`. No SeedVR
   checkpoint, customer media, credential, or populated model cache is baked.
3. Weights: `ByteDance-Seed/SeedVR2-3B` revision
   `37255ff8cccfb01071b87f635a5948ca8d53117c` is public and marked
   Apache-2.0. The four required payloads are fetched at runtime and verified
   against fixed SHA-256 hashes. Keeping them outside the image reduces a public
   image by about 14.6 GB and binds each run to the exact model bytes; it is not
   an access or license gate.
4. Datasets and inputs: none are baked. Operators supply an S3 video under
   their own rights. The lane's real proof uses a separately attributed
   CC-BY-4.0 RoboPro capture, fetched outside the image.
5. Runtime caches: Hugging Face payloads use the shared operator-selected cache
   environment. The default cache is node-local ephemeral. Credentials and
   cache bytes never enter a later image layer or output manifest.
6. Outputs: the reviewed SeedVR and model terms state no output field-of-use
   restriction. Input rights still apply. NPA labels restored video as derived
   review media and does not claim generated detail is observed sensor truth.

## Publication state

`npa.deploy.images.UNVALIDATED_PUBLICATION_TOOLS` contains `seedvr2`. Do not
publish an official development or release tag until all secure-image gates,
complete layer/config/SBOM scans, anonymous digest verification, real H100
restoration, S3 workflow execution, objective preservation gates, real VLM
review, and independent review pass for the exact candidate commit and digest.

An operator-controlled private validation image proves only the operator's run.
It does not authorize official publication or establish release acceptance.
Scan every immutable layer, including bytes hidden by whiteouts, with
`npa/scripts/scan_image_seedvr2_payload.py <image-ref>` and retain its JSON
report alongside the broader security/SBOM scans.
