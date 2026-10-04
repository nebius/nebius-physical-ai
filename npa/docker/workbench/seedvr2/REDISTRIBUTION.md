# SeedVR2 redistribution record

Status: `public` eligibility, publication quarantined pending built-image and
real-capability acceptance.

This record is engineering classification, not legal advice.

## Six artifact boundaries

1. Source: the image bakes ByteDance-Seed/SeedVR at
   `e4de8c24441a67e1b7df56abea10645059bb1185`. Upstream's repository license
   and file headers are Apache-2.0.
2. Baked runtime: extension compilation uses a digest-pinned NVIDIA CUDA 13.0
   development build stage that is absent from the final manifest. The final
   image uses the digest-pinned cuDNN runtime base, current PyTorch/CUDA runtime
   wheels, pinned Python dependencies, compiled FlashAttention/Apex extensions,
   and Apache-2.0 SeedVR source. cuDNN wheel SDK headers and static archives are
   removed before the runtime-only environment is copied; the retained shared
   libraries and notice are hash-inventoried in
   `/usr/share/doc/npa-seedvr2/cudnn-runtime.json`. License notices are retained
   in installed distributions and `THIRD_PARTY_NOTICES.md`. The PyTorch
   closure's NVSHMEM SDK headers, device archive, and bitcode are also removed;
   retained shared runtimes, the wheel license, and the exact v3.4.5-0 product
   license containing DF-NVSHMEM/Sandia and other bundled third-party terms are
   inventoried in `/usr/share/doc/npa-seedvr2/nvshmem-runtime.json`. Section 2
   of the product-specific NVSHMEM supplement identifies any portion of the SDK
   as distributable under its agreement; the image retains the full terms and
   supplies material application functionality around the runtime. No SeedVR
   checkpoint, customer media, credential, or populated model cache is baked.
   One source-hash-bound integration patch replaces the pinned entrypoint's
   unusable public TorchVision video import with a narrow NPA-authored PyAV
   decoder. Its original, patched, and adapter hashes are retained in
   `/usr/share/doc/npa-seedvr2/video-io-compat.json`; the model path remains the
   official upstream implementation.
   The ML environment's pip installer is removed in an intermediate export
   stage before any final-image copy. Final installed-dependency validation
   still runs with a read-only build-only installer mount. Inference and the
   pinned model downloader do not install ML packages at runtime. The separate
   NPA service environment retains its installer for workflow bootstrap; this
   is not a claim that the whole image is installer-free or vulnerability-free.
   The service installer is the source-identified `pip==26.2.1+npa.1`
   derivative documented in `../common/secure_pip/README.md`, not an upstream
   pip release. Ubuntu's pip and setuptools seed wheels are replaced by the
   derivative and pinned setuptools 80.9.0 in the same package-install layer,
   before any final layer export. Standard Python 3.12 venv creation uses this
   repaired seed, and the exact build receipt is retained at
   `/usr/share/doc/npa-seedvr2/secure-pip-build.json`. Temporary upstream
   bootstrap and vendoring tools remain confined to a separate build stage.
3. Weights: `ByteDance-Seed/SeedVR2-3B` revision
   `37255ff8cccfb01071b87f635a5948ca8d53117c` is public and marked
   Apache-2.0. The four required payloads are fetched at runtime and verified
   against fixed SHA-256 hashes. Keeping them outside the image reduces a public
   image by about 14.6 GB and binds each run to the exact model bytes; it is not
   an access or license gate.
4. Datasets and inputs: none are baked. Operators supply an S3 video under
   their own rights. The lane's execution proof uses a separately attributed
   CC-BY-4.0 RoboPro `roboreal_all_80tasks` robot-kitchen video, fetched outside
   the image. Its real-world capture origin is unverified; the immutable
   quality pack describes synthetic footage.
5. Runtime caches: Hugging Face payloads use the shared operator-selected cache
   environment. The default cache is node-local ephemeral. Credentials and
   cache bytes never enter a later image layer or output manifest.
6. Outputs: the reviewed SeedVR and model terms state no output field-of-use
   restriction. Input rights still apply. NPA labels restored video as derived
   review media and does not claim generated detail is observed sensor truth.

## Publication state

`npa.deploy.images.UNVALIDATED_PUBLICATION_TOOLS` contains `seedvr2`. Do not
publish an official development or release tag until all secure-image gates,
complete layer/config/SBOM scans, anonymous digest verification, real restoration on the declared, qualified H100 or B200 target, S3 workflow execution, objective preservation gates, real VLM
review, and independent review pass for the exact candidate commit and digest.

An operator-controlled private validation image proves only the operator's run.
It does not authorize official publication or establish release acceptance.
Scan every immutable layer, including bytes hidden by whiteouts, with
`npa/scripts/scan_image_seedvr2_payload.py <image-ref>` and retain its JSON
report alongside the broader security/SBOM scans. The report separately binds
each compressed blob digest and uncompressed rootfs diff ID, the config and OCI
manifest graph, the complete saved archive, and the scanner implementation.
Python `.pth` files fail closed except for the exact path and bytes of the
inventoried setuptools and Rerun bootstrap files. The publication workflow
runs this gate before push and again against bytes pulled by immutable digest.

### MCAP service dependency notice

The runtime includes the `mcap==1.4.0` Python package under MIT. Its installed
distribution omits the full grant, so the final image separately includes the
unchanged Foxglove Technologies notice at
`/usr/share/doc/npa-seedvr2/MCAP-1.4.0-LICENSE.txt`.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#mcap-python-140) binds the
upstream release source and exact notice hash.
