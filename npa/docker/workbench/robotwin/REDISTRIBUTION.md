# RoboTwin bootstrap redistribution boundary

This bootstrap contains only an unmodified official Ubuntu
base layer, exact unmodified Ubuntu `main` packages, and NPA-authored bootstrap
files. It does not contain RoboTwin, CuRobo, CUDA, cuDNN, PyTorch CUDA, SAPIEN,
MPLib, Warp, asset, cache, credential, or output bytes.

The linux/amd64 Ubuntu 22.04 manifest and Ubuntu snapshot package/source closure
are immutable in `runtime-lock.json` and `apt-packages.lock`. The selected 84
binary packages all come from Ubuntu `main`; their 84 installed copyright files
are individually hash-bound and must remain in the image. Ubuntu's policy leaves
each component's license in force. `npa-robotwin` does not use an Ubuntu mark in
its software title or imply Canonical endorsement; Ubuntu is referenced only to
identify the unchanged base and package origin.

The neutral bootstrap builds without customer runtime credentials. The trusted
public development workflow requires the exact public-content policies, complete
native byte scan, payload scan, license/security/SBOM gates, and a publicly
available corresponding-source annex before pushing any image. Supported release
promotion remains quarantined until the separate real RTX capability gate passes.

The source annex includes 280 hash- and size-locked official archives covering
91 source package versions across every Ubuntu ancestor and installed layer.
`source_annex.py --output-dir DIR` reproduces the archive from
`corresponding-sources.json`; `source-bundle.json` binds its deterministic bytes.
It is delivered as `robotwin-corresponding-sources.tar` under the public
`robotwin-sources-dev-<full-source-sha>` GitHub prerelease. The accompanying
`source-manifest.json` binds that revision; the publication receipt adds the
resulting image digest. Retain the annex for as long as its image is retained.
Package-specific licenses and build instructions are included in those complete
upstream source archives. Runtime-fetched vendor components are separate.

The operator's exact `noncommercial` statement is recorded once for this
bounded manager run and is compatible only with CuRobo v0.7.8 noncommercial
research/evaluation. It expires with the run and does not authorize hosted
service or broader derivative/output use. Before any governed CUDA, cuDNN, or
CuRobo runtime fetch, install, or cache mutation, a customer representative
authorized to bind that customer must issue an owner-only, run-scoped
entitlement for the exact CUDA 12.8.1, cuDNN 9.7.1, and CuRobo v0.7.8 terms.
The record is bound to customer, run, runtime-lock SHA-256, intended activity,
exact terms, and expiry; NPA and the infrastructure manager do not accept or
sign those terms for the customer.

The exact public, ungated RoboTwin2.0 revision declares MIT for both locked
archive members. It requires an anonymous exact-revision payload-byte probe,
not a credential or local acceptance flag. If later artifacts are token-gated,
only the customer's vendor-side entitlement and exact payload probe may gate
their runtime delivery. The inspected authoritative terms impose no generated-
output restriction on the declared HDF5, MP4, frame, or JSON evidence; CuRobo's
underlying execution remains limited to noncommercial research/evaluation.
A credential, private registry, runtime fetch, or writable destination is not
permission. Customer entitlement does not grant broader service, output,
derivative, or redistribution rights, and none of the probe, native-content, built-image, storage/context, or live
evidence gates is waived. The runtime artifact lock is complete; real
installation and GPU capability remain customer-authorized validation steps.
