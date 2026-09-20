# RoboTwin bootstrap redistribution boundary

This unbuilt recipe is designed to contain only an unmodified official Ubuntu
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

The recipe is deliberately unbuilt and publication-quarantined. The trusted
build path refuses before Docker while the exact native-content policy remains
unresolved. Exact OCI bytes must later pass license, source-availability, SBOM,
provenance, vulnerability, secret, complete-byte, and RoboTwin-specific scans.
Redistributability of a future neutral bootstrap is separate from permission to
run the runtime-fetched components.

The operator's exact `noncommercial` statement is recorded once for this
bounded manager run and is compatible only with CuRobo v0.7.8 noncommercial
research/evaluation. It expires with the run and does not authorize hosted
service or broader derivative/output use. Before any governed CUDA, cuDNN, or
CuRobo runtime fetch, install, or cache mutation, a customer representative
authorized to bind that customer must issue an owner-only, run-scoped
entitlement for the exact CUDA 12.8.1, cuDNN 9.8.0, and CuRobo v0.7.8 terms.
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
derivative, or redistribution rights, and none of the still-missing technical
artifact-lock, probe, native-content, built-image, storage/context, or live
evidence gates is waived.
