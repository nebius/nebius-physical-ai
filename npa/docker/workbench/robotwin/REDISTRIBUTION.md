# RoboTwin bootstrap redistribution boundary

This unbuilt recipe is designed to contain only an unmodified official Ubuntu
base layer, exact unmodified Ubuntu `main` packages, and NPA-authored bootstrap
files. It does not contain RoboTwin, CuRobo, CUDA, cuDNN, PyTorch CUDA, SAPIEN,
MPLib, Warp, asset, cache, credential, or output bytes.

The linux/amd64 Ubuntu 22.04 manifest and Ubuntu snapshot package/source closure
are immutable in `runtime-lock.json` and `apt-packages.lock`. The selected 75
binary packages all come from Ubuntu `main`; their 75 installed copyright files
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
service or broader derivative/output use. Runtime use remains refused until a
genuine manager receipt binds that scope plus CUDA and cuDNN delivery/use,
CuRobo's remaining service/output boundary, aggregate RoboTwin asset/output
treatment, and exact provider access evidence. Customer credentials, when an
artifact is gated, are runtime-only secret values and must pass the exact
provider/artifact/revision/terms payload probe. A credential, private registry,
runtime fetch, or writable destination is not permission.
