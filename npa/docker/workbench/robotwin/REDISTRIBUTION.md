# RoboTwin bootstrap redistribution boundary

This Phase A candidate is designed to contain only an Ubuntu bootstrap closure
and NPA-authored files. It does not contain RoboTwin, CuRobo, CUDA, cuDNN,
PyTorch CUDA, SAPIEN, MPLib, Warp, asset, cache, credential, or output bytes.

The candidate is deliberately unbuilt and publication-quarantined until the
base and apt locks are complete and exact OCI bytes pass license, SBOM,
provenance, vulnerability, secret, complete-byte, and RoboTwin-specific scans.
Redistributability of a future bootstrap is separate from permission to run the
runtime-fetched components.

Runtime use remains refused until the manager records genuine decisions for the
CUDA and cuDNN terms, CuRobo v0.7.8's noncommercial research/evaluation field of
use and service boundary, and the aggregate RoboTwin asset/output rights. A
credential, private registry, runtime fetch, or writable destination is not
permission.
