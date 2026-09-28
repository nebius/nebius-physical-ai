# CUDA 13 base for Blackwell, including RTX PRO 6000

**Adopting FA4 on RTX PRO 6000? Start with the
[application-image guide](../../../../../docs/workbench/guides/rtx6000-fa4.md).**
It covers the local build, application Dockerfile, dependency constraints,
explicit model integration, GPU checks and deployment by digest.

`cuda13-blackwell` is the canonical shared image-family name for RTX PRO 6000,
B200 and B300 build targets. Select the physical GPU in the workload's resource
configuration. Architecture coverage is separate from a measured capability
result on that GPU; the current FA4 qualification is specific to RTX PRO 6000.

This directory builds `npa-base:cuda13-blackwell-dev-<full-source-sha>` when
`build.sh --tag dev-<full-source-sha>` is used. That is a local artifact until
published through the appropriate registry process. The base contains the
Python/CUDA/FA4 environment and validation scripts; add application code,
dependencies, launch commands and scheduler bootstrap in the derived image.

For existing consumers, `base/cuda13-b300` is a symlink to this directory.
The build script tags the same image as both `cuda13-blackwell-<suffix>` and
`cuda13-b300-<suffix>`. `--registry` adds both registry references, and `--push`
pushes both, using `DOCKER_CONTEXT` when set. Already-published image references
and recorded validation identities are unchanged; the rename publishes nothing.

See [FA4 pins, restrictions and measured evidence](../../../../../docs/workbench/flash-attention.md)
and the [public image catalog](../../../../../docs/workbench/container-image-catalog.md)
for the distinction between source recipes and accepted published images.
