# CUDA 13 base for Blackwell, including RTX PRO 6000

**Adopting FA4 on RTX PRO 6000? Start with the
[application-image guide](../../../../../docs/workbench/guides/rtx6000-fa4.md).**
It covers the local build, application Dockerfile, dependency constraints,
explicit model integration, GPU checks and deployment by digest.

`cuda13-b300` is the historical shared image-family name. The build targets
include B200, B300 and RTX PRO 6000; the name is retained for existing build
scripts and derived-image references. Select the physical GPU in the workload's
resource configuration. Architecture coverage is separate from a measured
capability result on that GPU.

This directory builds `npa-base:cuda13-b300-dev-<full-source-sha>` when
`build.sh --tag dev-<full-source-sha>` is used. That is a local artifact until
published through the appropriate registry process. The base contains the
Python/CUDA/FA4 environment and validation scripts; add application code,
dependencies, launch commands and scheduler bootstrap in the derived image.

See [FA4 pins, restrictions and measured evidence](../../../../../docs/workbench/flash-attention.md)
and the [public image catalog](../../../../../docs/workbench/container-image-catalog.md)
for the distinction between source recipes and accepted published images.
