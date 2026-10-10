# MolmoAct2 Jetson-Thor evidence worker notices

This private operator validation image carries the Nebius Physical AI adapter
source from the surrounding repository under Apache-2.0.

It deliberately does **not** carry `vla-edge`, the Agents2Agents
`MolmoAct2-LIBERO-Jetson-Thor` bundle, TensorRT engines or plugins, model
weights, JetPack, TensorRT, Jetson PyTorch, calibration captures, or populated
caches. Those artifacts are obtained and used only by the operator's native
Jetson AGX Thor runtime. Their notices, attribution, provenance, and terms are
recorded in `docs/workbench/molmoact2-jetson-thor.md` and travel with the
operator-fetched artifacts rather than this image.

The image installs `boto3`, `numpy`, and `rerun-sdk` only to move, evaluate,
and visualize artifact evidence. It also installs Debian's `openssh-server`,
`rsync`, and `sudo` solely for the documented SkyPilot non-root task bootstrap.
Their installed distribution metadata is part of the image SBOM produced during
private qualification.
