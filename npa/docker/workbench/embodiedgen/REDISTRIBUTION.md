# EmbodiedGen image delivery boundary

This recipe builds an **operator-private** CUDA development image. The pinned
NVIDIA CUDA development base is not placed in an anonymous registry or promoted
by this repository. NVIDIA's CUDA Toolkit terms govern that base and must be
reviewed by the operator for the selected delivery channel.

The image contains NPA bootstrap code, the runtime manifest, notices, and the
ordinary worker prerequisites needed to compile upstream extensions. It does
not contain EmbodiedGen source, TRELLIS source, model weights, Python CUDA
packages, upstream sample media, customer inputs, credentials, caches, or
outputs. Those bytes are fetched into an operator-owned writable cache only at
execution time.

EmbodiedGen V2.1.0 source is Apache-2.0. TRELLIS source and the selected
`microsoft/TRELLIS-image-large` model revision are MIT. The VLM property
estimator is used only to estimate URDF metadata; its results are not calibrated
physical measurements. Runtime fetching changes delivery, not the upstream
license, output, service, or commercial-use rights.
