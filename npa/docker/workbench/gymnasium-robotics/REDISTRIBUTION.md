# Gymnasium-Robotics Phase A redistribution boundary

This directory describes a directly redistributable candidate; it does not
describe a built, scanned, GPU-qualified, or publicly available image. The
Dockerfile deliberately refuses before network access while any evidence lock
is incomplete.

The intended image contains pinned Gymnasium-Robotics MIT source, MuJoCo
Apache-2.0 runtime, the packaged Shadow Dexterous Hand assets and notices, and
a minimal Ubuntu runtime. It contains no model, weights, external dataset,
runtime cache, credential, prior output, CUDA toolkit, NGC, Isaac, Omniverse,
or host NVIDIA driver/EGL library.

Public eligibility depends on conveying real corresponding source for every
retained reciprocal binary. In particular, the exact Shadow preferred form,
transformation/build chain, and the complete Ubuntu binary-to-source closure
must be present as bytes in the image's source annex. URLs and written offers
do not satisfy this contract. Until those locks are complete and independently
accepted, no build, publication, or capability claim is permitted.

Operator-created `gymnasium-robotics-smoke.json` telemetry remains an output of
the operator's run. It is not a redistribution grant for upstream material and
must contain no source, asset, credential, or infrastructure bytes.
