# Third-party notices

## Isaac Lab-Arena 0.3.0

- Source: <https://github.com/isaac-sim/IsaacLab-Arena>
- Commit: `ed0fd12be862078be316c73eb7cf423ba9b1c5cd`
- License: Apache License 2.0
- License text: `/opt/isaac-arena/LICENSE.md`

The upstream documentation and tests are not included in this image. The
official immutable source archive is checksum-verified during the build.

## Open-source evaluation dependencies

Arena imports its embodiment registry before selecting an evaluation policy.
The image therefore includes a hash-locked Python dependency closure for
Pinocchio (`pin`), Pink (`pin-pink`), ONNX Runtime, DAQP, and QPSolvers. Their
transitive open-source packages and installed license metadata are retained in
the Python environment and recorded in the image SBOM. Authoritative project
and license metadata is available from:

- <https://github.com/stack-of-tasks/pinocchio>
- <https://github.com/stephane-caron/pink>
- <https://github.com/microsoft/onnxruntime>
- <https://github.com/darnstrom/daqp>
- <https://github.com/qpsolvers/qpsolvers>

## NVIDIA Isaac Sim and Isaac Lab

Not included in image layers. They download at runtime from NVIDIA under the
operator's acceptance of the NVIDIA Omniverse, Isaac Sim additional software,
and NVIDIA software license terms documented by the inherited
`npa-isaac-lab` bootstrap.
