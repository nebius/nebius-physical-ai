# Third-party notices

## Isaac Lab-Arena 0.3.0

- Source: <https://github.com/isaac-sim/IsaacLab-Arena>
- Commit: `ed0fd12be862078be316c73eb7cf423ba9b1c5cd`
- License: Apache License 2.0
- License text: `/opt/isaac-arena/LICENSE.md`

The upstream documentation and tests are not included in this image. The
official immutable source archive is checksum-verified during the build. NPA
then applies a narrow Apache-2.0-compatible source patch that leaves Kit and
environment rendering enabled while masking unused embodiment-camera
observations; the context-bound build helper is removed from the final image.

## Lightwheel SDK 1.0.3

- Distribution: <https://pypi.org/project/lightwheel-sdk/1.0.3/>
- License: Apache License 2.0
- Wheel SHA-256: `841ec064ab21a403de024e1e860541e9949e0ea2330d51961b1fdf49d0ec21cd`
- License evidence: the installed distribution metadata carries the complete
  Apache-2.0 grant and the installed modules carry Apache-2.0 headers

The SDK is the upstream-declared client needed to resolve certain Arena assets.
The public image includes the SDK but no Lightwheel registry object. Registry
USDs and generated layouts are provider-controlled runtime downloads; NPA does
not redistribute them or grant rights to them.

## Open-source evaluation dependencies

Arena imports its asset and embodiment registries before selecting an evaluation policy.
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

## NVIDIA viewport graphics userspace (runtime only)

Not included in image layers. A viewport evaluation prefers the target's
native headless EGL and Vulkan stack. If those libraries are absent while CUDA
is healthy, NPA downloads the exact loaded-driver version of Ubuntu's signed
`libnvidia-gl-<branch>-server` package, validates its package and ICD metadata,
and extracts it into run-private scratch without installing it. NPA derives a
canonical private EGL ICD because NVIDIA documents EGL as the headless Vulkan
entrypoint. The package is not retained or redistributed, and remains governed
by NVIDIA's driver terms.
