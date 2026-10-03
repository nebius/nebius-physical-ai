# Third-party notices

## Isaac Lab-Arena 0.3.0

- Source: <https://github.com/isaac-sim/IsaacLab-Arena>
- Commit: `ed0fd12be862078be316c73eb7cf423ba9b1c5cd`
- License: Apache License 2.0
- License text: `/opt/isaac-arena/LICENSE.md`

The upstream documentation and tests are not included in this image. The
official immutable source archive is checksum-verified during the build. NPA
then applies context-bound NPA integration changes: restore the replay's initial
state, retain run-bound simulator traces and pre-action revolute state, capture
viewport frames before automatic episode reset, and mask unused embodiment
camera observations. The real upstream renderer and task success calculation
remain in authority. Modified source retains its upstream notices and identifies
the integration changes; the build helper is removed from the final image.

## Lightwheel SDK 1.0.3

- Distribution: <https://pypi.org/project/lightwheel-sdk/1.0.3/>
- License: Apache License 2.0
- Wheel SHA-256: `841ec064ab21a403de024e1e860541e9949e0ea2330d51961b1fdf49d0ec21cd`
- Wheel METADATA SHA-256: `9c6ca9f214143e66b7ca8827b730a3b261437ae43ffd7ef494e5225f2dc4e850`
- License evidence: the exact wheel's package description contains an
  Apache-2.0 license notice and URL; several client modules repeat the notice.
  It has no standalone license member or structured license metadata field.
- Machine-readable evidence: `/opt/npa/docker/workbench/isaac-arena/license-evidence.json`;
  the image build reads the installed distribution's actual `METADATA` bytes,
  checks their hash and notice, and fails if the absent standalone-license or
  structured-license-field shape changes.
- Full license text shipped with the image: `/opt/isaac-arena/LICENSE.md`
- Copyright notice: Copyright 2025 Lightwheel Team; retained in the installed
  distribution metadata and the applicable client modules

The SDK is the upstream-declared client needed to resolve certain Arena assets.
The public image includes the SDK but no Lightwheel registry object. Registry
USDs and generated layouts are provider-controlled runtime downloads; NPA does
not redistribute them or grant rights to them.

The pinned wheel also contains nested copies of its public client build tree
and Python tests. These are included in the installed-byte security and license
inspection; they are not registry assets or evidence of runtime asset rights.
PyPI publishes no source distribution or standalone license artifact for this
release. The redistribution basis is therefore the exact wheel's own immutable
metadata and repeated module notices, not an inferred repository license. A
different wheel or metadata hash requires a new review and fails the image build.

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

## Inherited open-source dependency fixtures

The pinned parent environment includes Newton 1.2.1 and ONNX 1.21.0 with their
packaged examples and test fixtures. Newton includes example USD assets and
`newton/examples/assets/anymal_walking_policy.pt`; ONNX includes conformance
models under `onnx/backend/test/data`. Their distribution metadata declares
Apache-2.0. Full license texts remain in the image's Python environment at
`newton-1.2.1.dist-info/licenses/LICENSE.md` and
`onnx-1.21.0.dist-info/licenses/LICENSE`, with Newton's additional bundled
notices under `newton/licenses` and its distribution license directory.

These are public dependency fixtures inherited unchanged from the pinned base.
They are not Arena evaluation policy checkpoints, replay datasets, provider
registry objects, or evidence of task success. The runtime-only boundary for
operator inputs and Lightwheel assets remains separate from these packages.

## NVIDIA Isaac Sim and Isaac Lab

Neither is included in image layers. The pinned Isaac Lab 3.0.0b2.post1 wheel
(`sha256:dd32886588479ffd70f7348019aeac1582eb9ad16c40244f41d9f458f96f73c3`)
has `METADATA` hash
`365d9b867dddc244d5aebe02becfe7a9bb8afea8fcdbfef7a4a9fb0a0266fae0`
and declares `License: BSD-3-Clause`; it has no standalone license member. The
runtime bootstrap hash-verifies the wheel and rejects a different installed
license field. Its Isaac Sim and proprietary runtime dependencies have
separate NVIDIA Omniverse, Isaac Sim additional software, and NVIDIA software
license terms. The inherited `npa-isaac-lab` bootstrap fetches these runtime
components only after the operator's acceptance of those applicable terms;
the Lab wheel's BSD license does not replace them.

## NVIDIA viewport graphics userspace (runtime only)

Driver libraries and OptiX weights are not included in image layers. A viewport
evaluation prefers native headless EGL/Vulkan, `libnvoptix.so.1`, and readable
regular nonempty weights at `/usr/share/nvidia/nvoptix.bin`, the path documented
in [NVIDIA's driver component reference](https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/README/installedcomponents.html).
If dependencies are absent, NPA downloads the exact loaded-driver version of Ubuntu's signed
`libnvidia-gl-<branch>-server` package, validates its package and ICD metadata,
and extracts it into run-private scratch without installing it on the node. NPA derives a
canonical private EGL ICD because NVIDIA documents EGL as the headless Vulkan
entrypoint. The image contains an empty user-owned weights directory; missing
weights are copied only into the verified private root overlay, atomically
without overwriting and with hash readback. Symlink and submount destinations
are refused. Copied weights remain only for the worker container lifetime;
they are never baked, published as artifacts, or redistributed. The package
remains governed by NVIDIA's driver terms. Readiness checks do not establish
successful denoising, which requires actual runtime validation.
