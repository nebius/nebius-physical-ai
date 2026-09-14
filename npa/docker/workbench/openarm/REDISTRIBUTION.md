# OpenArm container redistribution record

Review date: 2026-09-14. This is an engineering classification, not legal advice.

The public image bakes Enactic `openarm_mujoco` 2.2.0 at commit
`a8c979629f2591ad035d99d338ce114969e6cddc` and `openarm_isaac_lab` at commit
`bad82e23716e6941c2de78ccb978f57c78b37734`. Both upstream repositories include
the Apache License 2.0 and their robot model assets in the licensed source tree.
The image retains both license texts and does not copy hardware CAD from the
separately licensed OpenArm hardware repository.

MuJoCo 3.6.0 and its complete additive dependency closure are installed from a
hash-locked requirements file. NVIDIA Isaac Sim, Isaac Lab wheels, Omniverse Kit,
runtime caches, credentials, operator data, and generated artifacts are absent.
At first Isaac use, the shared NPA bootstrap applies the operator-owned acceptance
and refusal contract and fetches exact, hash-pinned Isaac Sim 5.1.0.0 and Isaac Lab
2.3.2.post1 wheels into a writable cache. Privacy and telemetry consent remain off.

Public release requires a full-filesystem and layer-history scan of the exact
published digest with `scan_image_omniverse_payload.py`, an SBOM/license review,
anonymous-pull verification, and real MuJoCo plus RTX Isaac Lab qualification.
