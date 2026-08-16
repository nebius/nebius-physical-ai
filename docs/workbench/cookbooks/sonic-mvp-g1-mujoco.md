# SONIC G1 MuJoCo Image Status

The historical `sonic-mujoco-h100-mvp` / `npa-sonic-mujoco:0.1.3-mvp`
variant is quarantined. It inherits `npa-sonic:0.1.2`, whose built bytes include
an old `nvcr.io/nvidia/isaac-lab` base and baked NVIDIA driver libraries.

Do not submit, mirror, or publish that variant. The resolver intentionally
rejects `h100`, `h200`, `mujoco`, and the explicit legacy variant ID. Supplying
credentials or EULA acceptance at runtime cannot change the licensing of bytes
already baked into the image.

A replacement must:

1. Build from the active runtime-fetch SONIC base without NVIDIA Isaac or driver
   payloads.
2. Add only redistributable MuJoCo/EGL dependencies.
3. Pass the built-byte Omniverse payload scan and no-baked-consent checks.
4. Record a new additive tag and immutable digest in
   `npa/src/npa/deploy/sonic_image_manifest.json`.
5. Pass real GPU training/evaluation validation before its status becomes
   `active`.

Until those gates pass, use the active RTX PRO 6000 Kubernetes SONIC workflow
where its supported evaluation path is sufficient.
