# Third-party notices — Phase A inventory

This is an inventory and retention contract, not a substitute for the exact
license files that a future authorized build must convey.

- Farama Foundation Gymnasium-Robotics 1.4.2 at commit
  `4d1ebecbc6436806cfbc0e42ebc36f594d05844e`: MIT; root license SHA-256
  `00668424e12956742815eb1d8e15c7be543192561511df5fde119ae1188315ef`.
- The packaged Shadow Hand assets retain
  `gymnasium_robotics/envs/assets/LICENSE.md`, SHA-256
  `872d90e6cbbe9e4390ea4d9e2598e5f194ec2782afd5af2a4c083b35a8fd451a`.
  The provenance review conservatively treats the Shadow `sr_common` source at
  `59d6bdf35bd9cf53185a20eb63413fdfe57fe77c` as GPL-2.0-only plus the
  asset notice's Apache-2.0 terms. Public packaging is blocked until the exact
  preferred-form and transformation-source delivery is accepted.
- MuJoCo 3.12.0 at release commit
  `13827e9ee56f097f57acf69ae52b078f9839682d`: Apache-2.0. Its wheel license
  and third-party notice hashes are recorded in `source-lock.json`.
- Ubuntu Noble runtime packages: mixed free-software licenses. Exact binary,
  copyright, source, and build-material closure is unresolved in Phase A.

No NVIDIA redistributable runtime is included. NVIDIA EGL/GL libraries are
supplied by the GPU node at run time and must be observed, never copied into
the image or an output artifact.
