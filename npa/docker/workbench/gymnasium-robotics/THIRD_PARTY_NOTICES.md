# Third-party notices — neutral-bootstrap inventory

This file records the runtime provenance and notice-retention contract. The
listed upstream payloads are not present in the neutral candidate layers, and
this inventory is not a license grant or a substitute for licenses that must
accompany runtime-fetched material.

- Farama Foundation Gymnasium-Robotics 1.4.2 at commit
  `4d1ebecbc6436806cfbc0e42ebc36f594d05844e`: MIT; root license SHA-256
  `00668424e12956742815eb1d8e15c7be543192561511df5fde119ae1188315ef`.
- Runtime-fetched Shadow Hand assets must retain
  `gymnasium_robotics/envs/assets/LICENSE.md`, SHA-256
  `872d90e6cbbe9e4390ea4d9e2598e5f194ec2782afd5af2a4c083b35a8fd451a`,
  plus every file-level notice represented by `asset-lock.json`. The Phase 1
  review conservatively maps retained Shadow-derived material to `sr_common`
  commit `59d6bdf35bd9cf53185a20eb63413fdfe57fe77c` and GPL-2.0-only plus the
  asset notice's Apache-2.0 terms. Preferred-form, transformation, derivative,
  and compatibility disposition remains unresolved; runtime fetch does not
  cure it. The public image carries no such bytes; the customer must complete
  the existing upstream notice and acceptance process before fetching them.
- MuJoCo 3.12.0 at release commit
  `13827e9ee56f097f57acf69ae52b078f9839682d`: Apache-2.0. Exact wheel,
  license, and third-party-notice hashes are recorded in `source-lock.json`.
- Ubuntu Noble neutral-bootstrap packages: mixed free-software licenses. The
  exact signed `20260905T000000Z` snapshot maps all 173 installed binary
  packages to 117 source packages, 370 source artifacts, and 173 installed
  copyright files in `apt-runtime.lock.json`. The public image carries only
  this mapped neutral bootstrap closure; restricted workload material remains
  runtime-fetch-only. The requested
  neutral set includes Ubuntu's system `python3-boto3` only for
  trusted receipt, summary, and artifact bookkeeping; it is not a fetched
  workload wheel and carries no permission for runtime-fetched material.
  Ubuntu libglvnd supplies the generic EGL/OpenGL dispatch libraries through
  `libegl1`, `libopengl0`, and `libglvnd0`; Ubuntu's required Mesa dependencies
  remain in this same signed binary/source/copyright closure. These are generic
  free-software libraries, not the injected NVIDIA driver implementation.

No NVIDIA driver runtime is included. NVIDIA vendor EGL/GL libraries are
supplied by the assigned GPU node at run time and must be observed, never
copied into the image, runtime cache, or output artifact.
