# flex-pi redistribution record

Review date: 2026-09-16. Engineering classification, not legal advice.

The image is `public`. It contains flex-pi source at immutable commit
`20c1b2b71ea35a415d5d47c39b04443cfadad7a1` under MIT, the digest-pinned
upstream PyTorch CUDA 12.8 runtime, Python dependencies, and NPA's Apache-2.0
integration. Only the package source, configs, RoboTwin policy adapter, and
license needed for inference are copied; the vendored simulator trees are not.

The `flex-pi/flexpi-robotwin` checkpoint is MIT-labelled. The public
`flex-pi/robotwin_3d` dataset card declares no license, so NPA does not infer
one from the MIT RoboTwin simulator source and never redistributes sample bytes.
Wan/T5 converted weights are fetched from the Apache-2.0 ModelScope repository
at a pinned revision, while DINOv3 uses its separate model license through timm
at a pinned revision. All are runtime-only, under the operator's accepted terms
and authorized use. None of those weights,
observation media, populated caches, credentials, or generated actions are
present in image layers.

Publication requires a complete built-layer scan, dependency/security review,
anonymous digest resolution, and real action inference on the exact image digest
using an RTX PRO 6000. A model import or dry run is not capability evidence.
