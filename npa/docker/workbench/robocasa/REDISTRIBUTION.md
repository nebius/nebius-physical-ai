# RoboCasa replacement image packaging

This recipe targets public development publication. Each new digest still needs
the complete pre-publication security, license, secret and payload checks, then
real execution on its selected physical GPU. This document records the proposed
payload boundary; it does not establish a built or validated image.

The source archives are immutable and checked before extraction:

| Component | Revision | Archive SHA256 |
| --- | --- | --- |
| RoboCasa v1.0 | `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` | `1893328b5222ac0443287e593c161c696c77e3f8018f6d9f6bd900871d2caad3` |
| robosuite | `85abee228d1c43ab1939bce33028099945d453b4` | `2369a5c3385bf122eeff3362e7f70aacf1d10436e860d12026ecd10b37c152cd` |

The unused MIT `render_dataset_with_omniverse.py` adapter is removed in the
same extraction step as its source archive; neither the adapter nor that
archive remains in an image layer. The service uses MuJoCo/EGL.

Both source licenses are MIT with separate Apache-2.0 MuJoCo portions. The
source archives also contain assets: 208 RoboCasa files and 943 robosuite files
under their respective `models/assets` trees. RoboCasa identifies its assets
and datasets as CC BY 4.0. robosuite's bundled Panda and Sawyer descriptions
identify separate Apache-2.0 and BSD terms. `ASSET-NOTICES.md`, both original
source trees and README credits, and the additional complete Apache/Sawyer
license files are retained. Source assets are not relabeled MIT by this image.

The CUDA base is the same digest-pinned, non-cuDNN CUDA 13 base reviewed for
cuRobo. The Python closure uses hash-locked PyTorch 2.13.0 CUDA 13, torchvision
0.28.0, and cuDNN 9.20.0.48. The shared cuRobo filter removes only the reviewed
SDK headers before the wheel-installation layer commits and retains unmodified
shared libraries and notices. The full NVSHMEM product supplement is also
included at a hash-verified path. CUDA, cuDNN, cuSPARSELt, NCCL and NVSHMEM remain
separate license boundaries documented in the
[cuRobo packaging record](../curobo/REDISTRIBUTION.md).
`verify_image.py` reuses its exact complete-layer runtime inventory, changing
only the NVSHMEM notice location to the RoboCasa documentation directory.
All other image-byte and vulnerability checks remain mandatory.

The complete NPA distribution is installed, so the service retains its required
`npa.clients.storage` and `npa.cli.path_contract` imports. The primary dependency
closure is `requirements.lock`; RoboCasa and robosuite source installs skip
their broad training dependencies, as the earlier candidate intended.

`policy-requirements.lock` supplies the exact Apache-2.0 LeRobot 0.5.1 wheel
without installing its whole trainer/environment dependency graph. The image
checks the ACT model, policy configuration and processor-factory import path.
This is an explicitly limited compatibility subset: RoboCasa's broad metadata
requests LeRobot 0.3.3, while LeRobot 0.5.1 declares newer Gymnasium and older
Torch/torchvision, AV, Rerun and setuptools ranges than this runtime. The
runtime deliberately keeps RoboCasa's Gymnasium 0.29.1, the current NPA AV/Rerun
pins, patched CUDA Python wheels, and diffusers 0.38.0, which fixes
[CVE-2026-44513](https://github.com/huggingface/diffusers/security/advisories/GHSA-98h9-4798-4q5v)
and [CVE-2026-45804](https://github.com/huggingface/diffusers/security/advisories/GHSA-7wx4-6vff-v64p).
No wheel metadata is rewritten, and no
successful whole-distribution `pip check` is claimed. ACT training or held-out
policy evaluation requires separate real evidence; a kitchen rollout/export
does not establish those capabilities.

Large kitchen textures, fixtures and objects are fetched only at runtime from
the revision-pinned `robocasa/robocasa-assets` and
`nvidia/PhysicalAI-Kitchen-Assets` repositories. These repositories were
anonymously readable at review; access and their separate terms must still be
checked before a workload. No model weights, checkpoints, credentials,
populated caches or acceptance values are baked. HF and Numba caches are
ephemeral under `/workspace/.cache` unless the operator mounts durable storage.

Simulation output includes real state/action arrays and encoded MP4 frames.
The pinned PyAV runtime encodes videos directly; there is no extra bundled
imageio-ffmpeg executable. Encoding failure makes the capability fail instead
of reporting an absent artifact as completed. Output provenance distinguishes
requested asset revisions from a separately verified cache or dataset identity.
