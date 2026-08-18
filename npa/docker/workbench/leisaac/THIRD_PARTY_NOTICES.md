# LeIsaac image third-party notices

| Component | Source in the image | License / packaging decision |
| --- | --- | --- |
| LeIsaac 0.4.0 | pinned Apache-2.0 source checkout | Apache-2.0; NPA's two reviewable patches are retained beside the Dockerfile and listed in `REDISTRIBUTION.md`. |
| FFmpeg | Debian distribution packages (`ffmpeg` and shared libraries) | GPL/LGPL components under the Debian package copyright files in `/usr/share/doc`; no static or bundled `imageio-ffmpeg` executable is installed. |
| imageio-ffmpeg 0.6.0 | optional `npa[leisaac]` host dependency only | BSD-2-Clause wrapper. Its PyPI wheel may contain a separately licensed static FFmpeg executable, so it is excluded from NPA core and from the LeIsaac runtime image. |
| pygame 2.6.1 | pinned PyPI runtime dependency | LGPL-2.1; source and license terms are published by the pygame project. The unmodified library is dynamically imported by LeIsaac's keyboard device path. |
| aiortc 1.15.0 | pinned PyPI runtime dependency | BSD-3-Clause; exact runtime version is asserted before codec/data-channel use. |
| NVIDIA Omniverse WebRTC streaming client 5.6.0 | operator-fetched runtime cache; never an image layer | NVIDIA proprietary notice: use, reproduction, disclosure, or distribution requires an express NVIDIA license. The served JavaScript and license are hash-verified and unmodified; this component is excluded from the public-image redistribution decision. |
| NVIDIA Kit livestream / StreamSDK | operator-fetched Isaac Sim runtime | NVIDIA proprietary runtime under the operator's Isaac/Omniverse agreement; GPU driver/NVENC and Kit bytes are never redistributed in the image. |

The image remains `redistribution: public`: NVIDIA drivers, NVENC libraries,
Isaac/Kit, the NVIDIA WebRTC client, task assets, scenes, datasets, credentials,
and recordings are runtime-injected or fetched only after the shared
default/opt-out EULA preflight and are rejected by the digest-bound payload
scan.
