# Habitat-Sim third-party notices

This candidate contains the exact PBR resources selected from
`facebookresearch/habitat-sim@57ee4941dc4765240f0f91f70b2c97a919bf9038`.
The original upstream notice is retained without modification at
`/usr/src/habitat-sim/data/pbr/license.txt`.

## BRDF lookup texture — MIT

`data/pbr/bluts/brdflut_ldr_512x512.png` comes from Sascha Willems' Vulkan
physically based rendering work. Habitat-Sim's upstream notice identifies its
license as MIT and links to the original project and license:

- <https://github.com/SaschaWillems/Vulkan-glTF-PBR>
- <https://github.com/SaschaWillems/Vulkan-glTF-PBR/blob/master/LICENSE>

## Poly Haven environment maps — CC0 1.0

The following files are dedicated to the public domain under CC0 1.0:

- `data/pbr/env_maps/anniversary_lounge_1k.hdr`
- `data/pbr/env_maps/autoshop_01_1k.hdr`
- `data/pbr/env_maps/blue_photo_studio_1k.hdr`
- `data/pbr/env_maps/brown_photostudio_02_1k.hdr`
- `data/pbr/env_maps/lythwood_room_1k.hdr`

The upstream notice identifies their original records at
<https://polyhaven.com/a/anniversary_lounge>,
<https://polyhaven.com/a/autoshop_01>,
<https://polyhaven.com/a/blue_photo_studio>,
<https://polyhaven.com/a/brown_photostudio_02>, and
<https://polyhaven.com/a/lythwood_room>. Poly Haven's license statement is at
<https://polyhaven.com/license>; the CC0 1.0 legal tool is at
<https://creativecommons.org/publicdomain/zero/1.0/>.

The Habitat-Sim PBR configuration files remain covered by Habitat-Sim's root
MIT license. These PBR resources are image source/runtime support bytes, not
the separately licensed Skokloster demo scene, which remains runtime-fetched.

## Downstream packaging metadata modification

NPA changes the pinned Habitat-Sim dependency declaration by retaining
`pillow==12.3.0` and omitting the optional Matplotlib/Numba/SciPy scientific
stack from the bootstrap image. The omitted `contourpy`, `fonttools`,
`numpy-quaternion`, `kiwisolver`, `llvmlite`, `matplotlib`, `numba`, and `scipy` rows remain
hash-locked and are fetched only into a private run-owned cache when the smoke
workload actually needs them. This metadata-only/runtime-delivery design keeps
their bytes out of the public bootstrap image; it does not change Habitat-Sim
runtime source or grant redistribution rights. The source manifest binds the
complete upstream file, exact preimage and postimage, final patched file, and
resulting source-projection inventory by SHA-256.
