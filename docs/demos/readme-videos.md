# README demo videos

The root README embeds short, silent H.264 clips hosted as GitHub attachments.
The workflow links are maintained recipes for the demonstrated capabilities;
the clips are retained run outputs, not fresh validation of the current images.

| Clip | Length | Workflow | Presentation |
| --- | --- | --- | --- |
| [Cosmos3 appearance fanout](https://github.com/user-attachments/assets/95917c76-7920-420f-af08-c13590a36a3f) | 8 seconds | [PAIDF + Cosmos3](../../workflows/main/paidf-cosmos3.yaml) | Original source plus twelve labeled appearance variants on the same timeline. |
| [MJLab G1 locomotion](https://github.com/user-attachments/assets/7e605a6d-b32b-4746-808c-51258a11b21a) | 8 seconds | [MJLab render](../../workflows/testing/mjlab-render.yaml) | Seconds 2–10 of a retained native policy rollout; empty space above the scene cropped, resized to 1280 × 540. |
| [SAM 2.1 segmentation](https://github.com/user-attachments/assets/f629401a-346d-4862-b221-34b5ecc28128) | 6 seconds | [SAM 2.1 video masks](../../workflows/testing/byof-sam2.1.yaml) | First six seconds of native GPU mask predictions over generated input, re-encoded for playback. |

## Sources and interpretation

The fanout source is episode 0, camera `observation.images.cam_high`, seconds
0–8 of [LeRobot's ALOHA cups dataset](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/tree/d793c969cf716001dcca18a0842c3d7e9de9e41b),
published under MIT. The dataset credits Tony Zhao, Vikash Kumar, Sergey Levine,
and Chelsea Finn's [ALOHA work](https://tonyzhaozh.github.io/aloha/).
The twelve outputs illustrate appearance diversity. The retained evaluation
rejected the batch: two of twelve candidates passed its clip-level criteria.
They do not establish faithful contacts or suitability for policy training.
The current [ALOHA fanout recipe](../../workflows/testing/paidf-aloha-cups-fanout.yaml)
keeps rejected candidates for review; its single-pass configuration differs
from this retained comparison.

The G1 clip uses native MJLab policy inference, MuJoCo-Warp physics, and EGL
rendering with a retained checkpoint. It shows simulation behavior and makes
no real-robot or performance benchmark claim.

The SAM clip shows prompted video mask propagation. Its colored overlay is a
model prediction over generated footage, not a ground-truth annotation.

## Published clip hashes

SHA-256 identifies the exact uploaded playback files, including their edits and
compression. Full decoding was checked before publication.

| Clip | SHA-256 |
| --- | --- |
| Cosmos3 fanout | `5c27b77ce2788b8ab0121da66db8707a7377d6e784ce152ef6a0de6e4f467848` |
| MJLab G1 | `ba0baad8a4f1126391216808cdac162c6acee97c959a110bde9e37467568c4d3` |
| SAM 2.1 | `2245457ce1057f81bf9b0c04eca3590980cf876ff95a7b344f05c435a106a52c` |

## ALOHA source notice

The source dataset declares MIT licensing. The upstream
[ALOHA license](https://github.com/tonyzhaozh/aloha/blob/main/LICENSE) is retained
below with its attribution.

```text
MIT License

Copyright (c) 2023 Tony Z. Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
