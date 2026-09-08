# Third-party notices

## Physical Intelligence OpenPI

- Source: <https://github.com/Physical-Intelligence/openpi>
- Pinned revision: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- License: Apache License 2.0

The upstream license file is retained at `/opt/byof/LICENSE`.

## TensorFlow and cuDNN Frontend adapters

TensorFlow 2.15.0 and TensorFlow CPU 2.15.0 use
[Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/v2.15.0/LICENSE).
Their full licenses remain at
`/opt/venv/lib/python3.11/site-packages/tensorflow-2.15.0.dist-info/LICENSE` and
`/opt/venv/lib/python3.11/site-packages/tensorflow_cpu-2.15.0.dist-info/LICENSE`.
The complete bundled third-party notices remain at
`/opt/venv/lib/python3.11/site-packages/tensorflow/THIRD_PARTY_NOTICES.txt`.

The TensorFlow include tree contains fifteen Apache-2.0 TensorFlow/XLA
cuDNN adapters and 52 copies of NVIDIA cuDNN Frontend 0.8.0 headers. The latter
carry the full copyright and permission notice from
[cuDNN Frontend's MIT license](https://github.com/NVIDIA/cudnn-frontend/blob/8f488bd41229aa0a3d5f7c0168f59e4d69c618ee/LICENSE.txt)
inside each header. All 67 headers and the three license/notice files are
required by exact path, SHA-256 and size in `runtime-payload.json`. This
permissive frontend license does not license the separate cuDNN SDK payload.

## NVIDIA CUDA and cuDNN

The final image inherits the CUDA 12.8.1 runtime base without cuDNN development
payload. CUDA's license and container notices remain applicable. The unmodified
Linux `cuobjdump` utility comes from the separately pinned CUDA development
stage under the CUDA supplement's Linux distribution grant.

cuDNN 9.10.2.21 shared libraries are retained only as OpenPI application
components. Their full upstream license remains at
`/opt/venv/lib/python3.11/site-packages/nvidia_cudnn_cu12-9.10.2.21.dist-info/licenses/License.txt`.
Its SDK headers/static archives are excluded before the installation layer
commits. The image's applications provide material policy inference, training,
and evaluation functionality; these vendor files are not an independent SDK
distribution. Applicable NVIDIA use, distribution and intellectual-property
restrictions remain in force.

## NVIDIA NCCL and NVSHMEM

NCCL 2.27.5 is BSD-3-Clause. Its full license and NVTX reference remain at
`/opt/venv/lib/python3.11/site-packages/nvidia_nccl_cu12-2.27.5.dist-info/licenses/License.txt`.

NVSHMEM 3.2.5 is governed by the NVIDIA SDK agreement and NVSHMEM product
supplement. The complete product license, including its third-party notices,
is retained at `/usr/share/doc/npa-openpi/NVSHMEM-LICENSE.txt`. This supplements
the wheel's CUDA license; package metadata alone does not grant redistribution.
