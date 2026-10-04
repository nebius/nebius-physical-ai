#!/bin/sh
# Canonical build command passed verbatim to `npa workbench byof run`.
#
# The generic BYOF builder runs this from the pinned OpenVLA-OFT source at
# /opt/byof. It intentionally installs that source with --no-deps so its
# unpinned moojink/dlimp_openvla declaration is never fetched. The replacement
# source is Apache-2.0 kvablack/dlimp at the immutable revision below.
set -eu

runtime_python=/opt/byof/.venv/bin/python

python3 -m pip install --no-cache-dir virtualenv
python3 -m virtualenv /opt/byof/.venv
"$runtime_python" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"$runtime_python" -m pip install --no-cache-dir \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Do not let this install resolve the source's unlicensed dlimp fork.
"$runtime_python" -m pip install --no-cache-dir --no-deps -e /opt/byof
"$runtime_python" -m pip install --no-cache-dir \
  'accelerate>=0.25.0' draccus==0.8.0 einops huggingface_hub json-numpy \
  jsonlines matplotlib peft==0.11.1 protobuf==3.20.3 rich \
  sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
  'transformers @ git+https://github.com/moojink/transformers-openvla-oft@bc339d9ad707454c0c115970db43c260067c61ab' \
  wandb==0.16.6 tensorflow==2.15.0 tensorflow_datasets==4.9.3 \
  tensorflow_graphics==2021.12.3 diffusers==0.30.3 imageio uvicorn fastapi \
  numpy==1.26.4 imageio[ffmpeg] bddl easydict cloudpickle gym numba scipy \
  mujoco Pillow opencv-python-headless==4.8.1.78 termcolor boto3 \
  rerun-sdk==0.20.3 tensorflow-metadata==1.14.0
# The imageio-ffmpeg wheel carries a static FFmpeg binary. Keep its Python
# plugin but remove that binary from this private image; the renderer base
# supplies /usr/bin/ffmpeg, which imageio-ffmpeg discovers at runtime.
test -x /usr/bin/ffmpeg
find /opt/byof/.venv/lib -path '*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*' \
  -type f -delete
test -z "$(find /opt/byof/.venv/lib -path '*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*' -type f -print -quit)"

git clone --filter=blob:none https://github.com/kvablack/dlimp.git /opt/dlimp
git -C /opt/dlimp checkout --detach 92e3eca97af3b14d0b6aa15182c0dc240407698d
test "$(git -C /opt/dlimp rev-parse HEAD)" = \
  92e3eca97af3b14d0b6aa15182c0dc240407698d
test "$(sha256sum /opt/dlimp/LICENSE | cut -d ' ' -f1)" = \
  c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
test "$(sha256sum /opt/dlimp/dlimp/dataset.py | cut -d ' ' -f1)" = \
  54f7cf110d4ca1300c3c31726ee89d1f02d4105beb0b3520ffa0c0abbda77c42
sed -i 's/options\.deterministic = False/options.deterministic = True/' \
  /opt/dlimp/dlimp/dataset.py
test "$(sha256sum /opt/dlimp/dlimp/dataset.py | cut -d ' ' -f1)" = \
  86bc13c005112961498170f5258639dd52cc46a7614995aff6b8ef6851803d48
printf '%s\n' \
  'Modified by NPA solely to set options.deterministic = True.' \
  'Base: kvablack/dlimp@92e3eca97af3b14d0b6aa15182c0dc240407698d.' \
  'The unmodified Apache-2.0 LICENSE is retained beside this notice.' \
  > /opt/dlimp/NPA_MODIFICATIONS.md
printf '%s\n' \
  '{"license":"Apache-2.0","license_sha256":"c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4","modification":{"from":"options.deterministic = False","path":"dlimp/dataset.py","to":"options.deterministic = True"},"modified_dataset_sha256":"86bc13c005112961498170f5258639dd52cc46a7614995aff6b8ef6851803d48","original_dataset_sha256":"54f7cf110d4ca1300c3c31726ee89d1f02d4105beb0b3520ffa0c0abbda77c42","revision":"92e3eca97af3b14d0b6aa15182c0dc240407698d","schema":"npa.lorafleet.dlimp-runtime/v1","source":"https://github.com/kvablack/dlimp"}' \
  > /opt/byof/npa_lorafleet_dlimp_provenance.json
"$runtime_python" -m pip install --no-cache-dir --no-deps -e /opt/dlimp

git clone --filter=blob:none https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/libero
git -C /opt/libero checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
test "$(git -C /opt/libero rev-parse HEAD)" = \
  8f1084e3132a39270c3a13ebe37270a43ece2a01
"$runtime_python" -m pip install --no-cache-dir --no-deps -e /opt/libero
