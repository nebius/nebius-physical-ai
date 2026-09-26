# Third-party inventory: npa-flex-pi

| Boundary | Artifact | Shipped | Terms |
| --- | --- | --- | --- |
| Source | `geyan21/flex-pi@20c1b2b…` | yes | MIT; license retained in `/opt/flex-pi/LICENSE` |
| Runtime | PyTorch 2.7.1, CUDA 12.8, cuDNN 9, FFmpeg 7, Python closure | yes | upstream container/runtime and package-specific OSS terms |
| Checkpoint | `flex-pi/flexpi-robotwin@87d3833…` | no | MIT-labelled repository; runtime fetch |
| Public observation | `flex-pi/robotwin_3d@bee164a…`, episode 0 frame 0 | no | dataset card declares no license; authorized runtime fetch and hash verification only |
| Wan VAE/T5 assets | `DiffSynth-Studio/Wan-Series-Converted-Safetensors@150f75d…` | no | Apache-2.0 repository; runtime fetch with file-hash checks |
| Wan tokenizer | `Wan-AI/Wan2.1-T2V-1.3B@37ec512…` | no | Apache-2.0 repository; runtime fetch |
| DINOv3 encoder | `timm/vit_base_patch16_dinov3.lvd1689m@c6a5fb7…` | no | DINOv3 model license; runtime fetch with file-hash check |
| Credentials, caches, outputs | operator-owned | no | never build inputs or image layers |
