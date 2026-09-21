# SeedVR2: actual B200 native kernel proof

Five numerical tests passed in the final private image on one NVIDIA B200. Each CUDA result was finite and compared with a float32 reference at the prospectively fixed **rtol=0.02 and atol=0.02**. The checks exercise native kernels, beyond imports or CUDA availability.

Source: `0b5e47d97414a2fe46d6184efe9b4423b2b52198`. Runtime image manifest: `sha256:e48b35346e47d6d70f9336fda238c38c1c5cfc8c64448dca89840eb201ab0362`.

| Actual GPU check | Maximum absolute error | Mean absolute error | Result |
| --- | ---: | ---: | --- |
| torch_matmul_bf16 | 0.0624942780 | 0.0087667182 | Pass |
| cudnn_conv2d_bf16 | 0.0623493195 | 0.0091950111 | Pass |
| flash_attn_varlen_bf16 | 0.0038740635 | 0.0003734776 | Pass |
| apex_fused_layer_norm_bf16 | 0.0076391697 | 0.0011761039 | Pass |
| apex_fused_rms_norm_bf16 | 0.0077462196 | 0.0012162294 | Pass |

The tolerance combines absolute and relative error. Passing does not mean maximum absolute error is below 0.02.

Observed runtime: Torch 2.13.0+cu130, CUDA 13.0, cuDNN 92000, Flash Attention 2.8.3.post1; GPU compute capability 10.0, MIG disabled. The actual pod ran as UID 1000. Peak Torch-allocated memory during this small probe was 34,476,544 bytes; this is not model memory or throughput.

The private registry manifest/config/layers, target pod image digest, baked source revision, Job/Pod ownership and successful exit were checked. Owned test resources were removed with UID preconditions; existing resources were preserved. Anonymous registry access was denied.

[Actual probe code](kernel-probe.py) · [Measurements and original receipt hashes](measurements.json) · [File checksums](SHA256SUMS)

The raw cluster logs and device UUID remain private. Public files contain allowlisted measurements and hashes.

**Remaining acceptance:** the complete-byte image scan and real model restoration/quality evaluation. This probe used no model weights or quality inputs. Prior quality failures remain failures. It establishes native B200 kernel compatibility only; it does not qualify restored video, other GPUs, physical AI performance, or public image redistribution.
