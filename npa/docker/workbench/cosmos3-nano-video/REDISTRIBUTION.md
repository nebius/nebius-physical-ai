# Operator-private runtime

The pinned upstream `vllm/vllm-omni:cosmos3` image includes a vendor runtime.
This derived image is restricted to the owning operator's registry and is
excluded from NPA's public publishing plan. The added adapter source is NPA
source; its license does not change the redistribution terms of inherited
image layers.

The model is delivered directly to the operator at runtime from
[NVIDIA's pinned Cosmos3-Nano release](https://huggingface.co/nvidia/Cosmos3-Nano/tree/7a312c868bcce8e40b3eb40861300a9d0ba3fde1),
under its [OpenMDW 1.1 terms](https://openmdw.ai/license/1.1).
No model weights, example media, credentials or populated caches are included
in the image. The shared cache is runtime storage and must not enter a build
context or be published as a derived image.
