# Redistribution boundary

This image derives from the pinned `vllm/vllm-omni:cosmos3` runtime and is for
operator-controlled registry use only. Do not publish it to the NPA public GHCR
namespace or otherwise redistribute it outside the owning organization without
separately establishing redistribution rights for every inherited layer.

The build adds only the SkyPilot 0.12.2 bootstrap packages (`openssh-server`,
`rsync`, and `sudo`). Models remain runtime-fetched with the operator's own
Hugging Face credential where authentication is required. The upstream runtime
terms remain applicable; the benchmark does not add an NPA acceptance flag.
NVIDIA describes acceptance through registration or use in its
[Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/).
