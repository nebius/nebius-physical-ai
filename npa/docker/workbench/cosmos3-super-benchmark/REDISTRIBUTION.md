# Redistribution boundary

This image derives from the pinned `vllm/vllm-omni:cosmos3` runtime and is for
operator-controlled registry use only. Do not publish it to the NPA public GHCR
namespace or otherwise redistribute it outside the owning organization without
separately establishing redistribution rights for every inherited layer.

The build adds only the SkyPilot 0.12.2 bootstrap packages (`openssh-server`,
`rsync`, and `sudo`). Models remain runtime-fetched with the operator's own
Hugging Face credential and explicit NVIDIA Software License acceptance.
