"""Opt-in, shape-qualified SM120 inference tuning for the pinned FA4 implementation."""

from importlib import metadata
import os

import torch

_COMMIT = "eed1971f5132630dc296fe37601e834d4b57a248"
_VERSION = "4.0.0b33.dev10+geed1971"


def _device(device):
    if (
        os.getenv("NPA_FLASH_ATTN_COMMIT") != _COMMIT
        or metadata.version("flash-attn-4") != _VERSION
    ):
        raise RuntimeError("RTX inference tuning requires the exact pinned FA4 source")
    selected = torch.device("cuda" if device is None else device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("RTX inference tuning requires a CUDA device")
    selected = torch.device(
        "cuda",
        torch.cuda.current_device() if selected.index is None else selected.index,
    )
    if torch.cuda.get_device_capability(selected) != (12, 0):
        raise ValueError("RTX inference tuning requires SM120")
    return selected


def _tile(query, key, value, causal):
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        return None
    batch, sequence, heads, dimension = query.shape
    if batch != 2 or key.shape[0] != batch or key.shape[3] != dimension:
        return None
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return None
    if dimension == 64 and query.dtype == torch.float16 and not causal:
        expected_heads = {1024: 20, 1536: 20, 4096: 10, 6144: 10}.get(sequence)
        if heads != expected_heads or key.shape[2] != heads:
            return None
        if key.shape[1] == 77:
            return (128, 128)
        if key.shape[1] == sequence:
            return {
                1024: (128, 64),
                1536: (128, 64),
                4096: (64, 128),
                6144: (128, 128),
            }[sequence]
    if (
        dimension == 128
        and query.dtype == torch.bfloat16
        and causal
        and heads == 16
        and key.shape[2] in (4, 16)
        and key.shape[1] == sequence
    ):
        return {1024: (64, 128), 4096: (64, 64)}.get(sequence)
    return None


def _check_call(query, key, value, causal, device):
    if not isinstance(causal, bool):
        raise TypeError("causal must be a boolean")
    if torch.is_grad_enabled() or any(t.requires_grad for t in (query, key, value)):
        raise ValueError("RTX tuning is inference-only; disable gradients explicitly")
    if any(t.device != device for t in (query, key, value)):
        raise ValueError("All tensors must use the device bound by the factory")


def make_inference_attention(device=None):
    """Bind opt-in FA4 inference tuning to one SM120 device.

    Args:
        device: CUDA device for this callable; None selects the current device.
    Returns:
        A tensor-returning Q/K/V callable with keyword-only ``causal=False``.
        Qualified shapes use measured tiles; other shapes use native FA4.
    Raises:
        RuntimeError: The installed FA4 revision differs from the qualified pin.
        ValueError: The device is unsupported, gradients are enabled, or tensors
            require gradients or belong to another device.
        TypeError: The caller supplies unsupported options or a nonboolean causal flag.
        Exception: Native FA4 errors propagate without a backend substitution.
    """
    selected = _device(device)
    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.interface import _flash_attn_fwd

    def forward(query, key, value, *, causal=False):
        _check_call(query, key, value, causal, selected)
        tile = _tile(query, key, value, causal)
        options = {"causal": causal, "pack_gqa": False, "num_splits": 1}
        if tile is not None:
            return _flash_attn_fwd(query, key, value, tile_mn=tile, **options)[0]
        result = flash_attn_func(query, key, value, **options)
        return result[0] if isinstance(result, tuple) else result

    return forward
