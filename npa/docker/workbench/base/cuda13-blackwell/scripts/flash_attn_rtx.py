"""Opt-in, shape-qualified SM120 inference tuning for the pinned FA4 implementation."""

from importlib import metadata
import hashlib
import os
from pathlib import Path
from typing import NamedTuple

import torch

_COMMIT = "eed1971f5132630dc296fe37601e834d4b57a248"
_VERSION = "4.0.0b33.dev10+geed1971"


class _DenseArguments(NamedTuple):
    """Name the pinned kernel's positional ABI so omitted features remain explicit."""

    query: object
    key: object
    value: object
    output: object
    logsumexp: object = None
    softmax_scale: float = 1.0
    query_offsets: object = None
    key_offsets: object = None
    query_lengths: object = None
    key_lengths: object = None
    page_table: object = None
    window_left: object = None
    window_right: object = None
    learnable_sink: object = None
    block_sparse: object = None
    auxiliary_data: object = None
    query_tile_offsets: object = None
    split_tile_offsets: object = None


class _Runtime(NamedTuple):
    """Keep compiled kernels and imports local to one explicitly selected device."""

    device: object
    native: object
    tiled: object
    cache: object
    auxiliary_data: object


def _pinned_version():
    version, separator, source = metadata.version("flash-attn-4").partition("+g")
    # Git lengthens its abbreviation as the repository grows; the pin is unchanged.
    return (
        version == _VERSION.partition("+g")[0]
        and bool(separator)
        and len(source) >= 7
        and _COMMIT.startswith(source)
    )


def _device(device):
    if os.getenv("NPA_FLASH_ATTN_COMMIT") != _COMMIT or not _pinned_version():
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


def _runtime(selected):
    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.cache_utils import get_jit_cache
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.utils import AuxData

    # Upstream fingerprints cute/, but this launcher lives outside that directory.
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache = get_jit_cache(f"rtx6000_inference_{source_hash}")
    return _Runtime(selected, flash_attn_func, _flash_attn_fwd, cache, AuxData())


def _arguments(query, key, value, output, auxiliary_data):
    return _DenseArguments(
        query,
        key,
        value,
        output,
        softmax_scale=query.shape[-1] ** -0.5,
        auxiliary_data=auxiliary_data,
    )


def _compile_kernel(query, key, value, output, configuration, causal, auxiliary_data):
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

    dimension = query.shape[-1]
    tile, query_in_registers = configuration
    dtype = cutlass.Float16 if query.dtype == torch.float16 else cutlass.BFloat16
    kernel = FlashAttentionForwardSm120(
        dtype,
        dimension,
        dimension,
        query.shape[2] // key.shape[2],
        is_causal=causal,
        is_local=False,
        pack_gqa=False,
        tile_m=tile[0],
        tile_n=tile[1],
        num_stages=1,
        num_threads=128,
        Q_in_regs=query_in_registers,
    )
    converted = [to_cute_tensor(t) for t in (query, key, value, output)]
    arguments = _arguments(query, key, value, output, auxiliary_data)
    arguments = arguments._replace(
        query=converted[0], key=converted[1], value=converted[2], output=converted[3]
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, *arguments, stream, options="--enable-tvm-ffi")


def _direct_layout(query, key, value):
    return all(
        t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (query, key, value)
    )


def _direct_configuration(query, key, tile):
    if query.dtype == torch.float16 and query.shape[-1] == 64:
        if key.shape[1] == 77:
            return (64, 64), False
        if query.shape[1] in (1024, 1536, 6144) and key.shape[1] == query.shape[1]:
            return (128, 64), True
    return tile, False


def _run_attention(runtime, query, key, value, causal):
    tile = _tile(query, key, value, causal)
    options = {"causal": causal, "pack_gqa": False, "num_splits": 1}
    if tile is None:
        result = runtime.native(query, key, value, **options)
        return result[0] if isinstance(result, tuple) else result
    if not _direct_layout(query, key, value):
        return runtime.tiled(query, key, value, tile_mn=tile, **options)[0]
    configuration = _direct_configuration(query, key, tile)
    cache_key = (
        "sm120",
        str(query.dtype),
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
        configuration,
        causal,
    )
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    if cache_key not in runtime.cache:
        runtime.cache[cache_key] = _compile_kernel(
            query, key, value, output, configuration, causal, runtime.auxiliary_data
        )
    runtime.cache[cache_key](
        *_arguments(query, key, value, output, runtime.auxiliary_data)
    )
    return output


def make_inference_attention(device=None):
    """Bind opt-in FA4 inference tuning to one SM120 device.

    Args:
        device: CUDA device for this callable; None selects the current device.
    Returns:
        A tensor-returning Q/K/V callable with keyword-only ``causal=False``.
        Qualified contiguous shapes directly launch cached upstream kernels;
        other layouts retain tiled dispatch, and other shapes use native FA4.
    Raises:
        RuntimeError: The installed FA4 revision differs from the qualified pin.
        ValueError: The device is unsupported, gradients are enabled, or tensors
            require gradients or belong to another device.
        TypeError: The caller supplies unsupported options or a nonboolean causal flag.
        Exception: Native FA4 errors propagate without a backend substitution.
    """
    runtime = _runtime(_device(device))

    def forward(query, key, value, *, causal=False):
        _check_call(query, key, value, causal, runtime.device)
        if torch.cuda.current_device() == runtime.device.index:
            return _run_attention(runtime, query, key, value, causal)
        with torch.cuda.device(runtime.device):
            return _run_attention(runtime, query, key, value, causal)

    return forward
