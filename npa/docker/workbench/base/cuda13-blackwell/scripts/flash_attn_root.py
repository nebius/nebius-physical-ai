"""Tensor-only FA4 root imports for the CUDA 13 Workbench base.

This is a narrow adapter, not an implementation of the FA2 API. Arguments after
Q/K/V must be named: FA2 and FA4 assign different meanings to positional inputs.
Use flash_attn.cute directly for separately qualified advanced configurations.
"""

from importlib import metadata

from flash_attn.cute import flash_attn_func as _dense
from flash_attn.cute import flash_attn_varlen_func as _varlen

__version__ = metadata.version("flash-attn-4")
__all__ = ["flash_attn_func", "flash_attn_varlen_func"]


def _options(options):
    allowed = {"dropout_p", "softmax_scale", "causal", "pack_gqa", "num_splits"}
    unknown = options.keys() - allowed
    if unknown:
        raise TypeError(
            f"Unsupported root FA4 options: {', '.join(sorted(unknown))}. "
            "Use flash_attn.cute for separately qualified features."
        )
    if options.pop("dropout_p", 0.0) != 0.0:
        raise ValueError("The root FA4 adapter requires dropout_p=0.0")
    if options.get("pack_gqa", False) is not False:
        raise ValueError("The root FA4 adapter requires pack_gqa=False")
    if options.get("num_splits", 1) != 1:
        raise ValueError("The root FA4 adapter requires num_splits=1")
    return {**options, "pack_gqa": False, "num_splits": 1}


def _tensor_result(result):
    return result[0] if isinstance(result, tuple) else result


def flash_attn_func(q, k, v, **kwargs):
    """Execute dense FA4 with explicitly named options and no backend fallback.

    Args:
        q: Query tensor in batch, sequence, head, dimension order.
        k: Key tensor in the same layout.
        v: Value tensor in the same layout.
        **kwargs: softmax_scale, causal, and optional dropout_p=0,
            pack_gqa=False, num_splits=1. Other features are rejected.
    Returns:
        The attention output tensor, preserving its autograd graph.
    Raises:
        TypeError: Positional options or unknown keyword options were supplied.
        ValueError: Dropout, packed GQA or split-KV was requested.
        Exception: The native FA4 implementation failed; errors propagate.
    """
    return _tensor_result(_dense(q, k, v, **_options(kwargs)))


def flash_attn_varlen_func(
    q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
):
    """Execute variable-length FA4 with named sequence metadata.

    Args:
        q: Packed query tensor in token, head, dimension order.
        k: Packed key tensor in the same layout.
        v: Packed value tensor in the same layout.
        cu_seqlens_q: Cumulative query lengths, including the initial zero.
        cu_seqlens_k: Cumulative key lengths, including the initial zero.
        max_seqlen_q: Maximum query length.
        max_seqlen_k: Maximum key length.
        **kwargs: The same restricted options as flash_attn_func.
    Returns:
        The packed attention output tensor, preserving its autograd graph.
    Raises:
        TypeError: Sequence metadata is missing, positional or unsupported.
        ValueError: Dropout, packed GQA or split-KV was requested.
        Exception: The native FA4 implementation failed; errors propagate.
    """
    return _tensor_result(
        _varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            **_options(kwargs),
        )
    )
