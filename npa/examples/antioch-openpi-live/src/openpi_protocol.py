"""Safe NumPy/msgpack wire codec used by the OpenPI websocket protocol.

Adapted from Physical Intelligence's Apache-2.0 ``openpi-client`` at
15a9616a00943ada6c20a0f158e3adb39df2ccac. Object, void, and complex arrays
are rejected instead of falling back to executable pickle serialization.
"""

from __future__ import annotations

import functools
import math

import msgpack
import numpy as np


def _pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"unsupported array dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _wire_dtype(value):
    description = value.get(b"dtype")
    if not isinstance(description, str):
        raise ValueError("array dtype must be a string")
    try:
        dtype = np.dtype(description)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid array dtype") from error
    if dtype.hasobject or dtype.kind in ("V", "O", "c") or dtype.itemsize == 0:
        raise ValueError(f"unsupported array dtype: {dtype}")
    return dtype


def _wire_array(value, dtype):
    data = value.get(b"data")
    shape = value.get(b"shape")
    if not isinstance(data, bytes):
        raise ValueError("array data must be bytes")
    if not isinstance(shape, (list, tuple)) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("array shape must contain nonnegative integers")
    if math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError("array shape does not match its byte length")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def _unpack_array(value):
    markers = {b"__ndarray__", b"__npgeneric__"}.intersection(value)
    if not markers:
        return value
    if len(markers) != 1 or value[next(iter(markers))] is not True:
        raise ValueError("invalid array marker")
    dtype = _wire_dtype(value)
    if b"__ndarray__" in markers:
        return _wire_array(value, dtype)
    if b"data" not in value:
        raise ValueError("scalar data is missing")
    return dtype.type(value[b"data"])


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
