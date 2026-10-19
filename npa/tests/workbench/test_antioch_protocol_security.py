"""Exercise the real OpenPI wire codec against valid and hostile peer packets."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import msgpack
import numpy as np
import pytest


@pytest.fixture
def codec():
    path = (
        Path(__file__).resolve().parents[3]
        / "npa/examples/antioch-openpi-live/src/openpi_protocol.py"
    )
    spec = importlib.util.spec_from_file_location("openpi_security_codec", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array_packet(**changes):
    packet = {
        b"__ndarray__": True,
        b"dtype": "<f4",
        b"shape": [2],
        b"data": np.asarray([1.5, 2.5], dtype="<f4").tobytes(),
    }
    packet.update({key.encode(): value for key, value in changes.items()})
    return packet


def _forbid_native_array(*_args, **_kwargs):
    pytest.fail("malformed wire data reached native array construction")


@pytest.mark.parametrize("dtype", ["O", "|O8", "V8", "c16", "U0", "S0"])
@pytest.mark.parametrize("marker", [b"__ndarray__", b"__npgeneric__"])
def test_unsafe_wire_dtype_rejected_before_native_access(
    codec, monkeypatch, dtype, marker
):
    packet = {marker: True, b"dtype": dtype, b"shape": [1], b"data": b"A" * 8}
    wire = msgpack.packb(packet)
    monkeypatch.setattr(codec.np, "ndarray", _forbid_native_array)
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="unsupported array dtype"):
        codec.unpackb(wire)


@pytest.mark.parametrize("description", [None, False, 7, b"f4", [["field", "O"]]])
def test_dtype_requires_primitive_string(codec, monkeypatch, description):
    wire = msgpack.packb(_array_packet(dtype=description))
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="dtype must be a string"):
        codec.unpackb(wire)


def test_invalid_primitive_dtype_rejected(codec, monkeypatch):
    wire = msgpack.packb(_array_packet(dtype="not-a-numpy-type"))
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="invalid array dtype"):
        codec.unpackb(wire)


@pytest.mark.parametrize("shape", [None, 2, "2", [-1], [True], [1.0, 2], [[], 2]])
def test_invalid_shape_rejected_before_construction(codec, monkeypatch, shape):
    wire = msgpack.packb(_array_packet(shape=shape))
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="shape must contain nonnegative integers"):
        codec.unpackb(wire)


@pytest.mark.parametrize("shape", [[], [1], [3], [0, 2], [2**63 - 1]])
def test_byte_length_must_match_exactly(codec, monkeypatch, shape):
    wire = msgpack.packb(_array_packet(shape=shape))
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="shape does not match its byte length"):
        codec.unpackb(wire)


@pytest.mark.parametrize("data", [None, "12345678", [1, 2], {"value": 1}])
def test_array_payload_must_be_bytes(codec, monkeypatch, data):
    wire = msgpack.packb(_array_packet(data=data))
    monkeypatch.setattr(codec.np, "frombuffer", _forbid_native_array)
    with pytest.raises(ValueError, match="array data must be bytes"):
        codec.unpackb(wire)


@pytest.mark.parametrize("marker", [None, False, 1, "true"])
def test_array_marker_is_literal_true(codec, marker):
    packet = _array_packet()
    packet[b"__ndarray__"] = marker
    with pytest.raises(ValueError, match="invalid array marker"):
        codec.unpackb(msgpack.packb(packet))


def test_ambiguous_array_marker_rejected(codec):
    packet = _array_packet()
    packet[b"__npgeneric__"] = True
    with pytest.raises(ValueError, match="invalid array marker"):
        codec.unpackb(msgpack.packb(packet))


def test_missing_scalar_value_rejected(codec):
    with pytest.raises(ValueError, match="scalar data is missing"):
        codec.unpackb(msgpack.packb({b"__npgeneric__": True, b"dtype": "f4"}))


@pytest.mark.parametrize("dtype", ["<f4", ">f8", "<i2", "u8", "?", "S4", "U4"])
@pytest.mark.parametrize("shape", [(), (0, 3), (3, 4)])
def test_primitive_array_round_trip_preserves_bytes_shape_and_dtype(
    codec, dtype, shape
):
    source = np.zeros(shape, dtype=dtype)
    decoded = codec.unpackb(codec.Packer().pack({"array": source}))["array"]
    assert decoded.shape == source.shape
    assert decoded.dtype == source.dtype
    assert decoded.tobytes() == source.tobytes()


@pytest.mark.parametrize(
    "source", [np.float32(1.5), np.int16(-7), np.uint64(2**63 + 1)]
)
def test_numeric_scalar_round_trip(codec, source):
    decoded = codec.unpackb(codec.Packer().pack({"value": source}))["value"]
    assert type(decoded) is type(source)
    assert decoded == source


def test_real_observation_and_action_packets_round_trip(codec):
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    observation = {
        "observation/exterior_image_1_left": image[:, ::-1],
        "observation/wrist_image_left": image.swapaxes(0, 1),
        "observation/joint_position": np.arange(8, dtype=np.float32),
        "prompt": "pick up the red cube",
    }
    actions = np.arange(120, dtype=np.float32).reshape(15, 8) / 120
    for packet in (observation, {"actions": actions}):
        decoded = codec.unpackb(codec.Packer().pack(packet))
        for key, value in packet.items():
            np.testing.assert_array_equal(decoded[key], value)
    assert codec.unpackb(codec.Packer().pack({"server_version": "test"})) == {
        "server_version": "test"
    }
