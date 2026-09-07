"""Restricted Lance 4.0.0 profile written by the completed CLIP recipe.

Field numbers and framing follow the versioned upstream format specifications:
https://github.com/lance-format/lance/tree/v4.0.0/protos
https://github.com/lance-format/lance/blob/v4.0.0/rust/lance-table/src/io/manifest.rs
This independent wire reader rejects unsupported fields instead of implementing
general Lance portability. No Lance reader runs until this check succeeds.
"""

from __future__ import annotations

import re
import struct


def _require(condition):
    if not condition:
        raise ValueError("Unsupported or nonlocal CLIP Lance metadata")


def _varint(content, offset):
    value = 0
    for shift in range(0, 70, 7):
        _require(offset < len(content))
        byte = content[offset]
        offset += 1
        _require(shift != 63 or byte <= 1)
        value |= (byte & 127) << shift
        if byte < 128:
            _require(shift == 0 or byte != 0)
            return value, offset
    raise ValueError("Invalid Lance metadata integer")


def _message(content, types, repeated=()):
    """Read only the specified varint/length fields; reject duplicates/unknowns."""
    result, offset = {}, 0
    while offset < len(content):
        tag, offset = _varint(content, offset)
        field, wire = tag >> 3, tag & 7
        _require(field in types and wire == types[field])
        _require(field not in result or field in repeated)
        value, offset = _varint(content, offset)
        if wire == 2:
            end = offset + value
            _require(end <= len(content))
            value, offset = content[offset:end], end
        result.setdefault(field, []).append(value)
    return result


def _one(message, field, default=None):
    return message.get(field, [default])[0]


def _config(entries):
    result = {}
    for entry in entries:
        item = _message(entry, {1: 2, 2: 2})
        key, value = _one(item, 1), _one(item, 2)
        _require(key not in result and key is not None and value is not None)
        result[key] = value
    _require(result == {b"lance.auto_cleanup.interval": b"20",
                        b"lance.auto_cleanup.older_than": b"14days"})


def _schema(fields):
    expected = [(b"record_id", b"int64", 1), (b"input_sha256", b"string", 2),
                (b"processed_sha256", b"string", 2),
                (b"vector", b"fixed_size_list:float:512", 1)]
    _require(len(fields) == len(expected))
    for index, (content, (name, logical, encoding)) in enumerate(zip(fields, expected)):
        field = _message(content, {2: 2, 3: 0, 4: 0, 5: 2, 6: 0, 7: 0})
        _require(_one(field, 2) == name and _one(field, 3, 0) == index
                 and _one(field, 4) == 2**64 - 1 and _one(field, 5) == logical
                 and _one(field, 6) == 1 and _one(field, 7) == encoding)


def _fragments(entries, files, prefix, rows):
    paths, ids, count = set(), set(), 0
    _require(bool(entries))
    for index, entry in enumerate(entries):
        fragment = _message(entry, {1: 0, 2: 2, 4: 0}, repeated=(2,))
        identifier, physical_rows = _one(fragment, 1, 0), _one(fragment, 4, 0)
        _require(identifier == index and identifier < 2**32 and identifier not in ids
                 and physical_rows > 0 and len(fragment.get(2, [])) == 1)
        ids.add(identifier)
        count += physical_rows
        data = _message(fragment[2][0], {1: 2, 2: 2, 3: 2, 4: 0, 6: 0})
        name = _one(data, 1, b"")
        _require(re.fullmatch(rb"[01]{24}[0-9a-f]{26}\.lance", name) is not None)
        path = prefix + "data/" + name.decode("ascii")
        _require(path not in paths and path in files
                 and _one(data, 2) == bytes(range(4)) and _one(data, 3) == bytes(range(4))
                 and _one(data, 4) == 2 and _one(data, 6) == files[path]["size"])
        paths.add(path)
    _require(count == rows)
    return paths, max(ids)


def _initial_transaction_fragments(entries, committed):
    # Overwrite records the writer's provisional zero IDs. Commit assigns the
    # manifest's sequential IDs without updating those transaction records.
    _require(len(entries) == len(committed))
    for pending, assigned in zip(entries, committed):
        pending = _message(pending, {1: 0, 2: 2, 4: 0}, repeated=(2,))
        assigned = _message(assigned, {1: 0, 2: 2, 4: 0}, repeated=(2,))
        _require(pending.pop(1, [0]) == [0])
        assigned.pop(1, None)
        _require(pending == assigned)


def _read_manifest(root, manifest_name):
    content = (root / manifest_name).read_bytes()
    _require(len(content) >= 20)
    offset, major, minor, magic = struct.unpack("<QHH4s", content[-16:])
    _require((major, minor, magic) == (0, 2, b"LANC") and offset + 4 <= len(content) - 16)
    length = struct.unpack_from("<I", content, offset)[0]
    _require(offset + 4 + length == len(content) - 16)
    manifest = _message(content[offset + 4:-16], {
        1: 2, 2: 2, 3: 0, 7: 2, 10: 0, 11: 0, 12: 2, 13: 2, 15: 2, 16: 2, 21: 0,
    }, repeated=(1, 2, 16))
    _require(_one(manifest, 3) == 1 and _one(manifest, 10) == 8 and _one(manifest, 21) == 0)
    _require(_message(_one(manifest, 13, b""), {1: 2, 2: 2}) == {1: [b"lance"], 2: [b"4.0.0"]})
    _require(_message(_one(manifest, 15, b""), {1: 2, 2: 2}) == {1: [b"lance"], 2: [b"2.0"]})
    timestamp = _message(_one(manifest, 7, b""), {1: 0, 2: 0})
    _require(0 < _one(timestamp, 1, 0) < 2**63 and _one(timestamp, 2, 0) < 10**9)
    try:
        _schema(manifest.get(1, []))
    except ValueError as error:
        raise ValueError("Lance and Parquet schemas differ") from error
    _config(manifest.get(16, []))
    return content, offset, manifest


def _validate_data_versions(root, files, paths):
    for path in paths:
        _require(files[path]["size"] >= 8)
        with (root / path).open("rb") as data:
            data.seek(-8, 2)
            major, minor, magic = struct.unpack("<HH4s", data.read())
            # Lance 4.0.0 reader.rs maps both physical versions to logical V2_0.
            _require(magic == b"LANC" and (major, minor) in {(0, 3), (2, 0)})



def validate_local_lance(root, files, rows):
    """Require the recipe's single-version local table before invoking Lance.

    Args:
        root: Frozen private result tree.
        files: Complete regular-file inventory with exact byte sizes.
        rows: Number of independently validated Parquet records.
    Returns:
        None.
    Raises:
        ValueError: Metadata is unsupported or refers outside the completed table.
        OSError: A required local file cannot be read.
    """
    prefix = "lance/embeddings.lance/"
    names = {name for name in files if name.startswith("lance/")}
    manifests = {name for name in names if name.startswith(prefix + "_versions/")}
    _require(len(manifests) == 1)
    manifest_name = next(iter(manifests))
    _require(manifest_name in {prefix + "_versions/1.manifest",
                              prefix + "_versions/18446744073709551614.manifest"})
    content, offset, manifest = _read_manifest(root, manifest_name)
    paths, maximum = _fragments(manifest.get(2, []), files, prefix, rows)
    _require(_one(manifest, 11) == maximum)
    _validate_data_versions(root, files, paths)

    transaction_name = _one(manifest, 12, b"")
    _require(re.fullmatch(rb"0-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\.txn", transaction_name) is not None)
    transaction_path = prefix + "_transactions/" + transaction_name.decode("ascii")
    _require(names == {manifest_name, transaction_path, *paths})
    transaction = (root / transaction_path).read_bytes()
    _require(offset == 4 + len(transaction)
             and struct.unpack_from("<I", content)[0] == len(transaction)
             and content[4:offset] == transaction)
    operation = _message(transaction, {2: 2, 102: 2})
    _require(_one(operation, 2) == transaction_name[2:-4])
    overwrite = _message(_one(operation, 102, b""), {1: 2, 2: 2, 4: 2}, repeated=(1, 2, 4))
    _initial_transaction_fragments(overwrite.get(1, []), manifest.get(2, []))
    _require(overwrite.get(2) == manifest.get(1))
    _config(overwrite.get(4, []))
