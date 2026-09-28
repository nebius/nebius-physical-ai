"""Transfer hash-bound native capture and navigation bundles without archive escapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import tarfile

from npa.workflows.field_failure.artifacts import (
    _invalid_constant,
    _location,
    _storage,
    _unique_keys,
)


def _download(artifact, destination):
    bucket, key = _location(artifact["uri"])
    response = _storage().s3.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    with response["Body"] as source, destination.open("xb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if not destination.stat().st_size or digest.hexdigest() != artifact["sha256"]:
        raise ValueError("native input artifact digest differs from sealed request")


def _extract(archive, destination):
    destination.mkdir()
    with tarfile.open(archive, "r:*") as stream:
        names = set()
        for member in stream:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != member.name
                or member.name in names
                or "\\" in member.name
            ):
                raise ValueError(
                    "native bundle requires unique contained regular files"
                )
            names.add(member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    if not names:
        raise ValueError("native bundle is empty")


def _bundle(artifact, root, name):
    archive = root / (name + ".tar")
    _download(artifact, archive)
    output = root / name
    _extract(archive, output)
    return output


def _upload(path, uri):
    data = path.read_bytes()
    if not data:
        raise ValueError("native output artifact is empty")
    _storage().put_bytes_conditional(data, uri, if_none_match=True)
    return {"uri": uri, "sha256": hashlib.sha256(data).hexdigest()}


def _upload_json(value, path, uri):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return _upload(path, uri)


def _archive(root, path):
    with tarfile.open(path, "w") as stream:
        for member in sorted(root.rglob("*")):
            if member.is_symlink():
                raise ValueError("native output bundle contains a symbolic link")
            if member.is_file():
                stream.add(member, arcname=member.relative_to(root), recursive=False)


def _protocol(request, root):
    path = root / "protocol.json"
    _download(request["protocol"], path)
    value = _read_json(path)
    if set(value) != {
        "schema_version",
        "navigation_image",
        "task",
        "adapter_module",
        "adapter_sha256",
        "source_bundle_sha256",
    }:
        raise ValueError("native protocol has unsupported or missing fields")
    if value["schema_version"] != "npa.field-failure.native-protocol.v1":
        raise ValueError("unsupported native field-failure protocol")
    return value


def _identity(request, schema):
    return {
        "schema_version": schema,
        **{
            key: request[key]
            for key in ("bundle_sha256", "adapter", "attempt_id", "inputs_sha256")
        },
    }


def _recipe(root, protocol):
    path = root / "recipe.json"
    value = _read_json(path)
    expected = {
        key: protocol[key]
        for key in ("task", "adapter_module", "adapter_sha256", "source_bundle_sha256")
    }
    expected["image"] = protocol["navigation_image"]
    if any(value.get(key) != selected for key, selected in expected.items()):
        raise ValueError("native recipe differs from the sealed common protocol")
    if value.get("initial_checkpoint") is not None:
        raise ValueError("native scene bundles cannot choose the evaluated checkpoint")
    return value


def _read_json(path):
    return json.loads(
        path.read_text(),
        object_pairs_hook=_unique_keys,
        parse_constant=_invalid_constant,
    )
