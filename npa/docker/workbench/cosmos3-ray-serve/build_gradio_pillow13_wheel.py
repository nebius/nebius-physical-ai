"""Build the reviewed Gradio 6.16.0 Pillow 13 metadata derivative wheel."""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


SOURCE_VERSION = "6.16.0"
DERIVATIVE_VERSION = "6.16.0+npa.pillow13.1"
SOURCE_WHEEL_SHA256 = "a5de01bf17336bf421d52af4b5dbd9fdbd530762c485067748456ccb797700bb"
DERIVATIVE_WHEEL_SHA256 = (
    "2523b8b34006626382b2afa315c229c3ef4515b17c6925b766308f32e161c749"
)
ORIGINAL_METADATA_SHA256 = (
    "3c525b5b35fdebe7f1f1efbac3d33c466c2eb9e56254e638343c072f3488a7ae"
)
ORIGINAL_REQUIREMENT = b"Requires-Dist: pillow<13.0,>=8.0\n"
PATCHED_REQUIREMENT = b"Requires-Dist: pillow<14.0,>=13.0.0\n"
ORIGINAL_VERSION = b"Version: 6.16.0\n"
PATCHED_VERSION = b"Version: 6.16.0+npa.pillow13.1\n"
SOURCE_DIST_INFO = "gradio-6.16.0.dist-info"
DERIVATIVE_DIST_INFO = "gradio-6.16.0+npa.pillow13.1.dist-info"
DERIVATIVE_FILENAME = "gradio-6.16.0+npa.pillow13.1-py3-none-any.whl"
SOURCE_RECORD = f"{SOURCE_DIST_INFO}/RECORD"
DERIVATIVE_RECORD = f"{DERIVATIVE_DIST_INFO}/RECORD"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise RuntimeError("source wheel contains duplicate paths")
    for member in members:
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (
            not member.filename
            or "\\" in member.filename
            or path.is_absolute()
            or "." in path.parts
            or ".." in path.parts
            or stat.S_ISLNK(mode)
        ):
            raise RuntimeError("source wheel contains an unsafe path")
    if names.count(SOURCE_RECORD) != 1:
        raise RuntimeError("source wheel does not contain exactly one reviewed RECORD")
    return members


def _decode_record_digest(value: str) -> bytes:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise RuntimeError("source wheel RECORD uses an unsupported digest")
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("source wheel RECORD contains an invalid digest") from error


def _validate_source_record(
    archive: zipfile.ZipFile, members: list[zipfile.ZipInfo]
) -> None:
    rows = list(csv.reader(io.StringIO(archive.read(SOURCE_RECORD).decode("utf-8"))))
    if not rows or any(len(row) != 3 for row in rows):
        raise RuntimeError("source wheel RECORD is malformed")
    paths = [row[0] for row in rows]
    member_paths = [member.filename for member in members]
    if len(paths) != len(set(paths)) or set(paths) != set(member_paths):
        raise RuntimeError("source wheel RECORD does not cover the exact archive")
    for path, digest, size in rows:
        if path == SOURCE_RECORD:
            if digest or size:
                raise RuntimeError("source wheel RECORD must leave itself unhashed")
            continue
        payload = archive.read(path)
        if _decode_record_digest(digest) != hashlib.sha256(payload).digest():
            raise RuntimeError("source wheel RECORD contains a digest mismatch")
        if not size.isdecimal() or int(size) != len(payload):
            raise RuntimeError("source wheel RECORD contains a size mismatch")


def _patched_metadata(original: bytes) -> bytes:
    if _sha256(original) != ORIGINAL_METADATA_SHA256:
        raise RuntimeError("Gradio METADATA does not match the reviewed source wheel")
    replacements = (
        (ORIGINAL_VERSION, PATCHED_VERSION),
        (ORIGINAL_REQUIREMENT, PATCHED_REQUIREMENT),
    )
    patched = original
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError("reviewed Gradio metadata field is not unique")
        patched = patched.replace(old, new)
    return patched


def _derivative_path(source: str) -> str:
    prefix = f"{SOURCE_DIST_INFO}/"
    if source.startswith(prefix):
        return f"{DERIVATIVE_DIST_INFO}/{source.removeprefix(prefix)}"
    return source


def _record_row(path: str, payload: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return [path, f"sha256={digest.decode()}", str(len(payload))]


def _record(rows: list[list[str]]) -> bytes:
    rows.append([f"{DERIVATIVE_DIST_INFO}/RECORD", "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


def build_derivative(source_path: Path, destination: Path) -> dict[str, str]:
    """Build the deterministic local-version wheel from reviewed source bytes."""
    source_mode = source_path.lstat().st_mode
    if source_path.is_symlink() or not stat.S_ISREG(source_mode):
        raise RuntimeError("Gradio source wheel must be a regular file")
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != SOURCE_WHEEL_SHA256:
        raise RuntimeError("Gradio source wheel SHA-256 does not match")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("refusing to overwrite a derivative wheel")

    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with (
            zipfile.ZipFile(source_path) as source,
            zipfile.ZipFile(temporary, "w") as target,
        ):
            target.comment = source.comment
            members = _safe_members(source)
            _validate_source_record(source, members)
            rows = []
            derivative_paths: set[str] = set()
            for member in members:
                if member.filename == SOURCE_RECORD:
                    continue
                payload = source.read(member)
                if member.filename == f"{SOURCE_DIST_INFO}/METADATA":
                    payload = _patched_metadata(payload)
                path = _derivative_path(member.filename)
                if path in derivative_paths:
                    raise RuntimeError("derivative wheel would contain duplicate paths")
                derivative_paths.add(path)
                info = copy.copy(member)
                info.filename = path
                target.writestr(info, payload)
                rows.append(_record_row(path, payload))
            record_info = zipfile.ZipInfo(DERIVATIVE_RECORD)
            record_info.compress_type = zipfile.ZIP_DEFLATED
            record_info.external_attr = 0o100644 << 16
            target.writestr(record_info, _record(rows))
        derivative_sha256 = _sha256(temporary.read_bytes())
        if derivative_sha256 != DERIVATIVE_WHEEL_SHA256:
            raise RuntimeError("Gradio derivative wheel SHA-256 is not reproducible")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": "npa.gradio-pillow-wheel-derivative.v1",
        "package": f"gradio=={DERIVATIVE_VERSION}",
        "license": "Apache-2.0",
        "source_wheel_sha256": SOURCE_WHEEL_SHA256,
        "original_metadata_sha256": ORIGINAL_METADATA_SHA256,
        "derivative_wheel_sha256": derivative_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination_directory", type=Path)
    arguments = parser.parse_args()
    destination = arguments.destination_directory / DERIVATIVE_FILENAME
    print(json.dumps(build_derivative(arguments.source, destination), sort_keys=True))


if __name__ == "__main__":
    main()
