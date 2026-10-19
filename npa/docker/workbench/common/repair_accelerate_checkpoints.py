"""Repair reviewed Accelerate checkpoint indexes without changing package versions.

This NPA repair addresses GHSA-4j2p-28q2-5m79, including the unchanged unsafe
loader in 1.15.0. It is not an upstream release or an advisory suppression.
The original Apache-2.0 source and license remain installed. Only exact reviewed
source bytes are accepted; RECORD and checked-hash bytecode describe the repair.
"""

from __future__ import annotations

import base64
import csv
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import py_compile

SOURCE_PATH = "accelerate/utils/modeling.py"
SOURCE_HASHES = {
    "1.12.0": "421a88fba25d6a50791fb54ea52e7eefd3beb678d1a131055176c09a87e820d6",
    "1.13.0": "d6052abc0625740115176a4d2dfc0861a189a2dba94443c09cc8500254b57e2e",
    "1.14.0": "3f603a594d95afcf76df81bb487097095b3282b6aaab50728e691a2a6b17db80",
    "1.15.0": "e341cb326d09833424816b484031c71113afdac6d95418531ae30256066a8478",
}
REPAIRED_HASHES = {
    "1.12.0": "3056415b6c16df94225ba17e574cda657e6be6a44d30caf7dc3058048d15fa88",
    "1.13.0": "ef25cecdd324e22812e859251920992a3b493ee26be8917dc6db8a15e24007db",
    "1.14.0": "8c37bfa2c8c3a7932e2fc851524912fdd2398c79d8a6c7a43518c137564bc1a5",
    "1.15.0": "2dbd1c995788088a37f2d88b1fb6f131d148c0897e92df67660271a99b447efb",
}

ORIGINAL_INDEX = """        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename) as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
"""

INDEX_GUARD = '''def _npa_checkpoint_files(index_filename):
    """Validate index-selected paths before any checkpoint shard is loaded."""
    import stat

    # Nonblocking open and descriptor inspection reject an index FIFO without
    # waiting for a writer. Symlinks to regular HF cache blobs remain usable.
    descriptor = os.open(index_filename, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("checkpoint index must be a regular file")
        index = json.load(stream)
    if isinstance(index, dict) and "weight_map" in index:
        index = index["weight_map"]
    if not isinstance(index, dict) or not index:
        raise ValueError("checkpoint weight_map must be a nonempty mapping")
    folder = os.path.abspath(os.path.dirname(os.fspath(index_filename)))
    files = set()
    for name in index.values():
        if not isinstance(name, str) or not name or os.path.isabs(name):
            raise ValueError("checkpoint shard must be a relative file path")
        filename = os.path.abspath(os.path.join(folder, name))
        if os.path.commonpath([folder, filename]) != folder:
            raise ValueError("checkpoint shard escapes the checkpoint directory")
        # Lexical containment preserves HF snapshot -> blob symlinks. The
        # operator must own the local cache; this is not a filesystem sandbox.
        if not stat.S_ISREG(os.stat(filename).st_mode):
            raise ValueError("checkpoint shard must be a regular file")
        files.add(filename)
    return sorted(files)


'''


def _transform(source: bytes) -> bytes:
    text = source.decode("utf-8")
    anchor = "def load_checkpoint_in_model("
    if text.count(anchor) != 1 or text.count(ORIGINAL_INDEX) != 1:
        raise RuntimeError("Accelerate checkpoint repair anchors differ")
    text = text.replace(anchor, INDEX_GUARD + anchor, 1)
    return text.replace(
        ORIGINAL_INDEX,
        "        checkpoint_files = _npa_checkpoint_files(index_filename)\n",
        1,
    ).encode("utf-8")


def patched_source(version: str, source: bytes) -> bytes:
    """Return the exact reviewed checkpoint loader repair, idempotently.

    Args:
        version: Installed Accelerate version.
        source: Actual installed modeling module bytes.
    Returns:
        Reviewed repaired source bytes.
    Raises:
        RuntimeError: Version, original bytes, or resulting repair is unreviewed.
    """
    if version not in SOURCE_HASHES:
        raise RuntimeError("checkpoint repair requires reviewed Accelerate 1.12-1.15")
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_HASHES[version]:
        return source
    if digest != SOURCE_HASHES[version]:
        raise RuntimeError("Accelerate source differs from reviewed upstream bytes")
    result = _transform(source)
    if hashlib.sha256(result).hexdigest() != REPAIRED_HASHES[version]:
        raise RuntimeError("Accelerate checkpoint repair differs from reviewed result")
    return result


def _record_row(root: Path, path: Path) -> list[str]:
    content = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
    return [
        path.relative_to(root).as_posix(),
        "sha256=" + digest.rstrip(b"=").decode(),
        str(len(content)),
    ]


def _write_repair(root: Path, source: bytes, receipt: Path, version: str) -> list[Path]:
    path = root / SOURCE_PATH
    path.write_bytes(source)
    for cached in (path.parent / "__pycache__").glob("modeling.*.pyc"):
        cached.unlink()
    compiled = py_compile.compile(
        str(path),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
    )
    receipt.write_text(
        json.dumps(
            {
                "advisory": "GHSA-4j2p-28q2-5m79",
                "repair": "npa-checkpoint-index-v1",
                "version": version,
                "original_sha256": SOURCE_HASHES[version],
                "repaired_sha256": REPAIRED_HASHES[version],
                "scope": "index-selected relative paths and regular index/shard files",
            },
            indent=2,
        )
        + "\n"
    )
    return [path, Path(compiled), receipt]


def main() -> None:
    """Repair the active interpreter's exact Accelerate distribution.

    Args:
        None.
    Returns:
        None.
    Raises:
        RuntimeError: Installed source or RECORD is unreviewed or incomplete.
        OSError: An installed file cannot be read or written.
    """
    distribution = metadata.distribution("accelerate")
    root = Path(distribution.locate_file(""))
    source = patched_source(distribution.version, (root / SOURCE_PATH).read_bytes())
    record_name = f"accelerate-{distribution.version}.dist-info/RECORD"
    record = root / record_name
    rows = list(csv.reader(io.StringIO(record.read_text())))
    if SOURCE_PATH not in {row[0] for row in rows}:
        raise RuntimeError("Accelerate RECORD does not contain the reviewed module")
    receipt = record.parent / "NPA-CHECKPOINT-SECURITY.json"
    files = _write_repair(root, source, receipt, distribution.version)
    updates = {
        path.relative_to(root).as_posix(): _record_row(root, path) for path in files
    }
    rows = [updates.pop(row[0], row) for row in rows]
    rows.extend(updates.values())
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    record.write_text(output.getvalue())


if __name__ == "__main__":
    main()
