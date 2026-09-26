"""Verify LOCK SOURCES_XZ [DELIVERY_DIRECTORY] without fetching source bytes."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import lzma
from pathlib import Path
import re
import sys


def _require(condition: object, reason: str) -> None:
    """Refuse an inconsistent source claim with the existing diagnostic prefix."""
    if not condition:
        raise SystemExit("source package refused: " + reason)


def _one(pattern: str, text: str) -> str:
    """Read an unambiguous lock field."""
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    _require(len(matches) == 1, "lock field cardinality")
    return matches[0]


def _paragraphs(text: str):
    """Parse Debian control paragraphs without duplicate fields."""
    for block in text.strip().split("\n\n"):
        fields = {}
        previous = None
        for line in block.splitlines():
            if line.startswith((" ", "\t")):
                _require(previous is not None, "orphan continuation")
                fields[previous] += "\n" + line.strip()
                continue
            _require(":" in line, "invalid index field")
            key, value = line.split(":", 1)
            _require(key not in fields, "duplicate index field")
            fields[key] = value.strip()
            previous = key
        yield fields


def _checksums(value: str, digits: int) -> dict[str, tuple[int, str]]:
    """Bind each checksum record to one positive size and plain filename."""
    rows = {}
    for line in value.strip().splitlines():
        parts = line.split()
        _require(len(parts) == 3, "checksum record shape")
        digest, size, name = parts
        _require(re.fullmatch(r"[0-9a-f]{%d}" % digits, digest), "checksum syntax")
        _require(size.isdecimal() and int(size) > 0, "checksum size")
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_~-]*", name), "unsafe name")
        _require(name not in rows, "duplicate checksum record")
        rows[name] = (int(size), digest)
    _require(rows, "empty checksums")
    return rows


def _locked_records(lock: str) -> dict[str, tuple[int, str]]:
    """Require the exact four locked artifacts and immutable snapshot locators."""
    rows = re.findall(
        r'^      - \{filename: ([^,]+), bytes: ([0-9]+), sha256: ([0-9a-f]{64}), locator: "([^"]+)"\}$',
        lock,
        flags=re.MULTILINE,
    )
    _require(
        len(rows) == len(re.findall(r"^      - ", lock, re.MULTILINE)),
        "lock artifact syntax",
    )
    _require(
        len(rows) == 4 and len({row[0] for row in rows}) == 4,
        "lock artifact population",
    )
    directory = _one(r"^    directory: (.+)$", lock)
    snapshot = _one(r'^snapshot: "([0-9]{8}T[0-9]{6}Z)"$', lock)
    for name, _size, _digest, locator in rows:
        _require(
            locator
            == f"https://snapshot.ubuntu.com/ubuntu/{snapshot}/{directory}/{name}",
            "artifact locator",
        )
    return {name: (int(size), digest) for name, size, digest, _ in rows}


def _decoded_index(lock: str, index: Path) -> str:
    """Check compressed index bytes before bounded single-stream decoding."""
    _require(index.is_file() and not index.is_symlink(), "index type")
    size = int(_one(r"^      bytes: ([0-9]+)$", lock))
    _require(index.stat().st_size == size, "index size")
    compressed = index.read_bytes()
    _require(
        hashlib.sha256(compressed).hexdigest()
        == _one(r"^      sha256: ([0-9a-f]{64})$", lock),
        "index digest",
    )
    decoder = lzma.LZMADecompressor()
    decoded = decoder.decompress(compressed, max_length=64 * 1024 * 1024)
    _require(decoder.eof and not decoder.unused_data, "index compression boundary")
    return decoded.decode("utf-8")


def _verified_stanza(lock: str, index: Path) -> dict[str, str]:
    """Bind the selected source, version and directory to the verified index."""
    decoded = _decoded_index(lock, index)
    source = _one(r"^    source: (.+)$", lock)
    version = _one(r"^    version: (.+)$", lock)
    selected = [
        row
        for row in _paragraphs(decoded)
        if row.get("Package") == source and row.get("Version") == version
    ]
    _require(len(selected) == 1, "selected source cardinality")
    row = selected[0]
    _require(
        row.get("Directory") == _one(r"^    directory: (.+)$", lock), "source directory"
    )
    return row


def _verify_delivered(directory: Path, sha256_rows: dict, md5_rows: dict) -> None:
    """Verify the exact regular-file delivery population and its locked bytes."""
    _require(directory.is_dir() and not directory.is_symlink(), "source directory type")
    _require(
        {path.name for path in directory.iterdir()} == set(sha256_rows),
        "delivered population",
    )
    for name, (size, digest) in sha256_rows.items():
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), "artifact type")
        _require(path.stat().st_size == size, "artifact size")
        payload = path.read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == digest, "artifact SHA256")
        # MD5 is only an additional signed-index consistency check, never trust.
        _require(
            hashlib.md5(payload, usedforsecurity=False).hexdigest()
            == md5_rows[name][1],
            "artifact Files checksum",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify LOCK SOURCES_XZ [DELIVERY_DIRECTORY] against locked source identity.

    Args:
        argv: Two input paths and an optional delivery directory; defaults to CLI arguments.
    Returns:
        Zero only after every applicable source/index/delivery check passes.
    Raises:
        SystemExit: Invalid arguments or a source claim fails verification.
        OSError: An input cannot be read.
        UnicodeError: A text input is not valid UTF-8.
        lzma.LZMAError: The source index is not valid XZ data.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    _require(len(arguments) in {2, 3}, "usage: LOCK SOURCES_XZ [DELIVERY_DIRECTORY]")
    lock = Path(arguments[0]).read_text(encoding="utf-8")
    stanza = _verified_stanza(lock, Path(arguments[1]))
    sha256_rows = _checksums(stanza.get("Checksums-Sha256", ""), 64)
    md5_rows = _checksums(stanza.get("Files", ""), 32)
    _require(sha256_rows == _locked_records(lock), "index-to-lock artifact identity")
    _require(
        {name: row[0] for name, row in sha256_rows.items()}
        == {name: row[0] for name, row in md5_rows.items()},
        "Files population/size",
    )
    if len(arguments) == 3:
        _verify_delivered(Path(arguments[2]), sha256_rows, md5_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
