"""Verify one canonical robomimic Debian attribution against official bytes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Any, BinaryIO

from npa._public_https import PublicDownloadError, download_public_https


REPOSITORY_PATH = "npa/docker/workbench/robomimic/debian-packages.lock"
ATTRIBUTION_LINES = frozenset({249})
LOCK_GIT_BLOB_SHA1 = "7cc3fa143d8b7a55c9bc9e34307cbc09445ee973"
LOCK_SHA256 = "aebaefa21f527584d4a46cba8b22d0c691414a4762acd364252d746e03d3504e"
LOCK_SIZE = 76_026

_SNAPSHOT = "20260906T183022Z"
_SOURCE_ID = "debian:libbsd@0.11.7-2"
_SOURCE_NAME = "libbsd"
_SOURCE_VERSION = "0.11.7-2"
_SOURCE_DIRECTORY = "pool/main/libb/libbsd"
_COPYRIGHT_SHA256 = (
    "00fd7be5d907a6bcc409a80f2d565f508673db4761d54623691f89ae3aa20fc7"
)
_COPYRIGHT_SIZE = 23_960

_SOURCE_FILES = (
    (
        "21c62d65fa3b914d765b733fb0e03d331db830df45f1a5225f95902567f05146",
        2_330,
        "libbsd_0.11.7-2.dsc",
    ),
    (
        "9baa186059ebbf25c06308e9f991fda31f7183c0f24931826d83aa6abd8a0261",
        418_508,
        "libbsd_0.11.7.orig.tar.xz",
    ),
    (
        "b470d3fa5ad6948de7a85891e652970828f26eb7057028d57b94fa8644af934a",
        833,
        "libbsd_0.11.7.orig.tar.xz.asc",
    ),
    (
        "e588e52a99415226767362637071764ebfaf454450bda64d53652e7a451d3e67",
        18_116,
        "libbsd_0.11.7-2.debian.tar.xz",
    ),
)


@dataclass(frozen=True)
class _PublicProof:
    name: str
    url: str
    sha256: str
    size: int
    host: str


_INRELEASE = _PublicProof(
    "robomimic-debian-bookworm-inrelease",
    (
        "https://snapshot.debian.org/archive/debian/"
        f"{_SNAPSHOT}/dists/bookworm/InRelease"
    ),
    "77737fa4b34f2693e982cc9ee35736816c35a7778fc2d326cc1bbf5b301fe1aa",
    151_075,
    "snapshot.debian.org",
)
_SOURCES = _PublicProof(
    "robomimic-debian-bookworm-sources.xz",
    (
        "https://snapshot.debian.org/archive/debian/"
        f"{_SNAPSHOT}/dists/bookworm/main/source/Sources.xz"
    ),
    "92d11a035571df011f06f28c61cb9825130ef1d92abb2f2e0525e8674dc44af8",
    9_491_888,
    "snapshot.debian.org",
)
_FTP_MASTER_COPYRIGHT = _PublicProof(
    "robomimic-libbsd-ftp-master-copyright",
    (
        "https://metadata.ftp-master.debian.org/changelogs/main/libb/"
        "libbsd/libbsd_0.11.7-2_copyright"
    ),
    _COPYRIGHT_SHA256,
    _COPYRIGHT_SIZE,
    "metadata.ftp-master.debian.org",
)
_SOURCES_COPYRIGHT = _PublicProof(
    "robomimic-libbsd-sources-copyright",
    "https://sources.debian.org/data/main/libb/libbsd/0.11.7-2/debian/copyright",
    _COPYRIGHT_SHA256,
    _COPYRIGHT_SIZE,
    "sources.debian.org",
)
_ARCHIVE_KEY_12 = _PublicProof(
    "robomimic-debian-archive-key-12.asc",
    "https://ftp-master.debian.org/keys/archive-key-12.asc",
    "c2a9a16fde95e037bafd0fa6b7e31f41b4ff1e85851de5558f19a2a2f0e955e2",
    11_861,
    "ftp-master.debian.org",
)
_ARCHIVE_KEY_13 = _PublicProof(
    "robomimic-debian-archive-key-13.asc",
    "https://ftp-master.debian.org/keys/archive-key-13.asc",
    "6f1d277429dd7ffedcc6f8688a7ad9a458859b1139ffa026d1eeaadcbffb0da7",
    11_861,
    "ftp-master.debian.org",
)
_RELEASE_KEY_12 = _PublicProof(
    "robomimic-debian-release-12.asc",
    "https://ftp-master.debian.org/keys/release-12.asc",
    "521e9f6a9f9b92ee8d5ce74345e8cfd04028dae9db6f571259d584b293549824",
    461,
    "ftp-master.debian.org",
)
_SIGNING_KEYS = (_ARCHIVE_KEY_12, _ARCHIVE_KEY_13, _RELEASE_KEY_12)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_private_directory(proof_directory: Path) -> None:
    details = proof_directory.lstat()
    if not stat.S_ISDIR(details.st_mode) or proof_directory.is_symlink():
        raise ValueError("proof directory must be a real directory")
    if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise ValueError("proof directory must be caller-owned and private")


def _verified_file(path: Path, proof: _PublicProof) -> bytes:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise ValueError("cached proof must be a regular file")
    if (
        details.st_uid != os.getuid()
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) & 0o077
    ):
        raise ValueError("cached proof must be caller-owned and private")
    payload = path.read_bytes()
    if len(payload) != proof.size or _digest(payload) != proof.sha256:
        raise ValueError("cached proof does not match its public pin")
    return payload


@dataclass
class _PinnedOutput:
    output: BinaryIO
    expected_size: int
    received: int = 0

    def write(self, payload: bytes) -> int:
        self.received += len(payload)
        if self.received > self.expected_size:
            raise ValueError("public proof exceeds its size pin")
        return self.output.write(payload)


def _download_proof(path: Path, proof: _PublicProof) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                download_public_https(
                    proof.url,
                    _PinnedOutput(output, proof.size),
                    allowed_hosts=frozenset({proof.host}),
                )
            except PublicDownloadError:
                raise OSError("official public proof download failed") from None
            output.flush()
            os.fsync(output.fileno())
        _verified_file(temporary, proof)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _proof_payload(proof_directory: Path, proof: _PublicProof) -> bytes:
    path = proof_directory / proof.name
    if not path.exists():
        _download_proof(path, proof)
    return _verified_file(path, proof)


def _signed_body(inrelease: bytes) -> bytes:
    prefix = b"-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA256\n\n"
    signature = b"\n-----BEGIN PGP SIGNATURE-----\n"
    if not inrelease.startswith(prefix) or not inrelease.endswith(
        b"-----END PGP SIGNATURE-----\n"
    ):
        raise ValueError("snapshot release is not the pinned signed form")
    parts = inrelease[len(prefix) :].split(signature)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("snapshot release signature framing mismatch")
    return parts[0]


def _signed_source_pin(inrelease: bytes) -> tuple[str, int, str]:
    target = "main/source/Sources.xz"
    rows: list[tuple[str, int, str]] = []
    in_sha256 = False
    for raw_line in _signed_body(inrelease).decode("utf-8").splitlines():
        if raw_line == "SHA256:":
            in_sha256 = True
            continue
        if in_sha256 and raw_line and not raw_line.startswith(" "):
            in_sha256 = False
        if in_sha256 and raw_line.strip().endswith(target):
            fields = raw_line.split()
            if len(fields) != 3:
                raise ValueError("signed source-index record is malformed")
            rows.append((fields[0], int(fields[1]), fields[2]))
    expected = (_SOURCES.sha256, _SOURCES.size, target)
    if rows != [expected]:
        raise ValueError("signed source-index pin mismatch")
    return rows[0]


def _create_keyring(gpg: str, key_home: Path, key_paths: list[Path]) -> Path:
    imported = subprocess.run(
        [
            gpg,
            "--no-options",
            "--batch",
            "--quiet",
            "--homedir",
            str(key_home),
            "--import",
            *(str(path) for path in key_paths),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if imported.returncode:
        raise ValueError("official signing keys could not be imported")
    exported = subprocess.run(
        [
            gpg,
            "--no-options",
            "--batch",
            "--quiet",
            "--homedir",
            str(key_home),
            "--export",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if exported.returncode or not exported.stdout:
        raise ValueError("official signing keyring could not be created")
    keyring = key_home / "trustedkeys.gpg"
    keyring.write_bytes(exported.stdout)
    keyring.chmod(0o600)
    return keyring


def _verify_openpgp_signature(proof_directory: Path) -> None:
    gpg = shutil.which("gpg")
    gpgv = shutil.which("gpgv")
    if gpg is None or gpgv is None:
        raise ValueError("OpenPGP verification tools are unavailable")
    key_paths = []
    for key in _SIGNING_KEYS:
        _proof_payload(proof_directory, key)
        key_paths.append(proof_directory / key.name)

    with tempfile.TemporaryDirectory(dir=proof_directory) as temporary:
        key_home = Path(temporary)
        key_home.chmod(0o700)
        keyring = _create_keyring(gpg, key_home, key_paths)
        verified = subprocess.run(
            [
                gpgv,
                "--keyring",
                str(keyring),
                str(proof_directory / _INRELEASE.name),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if verified.returncode:
            raise ValueError("snapshot release signature verification failed")


def _deb822_records(payload: bytes) -> list[dict[str, str]]:
    try:
        text = lzma.decompress(payload).decode("utf-8")
    except (lzma.LZMAError, UnicodeDecodeError):
        raise ValueError("source index is not canonical UTF-8 xz data") from None
    records: list[dict[str, str]] = []
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        record: dict[str, str] = {}
        current: str | None = None
        for line in paragraph.splitlines():
            if line.startswith(" "):
                if current is None:
                    raise ValueError("source index has an orphan continuation")
                continuation = line[1:]
                record[current] = (
                    continuation
                    if not record[current]
                    else record[current] + "\n" + continuation
                )
                continue
            if ":" not in line:
                raise ValueError("source index field is malformed")
            current, value = line.split(":", 1)
            if not current or current in record:
                raise ValueError("source index has duplicate or empty fields")
            record[current] = value.lstrip()
        records.append(record)
    return records


def _checksum_rows(value: str) -> tuple[tuple[str, int, str], ...]:
    rows = []
    for line in value.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("source checksum record is malformed")
        try:
            size = int(fields[1])
        except ValueError:
            raise ValueError("source checksum size is malformed") from None
        rows.append((fields[0], size, fields[2]))
    return tuple(rows)


def _verify_snapshot_source(inrelease: bytes, source_index: bytes) -> None:
    _signed_source_pin(inrelease)
    matches = [
        record
        for record in _deb822_records(source_index)
        if record.get("Package") == _SOURCE_NAME
        and record.get("Version") == _SOURCE_VERSION
    ]
    if len(matches) != 1:
        raise ValueError("pinned snapshot source identity is not unique")
    source = matches[0]
    if source.get("Directory") != _SOURCE_DIRECTORY:
        raise ValueError("pinned snapshot source directory mismatch")
    if _checksum_rows(source.get("Checksums-Sha256", "")) != _SOURCE_FILES:
        raise ValueError("pinned snapshot source artifacts mismatch")


def _verify_lock_identity(lock: bytes) -> None:
    if len(lock) != LOCK_SIZE or _digest(lock) != LOCK_SHA256:
        raise ValueError("license lock does not match the canonical public bytes")
    try:
        document = json.loads(lock)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("license lock is not canonical JSON") from None
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("license lock has no source records")
    matches = [source for source in sources if source.get("id") == _SOURCE_ID]
    if len(matches) != 1:
        raise ValueError("license lock source identity mismatch")
    source = matches[0]
    if (
        document.get("snapshot") != _SNAPSHOT
        or source.get("name") != _SOURCE_NAME
        or source.get("version") != _SOURCE_VERSION
        or source.get("license_reference_url") != _FTP_MASTER_COPYRIGHT.url
    ):
        raise ValueError("license lock provenance mismatch")


def _proof_record(proof: _PublicProof) -> dict[str, Any]:
    return {"url": proof.url, "sha256": proof.sha256, "size": proof.size}


def verify_public_license_lock(
    lock: bytes, proof_directory: Path
) -> dict[str, Any]:
    """Verify the exact lock against pinned, mutually confirming Debian bytes.

    Args:
        lock: Complete candidate lock bytes.
        proof_directory: Existing caller-owned directory with mode ``0700``.

    Returns:
        Public hashes, sizes, and URLs for the verified provenance inputs.

    Raises:
        OSError: An official proof cannot be downloaded, cached, or read.
        ValueError: Any byte, Git-bound identity, signature, source, or mode
            check fails.
    """
    _verify_lock_identity(lock)
    _require_private_directory(proof_directory)

    inrelease = _proof_payload(proof_directory, _INRELEASE)
    _verify_openpgp_signature(proof_directory)
    source_index = _proof_payload(proof_directory, _SOURCES)
    _verify_snapshot_source(inrelease, source_index)

    ftp_master = _proof_payload(proof_directory, _FTP_MASTER_COPYRIGHT)
    sources = _proof_payload(proof_directory, _SOURCES_COPYRIGHT)
    if ftp_master != sources:
        raise ValueError("independent official copyright bytes differ")

    return {
        "lock": {"sha256": LOCK_SHA256, "size": LOCK_SIZE},
        "signed_snapshot": _proof_record(_INRELEASE),
        "signing_keys": [_proof_record(key) for key in _SIGNING_KEYS],
        "source_index": _proof_record(_SOURCES),
        "copyright_sources": [
            _proof_record(_FTP_MASTER_COPYRIGHT),
            _proof_record(_SOURCES_COPYRIGHT),
        ],
    }
