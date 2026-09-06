"""Apply the reviewed W&B compatibility change to the exact installed framework."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import stat
import tempfile
from pathlib import Path


ORIGINAL_SHA256 = "b036983ad0b9a6fb8e9954fa1ab927962d11844c8c3b34e8fe93470cdf9ae15f"
PATCHED_SHA256 = "d1bcb9642eae2838802abc10c01982facb8902a2e2e536563a96193d5a622afe"
RUNID_SHA256 = "34bad64508dc4b2cdac33b4930226205e47d0095031be9fdd5c1724b64b6bbd3"
WHEEL_SHA256 = "1db698d107871c66b2dcbb0cf4dc2af1ddb159ba94e957e890158ec60ab2de54"
WAND_VERSION = "0.28.2"
FRAMEWORK_REVISION = "a904d2d36b774a51dd06ff9ff906816b1a04f579"
ANCHOR = b"import wandb.util\n"
IMPORT = b"from wandb.sdk.lib.runid import generate_id as _npa_generate_wandb_id\n"
OLD_CALL = b"wandb.util.generate_id()"
NEW_CALL = b"_npa_generate_wandb_id()"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(source: bytes) -> bytes:
    if digest(source) != ORIGINAL_SHA256:
        raise ValueError("Unreviewed Cosmos framework source")
    if source.count(ANCHOR) != 1 or source.count(OLD_CALL) != 2:
        raise ValueError("Unexpected W&B compatibility patch anchors")
    patched = source.replace(ANCHOR, ANCHOR + IMPORT).replace(OLD_CALL, NEW_CALL)
    if digest(patched) != PATCHED_SHA256:
        raise ValueError("Unexpected patched framework digest")
    compile(patched, "cosmos_framework/utils/wandb_util.py", "exec")
    return patched


def identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
        info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Patch input must be a regular file with one link")
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as stream:
        opened = os.fstat(stream.fileno())
        if identity(before) != identity(opened):
            raise ValueError("Patch input changed before reading")
        data = stream.read()
        if identity(opened) != identity(os.fstat(stream.fileno())):
            raise ValueError("Patch input changed while reading")
        stream.seek(0)
        if stream.read() != data:
            raise ValueError("Patch input bytes changed while reading")
    if identity(opened) != identity(path.lstat()):
        raise ValueError("Patch input changed after reading")
    return data, opened


def replace_regular(path: Path, original_stat: os.stat_result, data: bytes) -> None:
    current, current_stat = read_regular(path)
    if identity(current_stat) != identity(original_stat) or digest(current) != ORIGINAL_SHA256:
        raise ValueError("Patch target changed before replacement")
    if digest(data) != PATCHED_SHA256:
        raise ValueError("Unreviewed replacement bytes")
    fd, temporary_name = tempfile.mkstemp(prefix=".npa-wandb-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchown(stream.fileno(), original_stat.st_uid, original_stat.st_gid)
            os.fchmod(stream.fileno(), stat.S_IMODE(original_stat.st_mode))
            os.fsync(stream.fileno())
        if identity(path.lstat()) != identity(original_stat):
            raise ValueError("Patch target changed before replacement")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def verify_dependencies() -> None:
    from packaging.requirements import Requirement

    # Check the selected wheel's actual default requirements in the vendor venv.
    # The upstream framework deliberately installs another package with --no-deps.
    for package in ("wandb", "opentelemetry-api"):
        for declaration in metadata.requires(package) or ():
            requirement = Requirement(declaration)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            if requirement.url or metadata.version(requirement.name) not in requirement.specifier:
                raise ValueError("Unsatisfied W&B runtime dependency")


def main() -> None:
    if metadata.version("wandb") != WAND_VERSION:
        raise ValueError("Unreviewed W&B version")
    verify_dependencies()
    framework = metadata.distribution("cosmos-framework")
    framework_origin = json.loads(framework.read_text("direct_url.json") or "{}")
    if framework_origin.get("vcs_info", {}).get("commit_id") != FRAMEWORK_REVISION:
        raise ValueError("Unreviewed Cosmos framework revision")
    wandb = metadata.distribution("wandb")
    target = Path(framework.locate_file("cosmos_framework/utils/wandb_util.py"))
    runid = Path(wandb.locate_file("wandb/sdk/lib/runid.py"))
    generator, _ = read_regular(runid)
    if digest(generator) != RUNID_SHA256:
        raise ValueError("Unreviewed W&B run ID implementation")
    source, original_stat = read_regular(target)
    patched = transform(source)
    from wandb.sdk.lib.runid import generate_id

    if Path(generate_id.__code__.co_filename).resolve() != runid.resolve():
        raise ValueError("Unexpected imported W&B run ID implementation")
    run_id = generate_id()
    if len(run_id) != 8 or not run_id.isascii() or not run_id.isalnum():
        raise ValueError("W&B run ID import or behavior failed")
    replace_regular(target, original_stat, patched)
    actual, _ = read_regular(target)
    if actual != patched:
        raise ValueError("Patched framework readback failed")
    provenance = {
        "schema_version": "npa.paidf.wandb-compatibility.v1",
        "framework_revision": FRAMEWORK_REVISION,
        "original_sha256": ORIGINAL_SHA256,
        "patched_sha256": PATCHED_SHA256,
        "runid_sha256": RUNID_SHA256,
        "wandb_version": WAND_VERSION,
        "wheel_sha256": WHEEL_SHA256,
        "patcher_sha256": digest(Path(__file__).read_bytes()),
        "call_sites": 2,
    }
    destination = Path("/usr/local/share/npa/paidf-wandb-compatibility.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
