"""Remove only verified, durably published checkpoint copies from one training run."""

import errno
from pathlib import Path, PurePosixPath
import sys

from npa.workbench.flex_pi.training_artifacts import sha256_file


def cleanup_checkpoint_copies(work, *, original, restored, step, manifest):
    """Release published state copies after successful result publication and resume.

    Args:
        work: This invocation's private persistent working directory.
        original: Generated checkpoint state directory.
        restored: Independently downloaded checkpoint state directory.
        step: Verified checkpoint update number.
        manifest: Published checkpoint file sizes and hashes.
    Returns:
        None; cleanup failures retain remaining files and warn on stderr.
    Raises:
        None; filesystem and manifest verification failures are reported safely.
    """
    try:
        roots, files = _verified_copies(work, original, restored, step, manifest)
        for path in files:
            _require_no_symlinks(path)
            path.unlink()
        _remove_empty_directories(roots, files)
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "Published checkpoint cache cleanup incomplete; retained files require operator review.",
            file=sys.stderr,
        )


def _verified_copies(work, original, restored, step, manifest):
    work = Path(work).absolute()
    if type(step) is not int or step < 1:
        raise ValueError("invalid checkpoint step")
    roots = [Path(original).absolute(), Path(restored).absolute()]
    expected = [
        work / "checkpoints" / "state" / f"step_{step:06d}",
        work / "restored-state",
    ]
    if roots != expected:
        raise ValueError("checkpoint copies do not belong to this run")
    for root in roots:
        _require_no_symlinks(root)
        if not root.is_dir():
            raise ValueError("checkpoint copy is not a directory")
    rows = _manifest_rows(manifest)
    files = []
    for root in roots:
        for relative, row in rows:
            path = root / relative
            _require_no_symlinks(path)
            if not path.is_file() or path.stat().st_size != row["size"]:
                raise ValueError("checkpoint copy size changed")
            if sha256_file(path) != row["sha256"]:
                raise ValueError("checkpoint copy content changed")
            files.append(path)
    return roots, files


def _manifest_rows(manifest):
    rows = []
    seen = set()
    for row in manifest["files"]:
        value = row["path"]
        relative = PurePosixPath(value)
        if (
            not value
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in value
            or relative.as_posix() != value
            or not relative.parts
            or value in seen
        ):
            raise ValueError("invalid checkpoint manifest path")
        seen.add(value)
        rows.append((relative, row))
    if not rows:
        raise ValueError("empty checkpoint manifest")
    return rows


def _require_no_symlinks(path):
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("checkpoint cleanup path contains a symlink")


def _remove_empty_directories(roots, files):
    directories = set(roots)
    for path in files:
        for parent in path.parents:
            if parent in roots:
                break
            directories.add(parent)
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        _require_no_symlinks(directory)
        try:
            directory.rmdir()
        except OSError as error:
            if error.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise
