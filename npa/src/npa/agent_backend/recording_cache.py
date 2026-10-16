"""Request-owned verified viewer inputs, separate from published viewer copies.

Each lease owns one private directory. No glob pruning, cross-request unlink,
or replacement of a reader's verified generation is permitted. Callers release
after publishing their viewer copy; finalization also handles exception paths.
"""

from pathlib import Path
import tempfile


class VerifiedRecordingPath(type(Path())):
    """A Path retaining exclusive cleanup ownership of its verified bytes."""

    def release(self) -> None:
        self._recording_owner.cleanup()


class VerifiedRecordingString(str):
    """JSON-compatible pathname keeping the same lease alive in result dicts."""

    def __new__(cls, path: VerifiedRecordingPath):
        value = super().__new__(cls, str(path))
        value._recording_owner = path._recording_owner
        return value

    def release(self) -> None:
        self._recording_owner.cleanup()


def retain_verified_recording(
    source: Path, root: Path, name: str
) -> VerifiedRecordingPath:
    """Move verified staging into a unique request-owned lease.

    Args:
        source: Fully verified staging file to transfer, not a published copy.
        root: Existing private recordings directory.
        name: Local leaf filename within the unique lease directory.

    Returns:
        Owned path to retain through synchronous apply, then explicitly release.

    Raises:
        ValueError: The name is not a safe local leaf.
        OSError: Private staging creation or transfer fails.
    """
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("verified recording requires a local filename")
    owner = tempfile.TemporaryDirectory(prefix=".verified-recording-", dir=root)
    try:
        path = VerifiedRecordingPath(Path(owner.name) / name)
        path._recording_owner = owner
        source.replace(path)
        return path
    except BaseException:
        owner.cleanup()
        raise


def release_verified_recording(value) -> None:
    """Release only an actual lease, never an arbitrary local/public path.

    Args:
        value: Input lease retained through apply, or an unowned value to ignore.

    Returns:
        None; repeated lease release is harmless.

    Raises:
        OSError: The owned temporary directory cannot be removed.
    """
    if isinstance(value, (VerifiedRecordingPath, VerifiedRecordingString)):
        value.release()
