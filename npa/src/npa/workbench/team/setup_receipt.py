"""Keep crash-safe private endpoint intent separate from personal client credentials."""

import fcntl
import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from .errors import ConflictError


class SetupReceipt:
    """Journal one selected control plane before changing persistent resources.

    Args:
        path: Private operator output path.
        request: Validated setup request whose identity must remain stable.
    Returns:
        A mutable private receipt.
    Raises:
        ConflictError, OSError: Receipt ownership, identity, or I/O fails.
    """

    def __init__(self, path: Path, request):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = request.model_dump(mode="json")
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise ConflictError("setup receipt must be a regular mode-0600 file")
            self.data = json.loads(path.read_text())
            if self.data.get("request_fingerprint") != fingerprint:
                raise ConflictError("setup receipt belongs to a different installation")
        else:
            self.data = {
                "request_fingerprint": fingerprint,
                "installation_id": uuid.uuid4().hex,
                "phase": "intent",
            }
            self.save()

    def save(self, **values):
        """Persist setup state atomically before the next resource mutation.

        Args:
            **values: Non-secret identities and verified phase evidence.
        Returns:
            None.
        Raises:
            OSError: Private durable write fails.
        """
        self.data.update(values)
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(self.data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@contextmanager
def setup_lock(path: Path):
    """Serialize retries using one private operator receipt.

    Args:
        path: Selected private receipt path.
    Returns:
        Context manager holding the receipt lock.
    Raises:
        OSError: Lock cannot be opened safely.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
