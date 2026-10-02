"""Decode RRD identities and chunks through Rerun's supported streaming reader."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from rerun.chunk import Chunk, RrdReader, StoreEntry


@dataclass(frozen=True)
class Recording:
    """A single recording selected explicitly from an RRD archive.

    Args:
        reader: Reader retaining the archive and its store metadata.
        entry: Recording identity returned by that reader.
    Returns:
        None.
    Raises:
        None.
    """

    reader: RrdReader
    entry: StoreEntry

    def application_id(self) -> str:
        """Return the decoded application identity.

        Args:
            None.
        Returns:
            Application id stored in the RRD.
        Raises:
            None.
        """
        return self.entry.application_id

    def recording_id(self) -> str:
        """Return the decoded recording identity.

        Args:
            None.
        Returns:
            Recording id stored in the RRD.
        Raises:
            None.
        """
        return self.entry.recording_id

    def chunks(self) -> Iterator[Chunk]:
        """Stream this recording's physical chunks, including legacy RRDs.

        Args:
            None.
        Returns:
            Iterator over decoded chunks from the selected recording only.
        Raises:
            ValueError: The selected store is not part of the archive.
            RuntimeError: Encoded chunks cannot be decoded.
        """
        # Unlike store(), stream() also reads old RRDs without a footer manifest.
        return iter(self.reader.stream(store=self.entry))


def load_recordings(path: str | Path) -> list[Recording]:
    """List actual recording stores while excluding blueprint stores.

    Args:
        path: Local RRD file to decode.
    Returns:
        Recording views retaining the archive reader.
    Raises:
        RuntimeError: The RRD file or its store metadata is invalid.
    """
    reader = RrdReader(path)
    return [Recording(reader, entry) for entry in reader.recordings()]


def load_recording(path: str | Path) -> Recording:
    """Decode exactly one recording, rejecting empty or ambiguous archives.

    Args:
        path: Local RRD file to decode.
    Returns:
        The archive's single recording, excluding any blueprints.
    Raises:
        ValueError: The archive does not contain exactly one recording.
        RuntimeError: The RRD file or its store metadata is invalid.
    """
    recordings = load_recordings(path)
    if len(recordings) != 1:
        raise ValueError(f"Expected exactly one recording; found {len(recordings)}")
    return recordings[0]
