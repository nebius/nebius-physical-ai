"""Read back MCAP recordings so operators can verify what Foxglove will show.

Pure reader-side helpers over the optional ``mcap`` dependency: they answer
"is this a real recording, and what is actually in it?" before the file is
published to the viewer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MCAP_MAGIC = b"\x89MCAP0\r\n"
NS_PER_S = 1_000_000_000


class McapInspectError(RuntimeError):
    """Raised when an MCAP recording cannot be read."""


@dataclass
class McapInfo:
    """Summary of an MCAP file's real contents."""

    path: str = ""
    size_bytes: int = 0
    valid_magic: bool = False
    message_count: int = 0
    channels: dict[str, int] = field(default_factory=dict)
    schemas: dict[str, str] = field(default_factory=dict)
    start_time_ns: int = 0
    end_time_ns: int = 0
    duration_s: float = 0.0
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "valid_magic": self.valid_magic,
            "message_count": self.message_count,
            "channels": dict(self.channels),
            "schemas": dict(self.schemas),
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_s": self.duration_s,
            "metadata": {k: dict(v) for k, v in self.metadata.items()},
        }


def has_mcap_magic(path: str | Path) -> bool:
    """Return True when the file starts with the MCAP magic record."""
    target = Path(str(path or "")).expanduser()
    try:
        with target.open("rb") as handle:
            return handle.read(len(MCAP_MAGIC)) == MCAP_MAGIC
    except OSError:
        return False


def summarize_mcap(path: str | Path) -> McapInfo:
    """Return channel/schema/message statistics for an MCAP recording."""
    target = Path(str(path or "")).expanduser()
    if not target.is_file():
        raise McapInspectError(f"MCAP file not found: {target}")

    info = McapInfo(
        path=str(target),
        size_bytes=target.stat().st_size,
        valid_magic=has_mcap_magic(target),
    )
    if not info.valid_magic:
        raise McapInspectError(
            f"{target} does not start with the MCAP magic record — not an MCAP recording"
        )

    try:
        from mcap.reader import make_reader  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch
        raise McapInspectError(
            "Reading MCAP requires the optional 'mcap' dependency. "
            'Install it with: pip install "npa[foxglove]"'
        ) from exc

    with target.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        if summary is not None:
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                info.schemas[channel.topic] = schema.name if schema else ""
                info.channels.setdefault(channel.topic, 0)
            for channel_id, stats in (
                summary.statistics.channel_message_counts.items()
                if summary.statistics
                else {}.items()
            ):
                channel = summary.channels.get(channel_id)
                if channel is not None:
                    info.channels[channel.topic] = int(stats)
            if summary.statistics is not None:
                info.message_count = int(summary.statistics.message_count)
                info.start_time_ns = int(summary.statistics.message_start_time)
                info.end_time_ns = int(summary.statistics.message_end_time)

        if not info.message_count:
            # No summary section (streamed writer): count messages directly.
            for schema, channel, message in reader.iter_messages():
                info.message_count += 1
                info.channels[channel.topic] = info.channels.get(channel.topic, 0) + 1
                if schema is not None:
                    info.schemas[channel.topic] = schema.name
                info.start_time_ns = (
                    message.log_time
                    if not info.start_time_ns
                    else min(info.start_time_ns, message.log_time)
                )
                info.end_time_ns = max(info.end_time_ns, message.log_time)

        handle.seek(0)
        reader = make_reader(handle)
        for record in reader.iter_metadata():
            info.metadata[record.name] = dict(record.metadata)

    if info.end_time_ns and info.start_time_ns:
        info.duration_s = round((info.end_time_ns - info.start_time_ns) / NS_PER_S, 3)
    return info


def format_mcap_info(info: McapInfo) -> str:
    """Render an operator-facing summary of an MCAP recording."""
    lines = [
        f"path: {info.path}",
        f"size: {info.size_bytes} bytes",
        f"messages: {info.message_count}",
        f"duration: {info.duration_s}s",
    ]
    if info.channels:
        lines.append("channels:")
        for topic in sorted(info.channels):
            schema = info.schemas.get(topic) or "(no schema)"
            lines.append(f"  {topic}  x{info.channels[topic]}  [{schema}]")
    if info.metadata:
        lines.append("metadata:")
        for name in sorted(info.metadata):
            lines.append(f"  {name}: {json.dumps(info.metadata[name], sort_keys=True)}")
    return "\n".join(lines)
