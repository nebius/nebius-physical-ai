"""Create metadata-free preview media and an evidence-based comparison film."""

from __future__ import annotations

from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, ImageOps

_INK = "#0a121b"
_TEXT = "#ecf2f3"
_MUTED = "#a5b3bc"
_LIME = "#c5f46a"
_CORAL = "#ff9a87"
_SIZE = (1600, 900)


def transcode(source: Path, target: Path, poster: Path) -> dict:
    """Decode every frame into a silent preview without inherited metadata.

    Args:
        source: Verified source media.
        target: New H.264 MP4 path.
        poster: New JPEG preview path.
    Returns:
        Frame count, duration, and preview SHA-256.
    Raises:
        ValueError: Media has no usable video timeline.
        av.FFmpegError: Decoding or encoding fails.
    """
    from npa.workflows.video_sweep.artifacts import file_digest

    with av.open(str(source)) as incoming, av.open(str(target), "w") as outgoing:
        video = incoming.streams.video[0]
        rate = video.average_rate
        if rate is None or rate <= 0:
            raise ValueError("Video has no frame rate")
        stream = _stream(outgoing, rate, (video.width, video.height))
        count = 0
        for frame in incoming.decode(video=0):
            pixels = frame.to_image()
            if count == 0:
                pixels.save(poster)
            _encode(outgoing, stream, pixels)
            count += 1
        _flush(outgoing, stream)
    if count < 2:
        raise ValueError("Video has fewer than two frames")
    return {
        "frames": count,
        "duration": count / float(rate),
        "sha256": file_digest(target),
    }


def render_movie(directory: Path, summary: dict) -> None:
    """Encode a complete comparison chapter for every candidate and its source.

    Args:
        directory: Materialized preview directory.
        summary: Allowlisted result metadata.
    Returns:
        None.
    Raises:
        av.FFmpegError: Video reading or encoding fails.
    """
    with av.open(str(directory / "demo.mp4"), "w") as output:
        stream = _stream(output, 24, _SIZE)
        _hold(output, stream, _title(summary), 4)
        for index, row in enumerate(summary["candidates"]):
            _chapter(output, stream, directory, summary, row, index)
        _hold(output, stream, _outcome(summary), 5)
        _flush(output, stream)


def _stream(container, rate, size):
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    return stream


def _encode(container, stream, pixels):
    frame = av.VideoFrame.from_image(pixels)
    for packet in stream.encode(frame):
        container.mux(packet)


def _flush(container, stream):
    for packet in stream.encode():
        container.mux(packet)


def _font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _text(canvas, xy, text, size=24, color=_TEXT):
    ImageDraw.Draw(canvas).text(xy, text, font=_font(size), fill=color)


def _base(kicker):
    canvas = Image.new("RGB", _SIZE, _INK)
    draw = ImageDraw.Draw(canvas)
    draw.line((64, 105, 1536, 105), fill="#2a3845", width=2)
    draw.rectangle((64, 48, 83, 67), fill=_LIME)
    _text(canvas, (98, 43), "NEBIUS  /  PHYSICAL AI", 22)
    _text(canvas, (64, 144), kicker, 20, _LIME)
    _text(canvas, (64, 842), "VIDEO VARIANT SWEEP    /    RECORDED RESULTS", 18, _MUTED)
    return canvas


def _title(summary):
    canvas = _base("CONDITION  /  GENERATE  /  REVIEW  /  PUBLISH")
    _text(canvas, (64, 223), "More variation.", 88)
    _text(canvas, (64, 320), "A visible quality gate.", 88, _LIME)
    _text(
        canvas,
        (68, 485),
        f"Real {summary.get('generator', 'Cosmos Transfer 2.5')} outputs, compared with their source.",
        30,
    )
    stats = f"{len(summary['sources'])} source(s)     {len(summary['candidates'])} variants     {summary['accepted']} accepted"
    _text(canvas, (68, 565), stats, 32)
    _text(
        canvas,
        (68, 666),
        "Every shown result is bound to verified media and publication receipts.",
        23,
        _MUTED,
    )
    _text(
        canvas,
        (68, 705),
        "Recorded component evidence; full workflow completion is separate.",
        23,
        _MUTED,
    )
    return canvas


def _timeline(container):
    frames = iter(container.decode(video=0))
    current, following = next(frames), next(frames, None)

    def at(seconds):
        nonlocal current, following
        while following is not None and float(following.time or 0) <= seconds:
            current, following = following, next(frames, None)
        return ImageOps.pad(current.to_image(), (714, 402), color=_INK)

    return at


def _chapter(output, stream, directory, summary, row, index):
    original = next(
        source for source in summary["sources"] if source["name"] == row["source"]
    )
    duration = max(original["duration"], row["duration"])
    canvas = _comparison(summary, row, index)
    with av.open(str(directory / (row["source"] + ".mp4"))) as source:
        with av.open(str(directory / (row["name"] + ".mp4"))) as candidate:
            before, after = _timeline(source), _timeline(candidate)
            _comparison_frames(output, stream, canvas, before, after, duration)


def _comparison_frames(output, stream, canvas, before, after, duration):
    for frame_index in range(round((duration + 2) * 24)):
        seconds = min(max(frame_index / 24 - 1, 0), duration)
        frame = canvas.copy()
        frame.paste(before(seconds), (64, 313))
        frame.paste(after(seconds), (822, 313))
        draw = ImageDraw.Draw(frame)
        draw.rectangle(
            (64, 798, 64 + int(1472 * min(seconds / duration, 1)), 802), fill=_LIME
        )
        _encode(output, stream, frame)


def _comparison(summary, row, index):
    accepted = row["accepted"]
    color = _LIME if accepted else _CORAL
    canvas = _base(f"VARIANT {index + 1:02d}  /  SEED {row['seed']}")
    _text(
        canvas,
        (64, 190),
        "Accepted for the dataset" if accepted else "Held out of the dataset",
        54,
        color,
    )
    _text(canvas, (64, 272), "SOURCE", 20, _MUTED)
    _text(canvas, (822, 272), "GENERATED  /  COSMOS TRANSFER 2.5", 20, _MUTED)
    _text(
        canvas,
        (64, 745),
        f"Visual score  {row['score']:.2f}     Gate  {summary['threshold']:.2f}",
        26,
        color,
    )
    _text(
        canvas,
        (822, 745),
        f"{summary['judge']}  /  {summary['samples']} sampled frames",
        22,
        _MUTED,
    )
    return canvas


def _outcome(summary):
    published = summary.get("published", True)
    canvas = _base("VERIFIED PUBLICATION" if published else "ALL VARIANTS HELD OUT")
    count, accepted = len(summary["candidates"]), summary["accepted"]
    _text(
        canvas,
        (64, 218),
        f"{accepted} accepted.  {count - accepted} held out.",
        76,
        _LIME,
    )
    _text(canvas, (68, 367), "All candidates retain Postgres + MLflow lineage.", 36)
    publication = (
        "Only accepted clips enter the published dataset."
        if published
        else "No dataset or next-run inventory was published."
    )
    following = (
        "The next-run inventory is ready for another sweep."
        if published
        else "Review the results before choosing the next inputs."
    )
    _text(canvas, (68, 427), publication, 36)
    _text(canvas, (68, 487), following, 36)
    _text(
        canvas,
        (68, 635),
        "The judge evaluates sampled frames; it does not certify every frame.",
        24,
        _MUTED,
    )
    _text(
        canvas,
        (68, 683),
        "Open index.html to scrub the footage and explore the gate.",
        24,
        _MUTED,
    )
    return canvas


def _hold(output, stream, canvas, seconds):
    for _ in range(seconds * 24):
        _encode(output, stream, canvas)
