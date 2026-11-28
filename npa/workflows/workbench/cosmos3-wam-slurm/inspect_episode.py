"""Inspect a LIBERO-compatible LeRobot v3 action window and its two camera streams on CPU."""

import argparse
import bisect
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

CAMERAS = ("observation.images.image", "observation.images.wrist_image")
TOLERANCE = 0.0001
WINDOW = 16


def contained(root, path):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("dataset file resolves outside the dataset")
    return resolved


def episode_rows(root, relative, episode):
    from pyarrow import parquet

    rows = []
    for path in sorted(contained(root, root / relative).glob("chunk-*/*.parquet")):
        rows.extend(
            parquet.read_table(
                contained(root, path), filters=[("episode_index", "=", episode)]
            ).to_pylist()
        )
    return rows


def vector(row, field, size):
    values = row[field]
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError(
            f"{field} needs {size} finite values at frame {row['frame_index']}"
        )
    return values


def action_window(rows, start):
    rows = sorted(rows, key=lambda row: row["frame_index"])
    window = [row for row in rows if start <= row["frame_index"] <= start + WINDOW]
    if [row["frame_index"] for row in window] != list(range(start, start + WINDOW + 1)):
        raise ValueError(
            "expected 17 consecutive, unique state frames in the selected episode"
        )
    times = [row["timestamp"] for row in window]
    if any(
        not math.isfinite(t) or abs(t - row["frame_index"] / 20) > TOLERANCE
        for row, t in zip(window, times, strict=True)
    ):
        raise ValueError(
            "expected episode-relative state timestamps on the 20 Hz frame grid"
        )
    return (
        times,
        [vector(row, "observation.state", 8) for row in window],
        [vector(row, "action", 7) for row in window[:-1]],
    )


def nearest_error(time, candidates):
    index = bisect.bisect_left(candidates, time)
    return min(
        abs(candidates[j] - time)
        for j in (index - 1, index)
        if 0 <= j < len(candidates)
    )


def decode_frames(path, key, expected):
    import av

    decoded = []
    image = None
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if frame.pts is None or (frame.width, frame.height) != (256, 256):
                raise ValueError(
                    f"{key} needs 256x256 frames with presentation timestamps"
                )
            ts = float(frame.pts * frame.time_base)
            if ts < expected[0] - 0.05:
                continue
            if decoded and ts <= decoded[-1]:
                raise ValueError(f"{key} has non-increasing timestamps")
            decoded.append(ts)
            if abs(ts - expected[0]) <= TOLERANCE:
                image = frame.to_image()
            if ts > expected[-1] + 0.05:
                break
    return image, decoded


def inspect_camera(root, template, episode, key, times):
    prefix = f"videos/{key}/"
    path = (
        root
        / template.format(
            video_key=key,
            chunk_index=episode[prefix + "chunk_index"],
            file_index=episode[prefix + "file_index"],
        )
    ).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{key} video resolves outside the dataset")
    expected = [episode[prefix + "from_timestamp"] + t for t in times]
    if (
        not all(math.isfinite(t) for t in expected)
        or expected[-1] >= episode[prefix + "to_timestamp"] + TOLERANCE
    ):
        raise ValueError(f"{key} window extends beyond the episode's video interval")
    image, decoded = decode_frames(path, key, expected)
    if image is None or max(nearest_error(t, decoded) for t in expected) > TOLERANCE:
        raise ValueError(
            f"{key} camera timestamps do not align with the selected state window"
        )
    if all(nearest_error(t + 0.025, decoded) <= TOLERANCE for t in expected):
        raise ValueError(f"{key} unexpectedly accepted a 25 ms timestamp shift")
    return image, {
        "matched_frames": len(times),
        "max_error_ns": round(max(nearest_error(t, decoded) for t in expected) * 1e9),
        "half_frame_shift_rejected": True,
    }


def inspect(root, episode_id, start, output):
    root = root.resolve(strict=True)
    if output.exists() or start < 0 or episode_id < 0:
        raise ValueError(
            "use a fresh output directory and nonnegative episode and frame indices"
        )
    if output.resolve().is_relative_to(root):
        raise ValueError("choose an output directory outside the dataset")
    info = json.loads(contained(root, root / "meta/info.json").read_text())
    if info["fps"] != 20 or any(key not in info["features"] for key in CAMERAS):
        raise ValueError(
            "the LIBERO recipe expects 20 Hz front and wrist camera streams"
        )
    episodes = episode_rows(root, "meta/episodes", episode_id)
    if len(episodes) != 1:
        raise ValueError("expected one metadata record for the selected episode")
    times, states, actions = action_window(
        episode_rows(root, "data", episode_id), start
    )
    result = {
        "episode": episode_id,
        "start_frame": start,
        "timestamps_s": times,
        "raw_states": states,
        "raw_actions": actions,
        "cameras": {},
    }
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "window"
        stage.mkdir()
        for key, view in zip(CAMERAS, ("front", "wrist"), strict=True):
            image, result["cameras"][key] = inspect_camera(
                root, info["video_path"], episodes[0], key, times
            )
            image.save(stage / f"{view}.png")
        (stage / "window.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--start-frame", default=0, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    inspect(args.dataset, args.episode, args.start_frame, args.output_dir)
