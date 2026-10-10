#!/usr/bin/env python3
"""Run-local P2 driver for the Nano H200 benchmark.

Reuses the published v1 harness's dispatch, timing window, record schema and
validity gate, and adds per-cell request shape, extra_params, file uploads and
expected-output contracts. Not part of the published harness.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def multipart_with_files(files):
    def build(fields):
        boundary = "cosmos3bench" + hashlib.sha256(os.urandom(16)).hexdigest()[:24]
        chunks = []
        for key, value in fields.items():
            if key.startswith("_"):
                continue
            chunks.append((
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8"))
        for key, (filename, payload) in files.items():
            chunks.append((
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: video/mp4\r\n\r\n"
            ).encode("utf-8") + payload + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
    return build


def read_text(path, compact_json):
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()
    if compact_json:
        text = json.dumps(json.loads(text), ensure_ascii=True, separators=(",", ":"))
    return text


def probe(vv, path):
    completed = vv.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", path,
    ])
    stream = json.loads(completed.stdout)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": round(float(num) / float(den), 3),
        "frames": int(stream["nb_read_frames"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempts", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups-per-replica", type=int, default=1)
    parser.add_argument("--ports", default="8100,8101,8102,8103,8104,8105,8106,8107")
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()

    bm = load("benchmark", os.path.join(args.harness, "reproduce", "benchmark.py"))
    vv = load("validate_video", os.path.join(args.harness, "reproduce", "validate-video.py"))
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)

    shape = {key: str(value) for key, value in spec["shape"].items()}
    bm.DEFAULT_SHAPE = shape
    prompt = read_text(spec["prompt_file"], spec.get("prompt_compact_json", False))
    negative = read_text(spec["negative_file"], spec.get("negative_compact_json", False))
    files = {}
    for field, path in spec.get("files", {}).items():
        with open(path, "rb") as handle:
            files[field] = (os.path.basename(path), handle.read())
    bm.build_multipart = multipart_with_files(files)

    expected = spec.get("expected")
    if expected:
        vv.EXPECTED = dict(expected)
    bm.load_validator = lambda: vv.validate
    # The published validator samples frames 0,47,94,141,188 (the 189-frame contract).
    # Space the five samples over the cell's expected frame count instead; for 189
    # frames this yields the same indices.
    original_run = vv.run
    def run_with_cell_sampling(cmd, **kwargs):
        anchor = "select='eq(n,0)+eq(n,47)+eq(n,94)+eq(n,141)+eq(n,188)'"
        if any(anchor in part for part in cmd if isinstance(part, str)):
            last = int(vv.EXPECTED["frames"]) - 1
            picks = "+".join(f"eq(n,{round(i * last / 4)})" for i in range(5))
            cmd = [part.replace(anchor, f"select='{picks}'") for part in cmd]
        return original_run(cmd, **kwargs)
    vv.run = run_with_cell_sampling

    base = dict(shape)
    base["extra_params"] = json.dumps(spec["extra_params"])
    base.update({
        "prompt": prompt,
        "negative_prompt": negative,
        "_prompt_sha256": bm.sha256_text(prompt),
        "_negative_prompt_sha256": bm.sha256_text(negative),
        "_guardrail": "enabled" if spec["extra_params"].get("guardrails") else "disabled",
    })
    bm.prepare_output(args.output)
    ports = [int(value) for value in args.ports.split(",")]
    urls = [f"http://127.0.0.1:{port}/v1/videos/sync" for port in ports]
    clips_dir = os.path.join(args.output, "clips")

    warmups, warmup_rounds = bm.dispatch_rounds(
        urls, base, len(urls) * args.warmups_per_replica, 1,
        args.timeout, clips_dir, "warmup",
    )
    adopted = None
    if not expected:
        first = next((row for row in warmups if row.get("output_file")), None)
        if first is None:
            bm.write_output([], warmups, {"error": "no warmup output", "spec": spec}, args.output)
            print("no warmup produced output", file=sys.stderr)
            return 1
        adopted = probe(vv, os.path.join(clips_dir, first["output_file"]))
        seconds = adopted["frames"] / adopted["fps"]
        vv.EXPECTED = {
            "width": adopted["width"], "height": adopted["height"],
            "fps": adopted["fps"], "frames": adopted["frames"],
            "min_seconds": seconds - 0.08, "max_seconds": seconds + 0.08,
        }
    bm.validate_records(warmups, clips_dir)
    if any(not row["technical_valid"] for row in warmups):
        bm.write_output([], warmups, {
            "error": "warmup failed", "spec": spec, "expected": vv.EXPECTED,
            "warmup_rounds": warmup_rounds,
        }, args.output)
        print("a warmup failed validation; production window did not start", file=sys.stderr)
        return 1

    window_start_utc = bm.utc_now()
    started = time.monotonic()
    records, rounds = bm.dispatch_rounds(
        urls, base, args.attempts, args.concurrency, args.timeout, clips_dir, "production",
    )
    window_seconds = time.monotonic() - started
    window_end_utc = bm.utc_now()
    validation_run_id = bm.validate_records(records, clips_dir)

    valid = [row for row in records if row["technical_valid"]]
    clip_seconds = vv.EXPECTED["frames"] / vv.EXPECTED["fps"]
    walls = sorted(row["client_wall_s"] for row in valid)
    derived = {
        "attempts": len(records),
        "valid_attempts": len(valid),
        "clip_video_seconds": round(clip_seconds, 4),
        "window_seconds": round(window_seconds, 3),
        "clips_per_node_hour": round(len(valid) * 3600 / window_seconds, 1),
        "video_seconds_per_node_hour": round(len(valid) * clip_seconds * 3600 / window_seconds, 1),
        "median_client_wall_s": round(statistics.median(walls), 3) if walls else None,
        "p95_client_wall_s": round(walls[max(0, int(round(0.95 * len(walls))) - 1)], 3) if walls else None,
        "max_client_wall_s": walls[-1] if walls else None,
    }
    window = {
        "protocol_version": "v1-p2-runlocal",
        "comparison_basis": "nano_h200_p2_cell",
        "spec": spec,
        "expected": vv.EXPECTED,
        "expected_adopted_from_warmup": adopted is not None,
        "prompt_hashes": {"prompt": base["_prompt_sha256"], "negative_prompt": base["_negative_prompt_sha256"]},
        "window_start_utc": window_start_utc,
        "window_end_utc": window_end_utc,
        "window_seconds": round(window_seconds, 6),
        "boundary": "first production dispatch to final response persisted; validation excluded",
        "validation_run_id": validation_run_id,
        "node_gpu_count": 8,
        "concurrency": args.concurrency,
        "replicas": len(urls),
        "replica_ports": ports,
        "dispatch_schedule": "synchronized_rounds",
        "rounds": rounds,
        "warmup_rounds": warmup_rounds,
        "derived": derived,
    }
    bm.write_output(records, warmups, window, args.output)
    print(json.dumps(derived))
    return 0 if len(valid) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
