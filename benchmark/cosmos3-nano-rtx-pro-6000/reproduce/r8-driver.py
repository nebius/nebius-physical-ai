#!/usr/bin/env python3
"""R8 driver: 8x1 TP-1 replicas, 24 synchronized attempts, 720p/189f T2V."""

import http.client
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid

BASE = os.environ["R8_BASE"]
CELL = os.environ.get("R8_CELL", "R8")
RUN = os.path.join(BASE, os.environ.get("R8_RUN_DIR", "r8"))
CLIPS = os.path.join(RUN, "clips")
PROMPT_FILE = os.path.join(BASE, "prompts", "example_t2v_prompt.json")
NEGATIVE_FILE = os.path.join(BASE, "prompts", "negative_prompt.json")
PORTS = [int(p) for p in os.environ.get("R8_PORTS", "8100,8101,8102,8103,8104,8105,8106,8107").split(",")]
SEEDS = [17, 23, 41]
ROUNDS = int(os.environ.get("R8_ROUNDS", "3"))
WIDTH = os.environ.get("R8_WIDTH", "1280")
HEIGHT = os.environ.get("R8_HEIGHT", "720")
FRAMES = os.environ.get("R8_FRAMES", "189")
GUIDANCE = os.environ.get("R8_GUIDANCE", "6.0")
SHAPE = {
    "width": WIDTH, "height": HEIGHT, "num_frames": FRAMES, "fps": "24",
    "num_inference_steps": "35", "guidance_scale": GUIDANCE,
    "flow_shift": "10.0", "max_sequence_length": "4096",
}
TIMEOUT = 5400
CLIP_SECONDS = int(FRAMES) / 24


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message):
    line = f"{utc()} {message}"
    print(line, flush=True)


def multipart(fields):
    boundary = "cosmos3r8" + uuid.uuid4().hex[:20]
    body = b""
    for key, value in fields.items():
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n"
        ).encode("utf-8")
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def build_fields(prompt, negative, seed):
    fields = {"model": "nvidia/Cosmos3-Nano", "prompt": prompt,
              "negative_prompt": negative, "seed": str(seed)}
    fields.update(SHAPE)
    return fields


def one_request(port, attempt_id, seed, prompt, negative, telemetry):
    body, ctype = multipart(build_fields(prompt, negative, seed))
    started = time.monotonic()
    started_utc = utc()
    record = {"attempt_id": attempt_id, "replica": f"r{port - 8100}",
              "seed": seed, "started_utc": started_utc, "kind": "production"}
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT)
        try:
            connection.request("POST", "/v1/videos/sync", body=body, headers={"Content-Type": ctype})
            response = connection.getresponse()
            payload = response.read()
        finally:
            connection.close()
        if response.status != 200:
            raise RuntimeError(f"HTTP Error {response.status}: {response.reason}")
        client_wall = time.monotonic() - started
        path = os.path.join(CLIPS, f"{attempt_id}.mp4")
        with open(path, "wb") as handle:
            handle.write(payload)
        record.update({"http_status": 200, "client_wall_s": round(client_wall, 3),
                       "finished_utc": utc(), "output_file": f"{attempt_id}.mp4",
                       "output_bytes": len(payload)})
    except Exception as error:
        record.update({"http_status": 0, "failure_reason": str(error)[:300],
                       "client_wall_s": round(time.monotonic() - started, 3),
                       "finished_utc": utc()})
    telemetry.append(record)


def dispatch_round(round_index, seed, prompt, negative, telemetry):
    threads = []
    for index, port in enumerate(PORTS):
        attempt_id = f"prod-r{round_index:02d}-r{index:02d}"
        thread = threading.Thread(target=one_request,
                                  args=(port, attempt_id, seed, prompt, negative, telemetry))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def gpu_sampler(stop_event, samples):
    while not stop_event.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = []
        for line in out.stdout.strip().splitlines():
            util, mem, power, clock = [part.strip() for part in line.split(",")]
            rows.append({"util_pct": util, "mem_mib": mem, "power_w": power, "sm_mhz": clock})
        samples.append({"utc": utc(), "gpus": rows})
        stop_event.wait(10)


def probe_clip(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
         "-of", "json", path], capture_output=True, text=True)
    try:
        stream = json.loads(out.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        return {"valid": False, "errors": ["ffprobe could not read stream"]}
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    errors = []
    if int(stream["width"]) != int(WIDTH): errors.append("width")
    if int(stream["height"]) != int(HEIGHT): errors.append("height")
    if int(stream["nb_read_frames"]) != int(FRAMES): errors.append("frames")
    if abs(fps - 24.0) > 0.01: errors.append("fps")
    return {"valid": not errors, "errors": errors, "stream": stream}


def main():
    os.makedirs(CLIPS, exist_ok=True)
    with open(PROMPT_FILE, encoding="utf-8") as handle:
        prompt = json.dumps(json.load(handle), ensure_ascii=True, separators=(",", ":"))
    with open(NEGATIVE_FILE, encoding="utf-8") as handle:
        negative = json.dumps(json.load(handle), ensure_ascii=True, separators=(",", ":"))

    warmups = []
    log("warmup round starting (8 replicas, seed 17)")
    dispatch_round(-1, 17, prompt, negative, warmups)
    for record in warmups:
        path = os.path.join(CLIPS, record["output_file"]) if record.get("output_file") else None
        record["video_validation"] = probe_clip(path) if path else {"valid": False}
        log(f"warmup {record['replica']}: http={record['http_status']} wall={record['client_wall_s']}s valid={record['video_validation']['valid']}")
    if any(not record["video_validation"]["valid"] for record in warmups):
        log("warmup failed validity gate; aborting before production window")
        json.dump({"status": "warmup_failed", "warmups": warmups}, open(os.path.join(RUN, "window.json"), "w"), indent=1)
        return 1

    records = []
    telemetry_samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=gpu_sampler, args=(stop_event, telemetry_samples), daemon=True)
    sampler.start()
    window_start = utc()
    started = time.monotonic()
    for round_index in range(ROUNDS):
        seed = SEEDS[round_index % len(SEEDS)]
        log(f"production round {round_index + 1}/{ROUNDS} starting (seed {seed})")
        dispatch_round(round_index, seed, prompt, negative, records)
        done = len(records)
        log(f"production round {round_index + 1}/{ROUNDS} done ({done}/{ROUNDS * len(PORTS)} attempts)")
    window_seconds = time.monotonic() - started
    stop_event.set()
    sampler.join(timeout=15)
    window_end = utc()

    for record in records:
        path = os.path.join(CLIPS, record["output_file"]) if record.get("output_file") else None
        record["video_validation"] = probe_clip(path) if path else {"valid": False}
    valid = [record for record in records if record["video_validation"]["valid"]]
    walls = sorted(record["client_wall_s"] for record in valid)
    clips_per_node_hour = len(valid) * 3600 / window_seconds
    video_seconds_per_node_hour = len(valid) * CLIP_SECONDS * 3600 / window_seconds
    derived = {
        "attempts": len(records), "valid_attempts": len(valid),
        "window_seconds": round(window_seconds, 3),
        "clips_per_node_hour": round(clips_per_node_hour, 1),
        "video_seconds_per_node_hour": round(video_seconds_per_node_hour, 1),
        "median_client_wall_s": round(statistics.median(walls), 3) if walls else None,
        "p95_client_wall_s": round(walls[max(0, int(round(0.95 * len(walls))) - 1)], 3) if walls else None,
        "max_client_wall_s": walls[-1] if walls else None,
        "expected_clip_video_seconds": CLIP_SECONDS,
    }
    utils = [float(row["util_pct"]) for sample in telemetry_samples for row in sample["gpus"]]
    mems = [float(row["mem_mib"]) for sample in telemetry_samples for row in sample["gpus"]]
    derived["gpu_util_mean_pct"] = round(sum(utils) / len(utils), 1) if utils else None
    derived["peak_mem_gib"] = round(max(mems) / 1024, 2) if mems else None

    with open(os.path.join(RUN, "records.jsonl"), "w") as handle:
        for record in records + warmups:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    window = {
        "protocol_version": "v1-p2-runlocal-r8", "comparison_basis": "nano_h200_N8_V30",
        "cell": CELL, "topology": "8x1", "shape": SHAPE,
        "prompt_sha256": subprocess.run(["sha256sum", PROMPT_FILE], capture_output=True, text=True).stdout.split()[0],
        "negative_sha256": subprocess.run(["sha256sum", NEGATIVE_FILE], capture_output=True, text=True).stdout.split()[0],
        "seeds": SEEDS, "replica_ports": PORTS,
        "dispatch_schedule": "synchronized_rounds", "concurrency": 8,
        "window_start_utc": window_start, "window_end_utc": window_end,
        "warmup_records": warmups, "telemetry_samples": telemetry_samples,
        "derived": derived,
    }
    with open(os.path.join(RUN, "window.json"), "w") as handle:
        json.dump(window, handle, indent=1)
    log(f"R8 complete: {json.dumps(derived)}")
    return 0 if len(valid) == len(records) and len(records) == ROUNDS * len(PORTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
