"""Reconcile both GPU workers and build standalone visual proof from the same workflow run."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import shutil
import subprocess

from .contracts import digest
from .public_vla_data import write_json
from .public_vla_export import file_sha256
from .turnkey_store import inherit, materialize, read, record, require_identity


PUBLIC_HISTORY_KEYS = {
    "stage",
    "engine",
    "phase",
    "iteration",
    "runtime",
    "resumed",
    "training_kind",
    "checkpoint_sha256",
    "input_checkpoint_sha256",
    "recipe_sha256",
    "corpus_sha256",
    "input_count",
    "selected_count",
    "scope",
    "partitions",
    "episodes",
    "frames",
    "successes",
    "trials",
    "success_rate",
    "decision",
    "minimum_success",
    "initial_state_offset",
    "reserved_demonstrations",
    "rank",
    "workers",
}


def report(args, workspace: Path, output: Path) -> None:
    """Reconcile actual served actions and embed every deployment rollout in offline HTML.

    Args:
        args: Two-worker serving prefix and report destination.
        workspace: Private CPU workspace.
        output: Private artifact directory; demo.html, demo.mp4 and proof.json are public-safe.
    Returns:
        None.
    Raises:
        ValueError: Worker identity, actions or decoded videos differ.
        subprocess.CalledProcessError: Native media cannot be decoded or joined.
    """
    server = materialize(args.input_uri.rstrip("/") + "/server/", workspace / "server")
    client = materialize(args.input_uri.rstrip("/") + "/client/", workspace / "client")
    require_identity(server, read(client, "recipe.json"), read(client, "corpus.json"))
    summary = read(client, "summary.json")
    steps = reconcile(server, client, summary)
    videos = _videos(client, summary)
    _montage(videos, output)
    proof = _proof(client, summary, steps, videos)
    write_json(output / "proof.json", proof)
    _html(client, output, proof, steps)
    inherit(client, output)
    record(
        output,
        "report",
        {
            "engine": "measured-offline-html",
            "qualified": proof["qualified"],
            "actions_reconciled": len(steps),
            "checkpoint_sha256": proof["checkpoint_sha256"],
        },
    )
    shutil.rmtree(output / "previews", ignore_errors=True)


def reconcile(server: Path, client: Path, summary: dict) -> list[dict]:
    """Require exact server/client agreement for every action actually applied to physics.

    Args:
        server: Verified inference-worker artifacts.
        client: Verified simulation-worker artifacts.
        summary: Native benchmark summary.
    Returns:
        Verified applied action records.
    Raises:
        ValueError: Checkpoint or request/action identities differ.
    """
    runtime = read(server, "runtime.json")
    if runtime["checkpoint_sha256"] != summary["model"]["checkpoint_sha256"]:
        raise ValueError("benchmark used a different server checkpoint")
    served = [
        json.loads(line)
        for line in (server / "requests.jsonl").read_text().splitlines()
    ]
    applied = [
        json.loads(line) for line in (client / "steps.jsonl").read_text().splitlines()
    ]
    if len(served) != len(applied) or len(applied) != summary["steps"] or not applied:
        raise ValueError("served and applied action counts differ")
    keys = ("episode", "step", "action", "server_ms", "new_action_chunk")
    for request, action in zip(served, applied, strict=True):
        if any(request[key] != action[key] for key in keys):
            raise ValueError("applied actions differ from the server request ledger")
        if len(action["action"]) != 7 or not all(
            math.isfinite(x) for x in action["action"]
        ):
            raise ValueError("invalid applied action vector")
    if sum(bool(row["success"]) for row in summary["episodes"]) != summary["successes"]:
        raise ValueError("native episode predicates disagree with summary")
    return applied


def _videos(client, summary):
    videos = []
    for index, episode in enumerate(summary["episodes"]):
        if (
            episode["episode"] != index
            or episode["video"] != f"episode-{index:02d}.mp4"
        ):
            raise ValueError("unexpected episode identity or video path")
        video = client / episode["video"]
        if file_sha256(video) != episode["video_sha256"]:
            raise ValueError("native episode video digest differs")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        info = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        duration = float(json.loads(info.stdout)["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("native rollout video is empty")
        videos.append(
            {"path": video, "duration": duration, "sha256": episode["video_sha256"]}
        )
    return videos


def _montage(videos, output):
    listing = output / "concat.txt"
    listing.write_text("".join("file '" + str(v["path"]) + "'\n" for v in videos))
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output / "demo.mp4"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    listing.unlink()


def _proof(client, summary, steps, videos):
    recipe = read(client, "recipe.json")
    corpus = read(client, "corpus.json")
    history = _public_history(client)
    gates = [r for r in history if r["stage"].endswith("-gate")]
    qualified = all(
        any(
            r["phase"] == phase and r["decision"] == "promote_checkpoint" for r in gates
        )
        for phase in ("pretrain", "finetune")
    )
    qualified = (
        qualified
        and summary["successes"] / summary["trials"] >= recipe["minimum_success"]
    )
    return {
        "schema": "npa.policy-public.proof.v1",
        "run_id": recipe["run_id"],
        "workflow_sha256": recipe["workflow_sha256"],
        "recipe_sha256": digest(recipe),
        "corpus_sha256": digest(corpus),
        "checkpoint_sha256": summary["model"]["checkpoint_sha256"],
        "model": summary["model"],
        "summary": summary,
        "history": history,
        "renderer": read(client, "renderer.json"),
        "qualified": qualified,
        "all_actions_reconciled": True,
        "public_data_only": True,
        "slurm_used": False,
        "physical_robot_tested": False,
        "video": "demo.mp4",
        "episode_durations": [v["duration"] for v in videos],
        "video_checksums": [v["sha256"] for v in videos],
        "visualization_source": "native NVIDIA-rendered simulation driven by served neural actions",
    }


def _public_history(client):
    result = []
    for path in sorted((client / "history").glob("*.json")):
        value = read(client / "history", path.name)
        if "stage" not in value:
            continue
        row = {key: value[key] for key in PUBLIC_HISTORY_KEYS if key in value}
        if "selection" in value:
            row["training_episodes"] = len(value["selection"]["episodes"])
            row["training_frames"] = value["selection"]["frames"]
        result.append(row)
    return result


def _html(client, output, proof, steps):
    template = Path(__file__).with_name("turnkey_report.html").read_text()
    metrics = {}
    for path in (client / "history").glob("*-metrics.json"):
        rows = read(client / "history", path.name)["metrics"]
        metrics[path.stem] = [
            {key: r[key] for key in ("step", "loss", "lr") if key in r} for r in rows
        ]
    timeline = [
        {
            key: row[key]
            for key in (
                "episode",
                "step",
                "action",
                "server_ms",
                "round_trip_ms",
                "new_action_chunk",
            )
        }
        for row in steps
    ]
    images = {
        path.stem: "data:image/png;base64,"
        + base64.b64encode(path.read_bytes()).decode()
        for path in client.glob("episode-*-observation.png")
    }
    payload = {"proof": proof, "metrics": metrics, "steps": timeline, "images": images}
    source = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = template.replace("__EVIDENCE_JSON__", source)
    template = template.replace(
        "__VIDEO_BASE64__",
        base64.b64encode((output / "demo.mp4").read_bytes()).decode(),
    )
    (output / "demo.html").write_text(template)
