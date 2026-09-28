"""Exercise native MJLab capabilities on an operator-selected GPU and S3 prefix.

Run inside the built MJLab image. Full reports stay in the private evidence
directory; acceptance.json contains only hardware, hashes and measured results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import httpx

from npa.sdk.workbench import mjlab
from npa.workbench.mjlab.runtime import _storage_client

TASKS = (
    ("cartpole", "Mjlab-Cartpole-Balance"),
    ("g1", "Mjlab-Velocity-Flat-Unitree-G1"),
    ("go1", "Mjlab-Velocity-Flat-Unitree-Go1"),
    ("yam", "Mjlab-Lift-Cube-Yam"),
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("b200", "rtx6000"), required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    return parser.parse_args()


def _hardware(args):
    import torch

    assert torch.cuda.is_available(), "CUDA is required"
    assert torch.cuda.device_count() == args.gpu_count
    expected_name = "B200" if args.family == "b200" else "RTX PRO 6000"
    expected_capability = (10, 0) if args.family == "b200" else (12, 0)
    devices = []
    for index in range(args.gpu_count):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        assert expected_name in name and capability == expected_capability
        with torch.cuda.device(index):
            values = torch.arange(1024, device=f"cuda:{index}")
            assert torch.equal((values + values).cpu(), 2 * torch.arange(1024))
        devices.append({"name": name, "capability": list(capability)})
    return {
        "devices": devices,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "image_source_sha": os.environ.get("NPA_IMAGE_SOURCE_SHA", ""),
        "versions": mjlab.system_info()["versions"],
    }


def _readback(report, directory):
    directory.mkdir(parents=True)
    (directory / "private-report.json").write_text(json.dumps(report, indent=2))
    client = _storage_client()
    paths = {}
    for index, (name, artifact) in enumerate(report["artifacts"].items()):
        path = directory / str(index)
        client.download_file(artifact["uri"], str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]
        assert path.stat().st_size == artifact["bytes"]
        paths[name] = path
    manifest = directory / "private-manifest.json"
    client.download_file(report["result_uri"], str(manifest))
    assert json.loads(manifest.read_text()) == report
    return paths


def _checkpoint(path):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    actor = data["actor_state_dict"]
    assert all(torch.isfinite(value).all() for value in actor.values())
    states = data["optimizer_state_dict"]["state"].values()
    step = max(float(value["step"]) for value in states if "step" in value)
    assert step > 0
    return {"actor": actor, "iteration": int(data["iter"]), "optimizer_step": step}


def _assert_resumed(before, after):
    import torch

    assert after["iteration"] > before["iteration"]
    assert after["optimizer_step"] > before["optimizer_step"]
    assert any(
        not torch.equal(value, after["actor"][key])
        for key, value in before["actor"].items()
    ), "Resumed training did not update the actor"


def _training(args, name, task, *, gpu_count=1, use_cli=False):
    request = mjlab.TrainRequest(
        task=task,
        output_path=args.output_path.rstrip("/") + "/" + name + "/train",
        iterations=2,
        num_envs=64,
        gpu_count=gpu_count,
    )
    print(f"Training {name}: {task}, {gpu_count} GPU(s)", flush=True)
    if use_cli:
        command = [sys.executable, "-m", "npa.cli.main", "workbench", "mjlab", "train"]
        for key in ("task", "output_path", "iterations", "num_envs", "gpu_count"):
            command.extend(["--" + key.replace("_", "-"), str(getattr(request, key))])
        result = subprocess.run(command, capture_output=True, text=True)
        (args.evidence_dir / f"{name}-cli.log").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
    else:
        report = mjlab.train(request)
    paths = _readback(report, args.evidence_dir / name / "train")
    state = _checkpoint(paths["checkpoint.pt"])
    return report, state


def _evaluation(args, name, training, *, video=False, connection=None):
    request = mjlab.EvalRequest(
        task=training["task"],
        checkpoint=training["artifacts"]["checkpoint.pt"]["uri"],
        output_path=args.output_path.rstrip("/") + "/" + name + "/eval",
        num_envs=4,
        episodes=8,
        video=video,
    )
    report = mjlab.eval(request, **(connection or {}))
    paths = _readback(report, args.evidence_dir / name / "eval")
    assert report["episodes_completed"] == 8
    assert all(row["length"] > 0 for row in report["episodes"])
    assert (
        report["input_sha256"]["checkpoint"]
        == training["artifacts"]["checkpoint.pt"]["sha256"]
    )
    assert json.loads(paths["episodes.json"].read_text()) == report["episodes"]
    if video:
        from npa.workbench.mjlab.worker import _verify_video

        assert _verify_video(paths["rollout.mp4"]) == report["video_frames"]
    return report


def _export(args, name, training, *, connection=None):
    import onnx

    request = mjlab.ExportRequest(
        task=training["task"],
        checkpoint=training["artifacts"]["checkpoint.pt"]["uri"],
        output_path=args.output_path.rstrip("/") + "/" + name + "/export",
    )
    report = mjlab.export(request, **(connection or {}))
    paths = _readback(report, args.evidence_dir / name / "export")
    model = onnx.load(str(paths["policy.onnx"]))
    onnx.checker.check_model(model)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata["task"] == training["task"]
    assert metadata["mjlab_version"] == "1.6.0"
    return report


def _cycle(args, name, task, *, gpu_count=1, use_cli=False, video=False):
    training, state = _training(args, name, task, gpu_count=gpu_count, use_cli=use_cli)
    evaluated = _evaluation(args, name, training, video=video)
    exported = _export(args, name, training)
    summary = {
        "task": task,
        "gpu_count": gpu_count,
        "checkpoint_sha256": training["artifacts"]["checkpoint.pt"]["sha256"],
        "iteration": state["iteration"],
        "optimizer_step": state["optimizer_step"],
        "episodes_completed": evaluated["episodes_completed"],
        "mean_return": evaluated["mean_return"],
        "survival_fraction": evaluated["score"],
        "video_frames": evaluated.get("video_frames"),
        "onnx_sha256": exported["artifacts"]["policy.onnx"]["sha256"],
        "artifact_hash_readbacks": sum(
            len(r["artifacts"]) for r in (training, evaluated, exported)
        ),
    }
    return summary, training, state


@contextmanager
def _service(args):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    token = secrets.token_urlsafe(32)
    env = dict(os.environ, MJLAB_TOKEN=token, MJLAB_ALLOWED_S3_ROOTS=args.output_path)
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "npa.workbench.mjlab.service:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with (args.evidence_dir / "private-service.log").open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        os.environ["NPA_MJLAB_ACCEPTANCE_TOKEN"] = token
        try:
            while process.poll() is None:
                try:
                    if httpx.get(endpoint + "/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            assert process.poll() is None, "Service startup failed"
            yield (
                {"endpoint": endpoint, "token_env": "NPA_MJLAB_ACCEPTANCE_TOKEN"},
                token,
            )
        finally:
            process.terminate()
            process.wait()
            os.environ.pop("NPA_MJLAB_ACCEPTANCE_TOKEN", None)


def _concurrent_resume(request, connection, token):
    with ThreadPoolExecutor() as pool:
        future = pool.submit(mjlab.train, request, **connection)
        while not future.done() and not mjlab.status(**connection)["busy"]:
            time.sleep(0.05)
        assert not future.done(), (
            "Training finished before concurrency could be checked"
        )
        busy = httpx.post(
            connection["endpoint"] + "/train",
            json=request.model_dump(),
            headers={"Authorization": "Bearer " + token},
        )
        assert busy.status_code == 409
        return future.result()


def _service_checks(args, training, before):
    with _service(args) as (connection, token):
        endpoint = connection["endpoint"]
        assert httpx.get(endpoint + "/status").status_code == 401
        assert mjlab.status(**connection)["installed"]
        assert training["task"] in mjlab.list(**connection)["tasks"]
        invalid = mjlab.EvalRequest(
            checkpoint="s3://outside-scope/checkpoint.pt",
            output_path=args.output_path + "/rejected",
        )
        denied = httpx.post(
            endpoint + "/eval",
            json=invalid.model_dump(),
            headers={"Authorization": "Bearer " + token},
        )
        assert denied.status_code == 400
        request = mjlab.TrainRequest(
            task=training["task"],
            checkpoint=training["artifacts"]["checkpoint.pt"]["uri"],
            output_path=args.output_path + "/service-resume/train",
            iterations=2,
            num_envs=64,
        )
        resumed = _concurrent_resume(request, connection, token)
        paths = _readback(resumed, args.evidence_dir / "service-resume" / "train")
        after = _checkpoint(paths["checkpoint.pt"])
        _assert_resumed(before, after)
        evaluated = _evaluation(args, "service-resume", resumed, connection=connection)
        _export(args, "service-resume", resumed, connection=connection)
    return {
        "unauthenticated_status": 401,
        "out_of_scope_status": 400,
        "concurrent_status": 409,
        "resumed_iteration": after["iteration"],
        "resumed_optimizer_step": after["optimizer_step"],
        "episodes_completed": evaluated["episodes_completed"],
    }


def main():
    """Run native training/resume, measured evaluation, ONNX, API and GPU checks.

    Args:
        None; configuration comes from command-line options.
    Returns:
        None; writes a sanitized acceptance report and private detailed evidence.
    Raises:
        AssertionError: Any hardware, capability, artifact or isolation check fails.
    """
    args = _arguments()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "npa.mjlab.gpu-acceptance.v1",
        "passed": False,
        "cases": {},
    }
    try:
        report["hardware"] = _hardware(args)
        for name, task in TASKS:
            result, training, state = _cycle(
                args,
                name,
                task,
                use_cli=name == "g1",
                video=args.family == "rtx6000" and name == "g1",
            )
            report["cases"][name] = result
            if name == "cartpole":
                report["service"] = _service_checks(args, training, state)
        if args.gpu_count > 1:
            result, _, _ = _cycle(
                args, "multi-gpu", "Mjlab-Cartpole-Balance", gpu_count=args.gpu_count
            )
            report["cases"]["multi-gpu"] = result
        report["passed"] = True
    finally:
        (args.evidence_dir / "acceptance.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
