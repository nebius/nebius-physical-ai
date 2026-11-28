"""Exercise native WAM launch topology, malformed measurements, and Slurm propagation."""

import importlib.util
from datetime import datetime, timezone
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"wam_{name}", RECIPE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "fault", [None, "model", "referenced_empty", "missing", "truncated", "no_records"]
)
def test_native_dcp_scheduler_writer_placeholders(tmp_path, monkeypatch, fault):
    monkeypatch.syspath_prepend(str(RECIPE))
    smoke = _load("smoke_report")
    name = {"referenced_empty": "__1_0.distcp", "missing": "__7_0.distcp"}.get(
        fault, "__0_0.distcp"
    )
    extent = SimpleNamespace(
        relative_path=name, offset=0, length=9 if fault == "truncated" else 8
    )
    storage = {} if fault == "no_records" else {0: extent}
    reader = SimpleNamespace(
        read_metadata=lambda: SimpleNamespace(storage_data=storage)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.checkpoint",
        SimpleNamespace(FileSystemReader=lambda _: reader),
    )
    files = {
        "scheduler/__0_0.distcp": {"bytes": 8},
        "scheduler/__1_0.distcp": {"bytes": 0},
        "model/__0_0.distcp": {"bytes": 0 if fault == "model" else 8},
    }
    if fault:
        with pytest.raises(ValueError):
            smoke._check_empty_shards(tmp_path, files)
    else:
        smoke._check_empty_shards(tmp_path, files)


def _args(tmp_path, nodes=1):
    return (
        _load("recipe")
        ._parser()
        .parse_args(
            [
                "plan",
                "--shared-root",
                str(tmp_path / "shared"),
                "--run-dir",
                str(tmp_path / "run"),
                "--name",
                "baseline",
                "--nodes",
                str(nodes),
                "--steps",
                "55",
            ]
        )
    )


@pytest.mark.parametrize("nodes,accumulation", [(1, 4), (2, 2), (4, 1)])
def test_fixed_global_batch_and_native_wam_config(tmp_path, nodes, accumulation):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    recipe = _load("recipe")
    settings = recipe._settings(_args(tmp_path, nodes))
    config = tomllib.loads(recipe._toml(settings))
    assert settings["gpus"] * settings["samples_per_rank"] * accumulation == 2048
    assert config["trainer"]["grad_accum_iter"] == accumulation
    assert config["model"]["parallelism"]["data_parallel_replicate_degree"] == nodes
    assert config["job"]["experiment"] == "action_policy_libero_nano"
    assert config["optimizer"]["lr"] == 5e-5


@pytest.mark.parametrize(
    "field,value",
    [("nodes", 3), ("nodes", 0), ("steps", -1), ("name", "bad\n#SBATCH --nodes=9")],
)
def test_invalid_plans_fail_before_writing(tmp_path, field, value):
    recipe = _load("recipe")
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        recipe._plan(args)
    assert not args.run_dir.exists()


def test_multinode_torchrun_uses_slurm_rank_and_shared_endpoint(tmp_path, monkeypatch):
    recipe = _load("recipe")
    settings = recipe._settings(_args(tmp_path, 2))
    monkeypatch.setenv("MASTER_ADDR", "rank-zero")
    monkeypatch.setenv("MASTER_PORT", "29507")
    argv = recipe._training_argv(settings, tmp_path, 1)
    assert "--nnodes=2" in argv and "--node_rank=1" in argv
    assert "--master_addr=rank-zero" in argv and "--master_port=29507" in argv
    assert "--standalone" not in argv
    assert "cosmos_framework.scripts.train" in argv


@pytest.mark.parametrize("destination", ["fifo", "regular", "unset"])
def test_sequential_nccl_launches_do_not_reuse_slurm_fifo(
    tmp_path, monkeypatch, destination
):
    recipe = _load("recipe")
    settings = recipe._settings(_args(tmp_path, 4))
    log = tmp_path / "operator-nccl.log"
    monkeypatch.setenv("SLURM_NODEID", "2")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    monkeypatch.delenv("NCCL_DEBUG_FILE", raising=False)
    if destination != "unset":
        monkeypatch.setenv("NCCL_DEBUG_FILE", str(log))
        if destination == "fifo":
            os.mkfifo(log)
        else:
            log.touch()
    env = recipe._environment(settings, tmp_path)
    expected = {
        "fifo": str(tmp_path / "nccl-node-2-%p.log"),
        "regular": str(log),
        "unset": None,
    }
    assert env.get("NCCL_DEBUG_FILE") == expected[destination]
    assert env["NCCL_DEBUG"] == "INFO" and env["NCCL_IB_DISABLE"] == "0"


def _camera_inputs(root):
    paths = []
    for camera in ("observation.images.image", "observation.images.wrist_image"):
        path = root / "data/libero_10/videos" / camera / "chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
        paths.append(path)
    return paths


@pytest.mark.parametrize("shape", [(3, 256, 256), (3, 128, 128)])
def test_camera_runtime_checks_both_views_and_frame_dimensions(
    tmp_path, monkeypatch, shape
):
    recipe = _load("recipe")
    paths = _camera_inputs(tmp_path)
    calls = []

    def decoder(path, *, device):
        calls.append((Path(path), device))
        return {0: SimpleNamespace(shape=shape)}

    monkeypatch.setitem(
        sys.modules, "torchcodec.decoders", SimpleNamespace(VideoDecoder=decoder)
    )
    if shape != (3, 256, 256):
        with pytest.raises(ValueError, match="incompatible decoded camera frame"):
            recipe._verify_video_runtime(tmp_path)
    else:
        recipe._verify_video_runtime(tmp_path)
        assert calls == [(path, "cpu") for path in paths]


def _prepared_video_run(recipe, args):
    recipe._plan(args)
    settings = json.loads((args.run_dir / "run.json").read_text())
    _camera_inputs(args.shared_root)
    (args.shared_root / "prepared.json").write_text(
        json.dumps({"sources": settings["sources"]})
    )
    for name in (
        "base-dcp/model/.metadata",
        "vae/Wan2.2_VAE.pth",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "data/libero_10/meta/info.json",
    ):
        path = args.shared_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return settings


def test_missing_decoder_runtime_stops_before_gpu_or_distributed_launch(
    tmp_path, monkeypatch
):
    recipe = _load("recipe")
    args = _args(tmp_path, nodes=2)
    settings = _prepared_video_run(recipe, args)
    monkeypatch.setattr(
        recipe.subprocess,
        "check_output",
        lambda argv, **_: (
            settings["sources"]["framework"] if "rev-parse" in argv else ""
        ),
    )
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setattr(
        recipe, "_verify_gpus", lambda: pytest.fail("GPU startup must not run")
    )

    def unavailable_decoder(*_, **__):
        raise RuntimeError("FFmpeg shared libraries are missing")

    monkeypatch.setitem(
        sys.modules,
        "torchcodec.decoders",
        SimpleNamespace(VideoDecoder=unavailable_decoder),
    )
    with pytest.raises(RuntimeError, match="FFmpeg shared libraries"):
        recipe._run_node(args)
    assert not (args.run_dir / "node-0.started.json").exists()


def test_slurm_script_preserves_argv_and_worker_failure(tmp_path, monkeypatch):
    recipe = _load("recipe")
    args = _args(tmp_path, 2)
    args.shared_root = tmp_path / "shared with spaces"
    recipe._plan(args)
    binary = args.shared_root / "framework/.venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_FILE"\nexit 19\n')
    binary.chmod(0o700)
    for name, body in {
        "scontrol": 'printf "rank-zero\\nrank-one\\n"',
        "srun": (
            'test -z "${SLURM_TRES_PER_TASK:-}" || exit 91\n'
            'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"'
        ),
    }.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "workers")
    monkeypatch.setenv("SLURM_TRES_PER_TASK", "cpu:128")
    monkeypatch.setenv("ARGS_FILE", str(tmp_path / "argv"))
    result = subprocess.run(["bash", str(args.run_dir / "train.sbatch")])
    assert result.returncode == 19
    assert (tmp_path / "argv").read_text().splitlines() == [
        str(args.run_dir / "recipe.py"),
        "run-node",
        "--run-dir",
        str(args.run_dir),
    ]
    with pytest.raises(FileExistsError):
        recipe._plan(args)


def _evaluation_summaries():
    return [
        {
            "task_results": [
                {
                    "task_id": task,
                    "episodes": 50,
                    "successes": 45,
                    "episode_results": [
                        {"episode": episode, "success": episode < 45, "error": None}
                        for episode in range(50)
                    ],
                }
                for task in range(10)
            ]
        }
    ]


def test_quality_counts_full_trials_and_rejects_server_errors():
    evaluation = _load("evaluate")
    summaries = _evaluation_summaries()
    tasks = evaluation._task_results(summaries, 50)
    assert sum(task["successes"] for task in tasks) == 450
    low, high = evaluation._wilson(450, 500)
    assert 0.86 < low < 0.9 < high < 0.93
    tasks[0]["episode_results"][0]["error"] = "server error: connection reset"
    with pytest.raises(ValueError, match="infrastructure or inference errors"):
        evaluation._task_results(summaries, 50)


def _quality_curve(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    return _load("quality_curve")


def test_checkpoint_timestamps_handle_year_rollover_and_require_complete_set(
    monkeypatch,
):
    curve = _quality_curve(monkeypatch)
    started = datetime(2026, 12, 31, 23, 59, 30, tzinfo=timezone.utc).timestamp()
    node = {"ended_unix": started + 90, "train_process_seconds": 90}
    rows = [
        f"[01-01 00:00:{second:02d}|INFO|dcp.py:1201:save_state_dict_worker] "
        f"Saved checkpoint to /fixture/checkpoints/iter_{step:09d}\n"
        for second, step in zip((1, 11, 21, 31), (500, 1000, 1500, 2000))
    ]
    timings = curve._checkpoint_times("".join(rows), node)
    assert timings[500] == [31, 32]
    assert timings[2000] == [61, 62]
    with pytest.raises(ValueError, match="all four"):
        curve._checkpoint_times("".join(rows[:-1]), node)
    with pytest.raises(ValueError, match="duplicate"):
        curve._checkpoint_times("".join(rows + rows[:1]), node)


def _quality_receipt(step):
    return {
        **_evaluation_summaries()[0],
        "schema": "npa.cosmos3.wam-quality.v1",
        "step": step,
        "full_500_trial_evaluation": True,
        "successes": 450,
        "success_rate": 0.9,
        "trials": 500,
        "threshold": 0.9,
        "threshold_met": True,
        "elapsed_seconds": 100,
        "model_manifest_sha256": "incorrect-checkpoint-digest",
    }


def test_quality_curve_rejects_wrong_checkpoint_even_with_complete_trials(
    tmp_path, monkeypatch
):
    curve = _quality_curve(monkeypatch)
    run = _completed_run(tmp_path)
    settings = json.loads((run / "run.json").read_text())
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    observed = {
        "run_dir": str(run),
        "seed": settings["seed"],
        "trials": 50,
        "record_rollouts": False,
        "workers": 8,
        "envs": 8,
        "step": 55,
    }
    (evaluation / "settings.json").write_text(json.dumps(observed))
    (evaluation / "quality.json").write_text(json.dumps(_quality_receipt(55)))
    with pytest.raises(ValueError, match="model hash"):
        curve._point(evaluation, run, settings, {55: [60, 61]})
    observed["run_dir"] = str(tmp_path / "different-run")
    (evaluation / "settings.json").write_text(json.dumps(observed))
    with pytest.raises(ValueError, match="different training run"):
        curve._point(evaluation, run, settings, {55: [60, 61]})


def test_quality_curve_rejects_inconsistent_threshold_claim(monkeypatch):
    curve = _quality_curve(monkeypatch)
    quality = _quality_receipt(500)
    assert curve._quality_counts(quality, {"step": 500})[1] == 450
    quality["threshold_met"] = False
    with pytest.raises(ValueError, match="individual trials"):
        curve._quality_counts(quality, {"step": 500})


@pytest.mark.parametrize("changed", ["workers", "envs"])
def test_quality_curve_rejects_changed_evaluation_parallelism(monkeypatch, changed):
    curve = _quality_curve(monkeypatch)
    contract = {"workers": 8, "envs": 8, "trials": 50, "seed": 42}
    points = [{"evaluation_contract": dict(contract)} for _ in range(4)]
    assert curve._matching_evaluation_contract(points) == contract
    points[2]["evaluation_contract"][changed] = 1
    with pytest.raises(ValueError, match="evaluation workers, environments"):
        curve._matching_evaluation_contract(points)


def test_time_to_quality_reports_first_observed_checkpoint_and_no_crossing(monkeypatch):
    curve = _quality_curve(monkeypatch)
    points = [
        {
            "step": step,
            "threshold_met": step >= 1500,
            "checkpoint_ready_train_seconds_interval": [step * 10, step * 10 + 1],
        }
        for step in (500, 1000, 1500, 2000)
    ]
    assert curve._first_passing(points) == {
        "status": "observed_at_scheduled_checkpoint",
        "step": 1500,
        "train_seconds_interval": [15000, 15001],
        "preceding_evaluated_step": 1000,
    }
    for point in points:
        point["threshold_met"] = False
    assert curve._first_passing(points)["step"] is None


@pytest.mark.parametrize("missing", ["task", "trial", "duplicate", "aggregate"])
def test_quality_rejects_incomplete_or_inconsistent_results(missing):
    evaluation = _load("evaluate")
    summaries = _evaluation_summaries()
    tasks = summaries[0]["task_results"]
    if missing == "task":
        tasks.pop()
    elif missing == "trial":
        tasks[0]["episode_results"].pop()
    elif missing == "duplicate":
        tasks[0]["episode_results"][0]["episode"] = 1
    else:
        tasks[0]["successes"] = 50
    with pytest.raises(ValueError):
        evaluation._task_results(summaries, 50)


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "8.8.8.8", "::1"])
def test_worker_join_rejects_nonprivate_peer_routes(address):
    with pytest.raises(ValueError, match="RFC1918"):
        _load("add_worker")._private_address(address)


def test_generated_slurm_config_remains_readable_by_job_users(tmp_path, monkeypatch):
    controller = _load("slurm_controller")
    original_path = Path

    def local_config_path(*parts):
        path = original_path(*parts)
        if str(path).startswith("/etc/slurm/"):
            return tmp_path / path.name
        return path

    monkeypatch.setattr(controller, "Path", local_config_path)
    monkeypatch.setattr(
        controller,
        "_node_settings",
        lambda: {
            "NODE": "synthetic-worker",
            "ADDRESS": "private-worker-address",
            "SHAPE": "CPUs=160",
            "MEMORY": "1700000",
        },
    )
    controller._configure("synthetic-cluster")
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o644
    assert "gres/gpu" in (tmp_path / "slurm.conf").read_text()


def test_profile_busy_time_does_not_double_count_overlapping_collectives():
    profiler = _load("profile_report")
    kernels = [
        {"name": "gemm", "ts": 0, "dur": 10_000_000, "args": {"device": 0}},
        {
            "name": "ncclAllReduce",
            "ts": 5_000_000,
            "dur": 10_000_000,
            "args": {"device": 0},
        },
    ]
    result = profiler._kernel_metrics(kernels, 0, 12_000_000)
    assert result["kernel_duration_seconds_by_category"] == {
        "matrix_multiply": 10,
        "collectives": 7,
    }
    assert result["observed_kernel_busy_seconds_by_device"] == {"0": 12}


@pytest.mark.parametrize(
    "operator,category",
    [
        ("aten::mm", "matrix_multiply"),
        ("aten::cudnn_convolution", "convolution"),
        ("aten::copy_", "other"),
    ],
)
def test_profile_links_blackwell_kernel_to_its_actual_operator(operator, category):
    profiler = _load("profile_report")
    events = [
        {"cat": "cpu_op", "name": operator, "args": {"External id": 17}},
        {
            "cat": "kernel",
            "name": "nvjet_sm100_tst_128x256",
            "ts": 0,
            "dur": 1_000_000,
            "args": {"External id": 17, "device": 0},
        },
    ]
    kernels = profiler._linked_kernels(events)
    result = profiler._kernel_metrics(kernels, 0, 1_000_000)
    assert result["kernel_duration_seconds_by_category"] == {category: 1}
    assert result["cpu_operator_linked_kernel_count"] == 1
    assert result["total_kernel_count"] == 1


def test_profile_does_not_guess_ambiguous_operator_links():
    profiler = _load("profile_report")
    events = [
        {"cat": "cpu_op", "name": name, "args": {"External id": 17}}
        for name in ("aten::mm", "aten::cudnn_convolution", "aten::mm")
    ]
    events.append({"cat": "kernel", "name": "nvjet_sm100", "args": {"External id": 17}})
    kernel = profiler._linked_kernels(events)[0]
    assert kernel["cpu_operator"] is None
    assert profiler._kernel_category(kernel) == "other"


@pytest.mark.parametrize(
    "operator,category",
    [
        ("natten::blackwell_fmha_forward", "attention"),
        ("triton_poi_fused_addmm_blackwell_fmha_forward_squeeze_t_view_8", "other"),
    ],
)
def test_profile_distinguishes_attention_operators_from_fused_pointwise_work(
    operator, category
):
    event = {"name": "opaque_device_kernel", "cpu_operator": operator}
    assert _load("profile_report")._kernel_category(event) == category


def _profile_trace(path, host_steps):
    events = []
    for index in range(host_steps):
        for category, offset in (("user_annotation", 0), ("gpu_user_annotation", 10)):
            events.append(
                {
                    "ph": "X",
                    "cat": category,
                    "name": f"ProfilerStep#{index}",
                    "ts": index * 100 + offset,
                    "dur": 100 - offset,
                }
            )
    events.append(
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm",
            "ts": 10,
            "dur": 10,
            "args": {"device": 0},
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({"traceEvents": events}).encode()))


def test_profile_gpu_annotation_cannot_masquerade_as_a_second_step(tmp_path):
    path = tmp_path / "trace.json.gz"
    _profile_trace(path, 1)
    with pytest.raises(ValueError, match="two distinct host steps"):
        _load("profile_report")._trace(path)


def test_profile_uses_host_boundaries_without_duplicate_gpu_markers(tmp_path):
    path = tmp_path / "trace.json.gz"
    _profile_trace(path, 2)
    steps, _, start, stop = _load("profile_report")._trace(path)
    assert [event["name"] for event in steps] == ["ProfilerStep#0", "ProfilerStep#1"]
    assert (start, stop) == (0, 200)


def test_profile_can_reanalyze_without_overwriting_the_recorded_summary(tmp_path):
    from argparse import Namespace

    run = tmp_path / "run"
    trace = (
        run
        / "output/cosmos3_wam/libero_10/baseline/torch_trace/iteration_100/rank0_trace.json.gz"
    )
    _profile_trace(trace, 2)
    (run / "run.json").write_text(
        json.dumps({"profile": True, "nodes": 1, "name": "baseline"})
    )
    (run / "node-0.finished.json").write_text(json.dumps({"returncode": 0}))
    original = run / "profile-summary.json"
    original.write_text("original archived report\n")
    output = tmp_path / "reanalyzed.json"
    _load("profile_report")._main(Namespace(run_dir=run, output_path=output))
    assert original.read_text() == "original archived report\n"
    assert json.loads(output.read_text())["ranks"]["0"]["profiler_steps"] == [
        "ProfilerStep#0",
        "ProfilerStep#1",
    ]


def _completed_nodes(run, nodes, gpus_per_node=8):
    for rank in range(nodes):
        (run / f"node-{rank}.log").write_text("Worker log\n")
        (run / f"node-{rank}.finished.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "run_sha256": hashlib.sha256(
                        (run / "run.json").read_bytes()
                    ).hexdigest(),
                    "returncode": 0,
                    "train_process_seconds": 200,
                    "hardware": {
                        "gpu_names": ["NVIDIA B200"] * gpus_per_node,
                        "torch": "fixture",
                        "cuda": "fixture",
                    },
                }
            )
        )
    (run / "distributed-preflight.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "backend": "nccl",
                "world_size": gpus_per_node * nodes,
                "hosts": nodes,
            }
        )
    )


def _completed_run(tmp_path, nodes=1, gpus_per_node=8):
    recipe = _load("recipe")
    args = _args(tmp_path, nodes)
    args.gpus_per_node = gpus_per_node
    recipe._plan(args)
    run = args.run_dir
    _completed_nodes(run, nodes, gpus_per_node)
    (run / "node-0.log").write_text(
        "\n".join(
            f"{step} : iter_speed {seconds} seconds per iteration | Loss: 0.2"
            " | 1,000 tokens per iteration (200 tokens/s)"
            " | vae_encode 0.2s/iter avg (4.0%), max 0.3s (6.0%)"
            " | prepare_data 0.4s/iter avg (8.0%), max 0.5s (10.0%)"
            for step, seconds in zip(range(52, 56), [2, 4, 6, 8], strict=True)
        )
    )
    job = run / "output/cosmos3_wam/libero_10/baseline"
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = job / "checkpoints/iter_000000055" / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_bytes(b"fixture metadata")
        (directory / "rank.distcp").write_bytes(b"fixture checkpoint")
    (job / "config.yaml").write_text("fixture: true\n")
    return run


def test_report_measures_complete_run_and_never_claims_quality(tmp_path):
    run = _completed_run(tmp_path)
    report = _load("report")._summarize(run, 50)
    assert report["step_mean_seconds"] == 5
    assert report["step_p50_seconds"] == 5
    assert report["step_p95_seconds"] == pytest.approx(7.7)
    assert report["training_gpu_hours"] == pytest.approx(8 * 200 / 3600)
    assert report["quality_measured"] is False
    assert report["work"]["measured_tokens"] == 4000
    assert report["work"]["tokens_per_second"] == 200


def test_public_iteration_series_reproduces_aggregate_measurements(tmp_path):
    import csv

    run = _completed_run(tmp_path)
    module = _load("report")
    report = module._summarize(run, 52)
    module._write_series(run, report)
    path = run / "iteration-series.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == [53, 54, 55]
    seconds = sum(float(row["iteration_seconds"]) for row in rows)
    tokens = sum(int(row["tokens"]) for row in rows)
    assert seconds / len(rows) == report["step_mean_seconds"]
    assert tokens == report["work"]["measured_tokens"]
    assert tokens / seconds == report["work"]["tokens_per_second"]
    assert (
        report["iteration_series_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_report_rejects_background_failure_on_a_nonprimary_node(tmp_path):
    run = _completed_run(tmp_path, 2)
    (run / "node-1.log").write_text(
        "Exception in thread feeder:\nTraceback (most recent call last):\n"
        "RuntimeError: shared-memory cleanup failure\n"
    )
    with pytest.raises(ValueError, match="Python failure"):
        _load("report")._summarize(run, 50)


@pytest.mark.parametrize(
    "mutation", ["missing_node", "failed_node", "checkpoint", "nccl"]
)
def test_incomplete_run_cannot_be_reported(tmp_path, mutation):
    run = _completed_run(tmp_path, 2)
    if mutation == "missing_node":
        (run / "node-1.finished.json").unlink()
    elif mutation == "failed_node":
        path = run / "node-1.finished.json"
        value = json.loads(path.read_text())
        value["returncode"] = 1
        path.write_text(json.dumps(value))
    elif mutation == "checkpoint":
        next(run.rglob("optim/.metadata")).unlink()
    else:
        (run / "distributed-preflight.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _load("report")._summarize(run, 50)


@pytest.mark.parametrize(
    "text",
    [
        "52 : iter_speed nan seconds per iteration | Loss: 0.2",
        "52 : iter_speed 1 seconds per iteration | Loss: inf",
        "52 : iter_speed 1 seconds per iteration | Loss: 0.2\n" * 2,
        "ALL ranks NaN/Inf at iteration 10, skipping optimizer step",
        "53 : iter_speed 1 seconds per iteration | Loss: 0.2\n55 : iter_speed 1 seconds per iteration | Loss: 0.2",
    ],
)
def test_bad_timing_evidence_is_rejected(text):
    with pytest.raises(ValueError):
        _load("report")._timings(text, 50, 55)


def test_successful_exit_does_not_hide_data_loader_thread_failure(tmp_path):
    run = _completed_run(tmp_path)
    with (run / "node-0.log").open("a") as log:
        log.write(
            "Traceback (most recent call last):\n"
            "RuntimeError: could not unlink the shared memory file /torch_fixture\n"
        )
    with pytest.raises(ValueError, match="worker threads"):
        _load("report")._summarize(run, 50)


def test_scaling_requires_same_work_and_excludes_profiler(tmp_path):
    report_module = _load("report")
    baseline = report_module._summarize(_completed_run(tmp_path / "one"), 50)
    report = report_module._summarize(_completed_run(tmp_path / "two", 2), 50)
    report["step_mean_seconds"] = 3
    report_module._compare(report, baseline)
    assert report["speedup_vs_8_gpus"] == pytest.approx(5 / 3)
    assert report["scaling_efficiency"] == pytest.approx(5 / 6)
    report["comparison_contract"]["profile"] = True
    with pytest.raises(ValueError, match="profile"):
        report_module._compare(report, baseline)


def test_worker_failure_is_recorded_after_real_launcher_boundary(tmp_path, monkeypatch):
    recipe = _load("recipe")
    args = _args(tmp_path, 2)
    recipe._plan(args)
    monkeypatch.setenv("SLURM_NODEID", "1")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("MASTER_ADDR", "rank-zero")
    monkeypatch.setattr(recipe, "_verify_inputs", lambda *_: None)
    monkeypatch.setattr(
        recipe,
        "_verify_gpus",
        lambda count, hardware=None: {"gpu_names": ["NVIDIA B200"] * count},
    )
    commands = []

    def execute(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0 if len(commands) == 1 else 17)

    monkeypatch.setattr(recipe.subprocess, "run", execute)
    assert recipe._run_node(args) == 17
    assert commands[0][-1].endswith("distributed_preflight.py")
    assert "cosmos_framework.scripts.train" in commands[1]
    assert "--node_rank=1" in commands[1]
    receipt = json.loads((args.run_dir / "node-1.finished.json").read_text())
    assert receipt["returncode"] == 17
    assert receipt["run_sha256"]


def test_modified_run_settings_invalidate_measurement(tmp_path):
    run = _completed_run(tmp_path)
    path = run / "run.json"
    value = json.loads(path.read_text())
    value["seed"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed after execution"):
        _load("report")._summarize(run, 50)


def test_profile_captures_one_rank_per_node(tmp_path):
    recipe = _load("recipe")
    args = _args(tmp_path, 4)
    args.profile = True
    options = recipe._native_options(recipe._settings(args), tmp_path)
    assert "trainer.profiling.target_ranks=[0,8,16,24]" in options
    assert "trainer.profiling.enable_profiling=true" in options


def test_reserved_cluster_plan_cannot_fall_back(tmp_path, monkeypatch):
    from npa.soperator.spec import load_spec

    monkeypatch.setenv("NPA_PROJECT_ID", "project-placeholder")
    monkeypatch.setenv("NPA_TENANT_ID", "tenant-placeholder")
    monkeypatch.setenv("NPA_CAPACITY_BLOCK_GROUP", "reservation-placeholder")
    args = type(
        "Arguments",
        (),
        {"name": "wamtest", "nodes": 2, "output": tmp_path / "cluster.json"},
    )()
    _load("cluster")._render(args)
    spec = load_spec(args.output)
    assert spec.workers[0].capacity_mode() == "reserved"
    assert not spec.workers[0].preemptible
    assert spec.workers[0].size == 2
    assert spec.slurm_operator_version == "4.1.13"
    assert spec.k8s_version == "1.35"
    assert spec.node_group_version == ""
    assert args.output.stat().st_mode & 0o077 == 0


def test_policy_readiness_never_follows_http_redirects():
    requests = []

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/unexpected")
            self.end_headers()

        def log_message(self, *_args):
            return

    with HTTPServer(("127.0.0.1", 0), RedirectHandler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with pytest.raises(ConnectionError, match="not ready"):
                _load("evaluate")._server_info(server.server_port - 8000)
        finally:
            server.shutdown()
            thread.join()
    assert requests == ["/info"]


def test_four_gpu_plan_keeps_work_fixed_and_launches_only_four_ranks(
    tmp_path, monkeypatch
):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    recipe = _load("recipe")
    args = _args(tmp_path)
    args.gpus_per_node = 4
    recipe._plan(args)
    settings = json.loads((args.run_dir / "run.json").read_text())
    config = tomllib.loads((args.run_dir / "train.toml").read_text())
    assert settings["gpus"] == 4
    assert settings["gradient_accumulation"] == 8
    assert settings["global_batch"] == 2048
    assert config["model"]["parallelism"]["data_parallel_shard_degree"] == 4
    assert config["trainer"]["grad_accum_iter"] == 8
    script = (args.run_dir / "train.sbatch").read_text()
    assert "#SBATCH --gpus-per-node=4" in script
    assert "--gpus-per-task=4" in script
    monkeypatch.setenv("MASTER_ADDR", "rank-zero")
    assert "--nproc_per_node=4" in recipe._training_argv(settings, args.run_dir, 0)
    assert recipe._environment(settings, args.run_dir)["NPA_WAM_GPUS_PER_NODE"] == "4"


@pytest.mark.parametrize("visible", [4, 8])
def test_four_gpu_run_rejects_accidental_eight_gpu_visibility(monkeypatch, visible):
    cuda = SimpleNamespace(
        device_count=lambda: visible, get_device_name=lambda _: "NVIDIA B200"
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=cuda, __version__="fixture", version=SimpleNamespace(cuda="fixture")
        ),
    )
    recipe = _load("recipe")
    if visible == 4:
        assert len(recipe._verify_gpus(4)["gpu_names"]) == 4
    else:
        with pytest.raises(ValueError, match="exactly 4"):
            recipe._verify_gpus(4)


@pytest.mark.parametrize(
    "identities,world",
    [
        ([("host", n) for n in range(4)], 4),
        ([("host", n) for n in (0, 1, 2, 2)], 4),
        ([("host", n) for n in (0, 1, 2, 4)], 4),
        ([("host", n) for n in range(8)], 8),
    ],
)
def test_four_gpu_collective_requires_exact_local_rank_placement(identities, world):
    check = _load("distributed_preflight")._verify_placement
    if identities == [("host", n) for n in range(4)]:
        assert check(identities, world, 4) == 1
    else:
        with pytest.raises(RuntimeError):
            check(identities, world, 4)


def _fake_torch(monkeypatch, names):
    cuda = SimpleNamespace(
        device_count=lambda: len(names), get_device_name=lambda index: names[index]
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=cuda, __version__="fixture", version=SimpleNamespace(cuda="fixture")
        ),
    )
    return _load("recipe")


def test_b200_remains_the_default_hardware_setting(tmp_path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    recipe = _load("recipe")
    args = _args(tmp_path)
    assert args.hardware == "b200"
    args.gpus_per_node = 4
    recipe._plan(args)
    settings = json.loads((args.run_dir / "run.json").read_text())
    assert "hardware" not in settings
    assert settings["schema"] == "npa.cosmos3.wam-slurm.v1"
    config = tomllib.loads((args.run_dir / "train.toml").read_text())
    assert config["model"]["parallelism"]["data_parallel_shard_degree"] == 4
    assert config["trainer"]["grad_accum_iter"] == 8
    script = (args.run_dir / "train.sbatch").read_text()
    assert "#SBATCH --gpus-per-node=4" in script
    assert "--gpus-per-task=4" in script
    env = recipe._environment(settings, args.run_dir)
    assert env["NPA_WAM_GPUS_PER_NODE"] == "4"
    assert env["NPA_WAM_HARDWARE"] == "NVIDIA B200"


def _rtx_args(tmp_path):
    args = _args(tmp_path)
    args.hardware = "rtx-pro-6000"
    args.gpus_per_node = 8
    return args


def test_rtx_pro_6000_plan_renders_one_node_of_eight_gpus(tmp_path, monkeypatch):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    recipe._plan(args)
    settings = json.loads((args.run_dir / "run.json").read_text())
    config = tomllib.loads((args.run_dir / "train.toml").read_text())
    assert settings["hardware"] == {
        "default": False,
        "label": "RTX PRO 6000 Blackwell Server Edition",
    }
    assert settings["nodes"] == 1 and settings["gpus"] == 8
    assert settings["gpus_per_node"] == 8
    assert settings["global_batch"] == 2048
    assert settings["samples_per_rank"] == 64
    assert settings["gradient_accumulation"] == 4
    assert config["model"]["parallelism"]["data_parallel_shard_degree"] == 8
    assert config["model"]["parallelism"]["data_parallel_replicate_degree"] == 1
    assert config["trainer"]["grad_accum_iter"] == 4
    script = (args.run_dir / "train.sbatch").read_text()
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "--ntasks=1 " in script and "--gpus-per-task=8" in script
    env = recipe._environment(settings, args.run_dir)
    assert env["NPA_WAM_HARDWARE"] == "RTX PRO 6000 Blackwell Server Edition"
    monkeypatch.setenv("MASTER_ADDR", "rank-zero")
    argv = recipe._training_argv(settings, args.run_dir, 0)
    assert "--nproc_per_node=8" in argv and "--nnodes=1" in argv
    assert "--node_rank=0" in argv


@pytest.mark.parametrize("nodes", [2, 4])
def test_rtx_pro_6000_rejects_multinode_plans(tmp_path, nodes):
    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    args.nodes = nodes
    args.global_batch = 2048
    with pytest.raises(ValueError, match="exactly one node"):
        recipe._plan(args)
    assert not args.run_dir.exists()


@pytest.mark.parametrize("gpus_per_node", [4])
def test_rtx_pro_6000_rejects_wrong_gpu_count_plan(tmp_path, gpus_per_node):
    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    args.gpus_per_node = gpus_per_node
    with pytest.raises(ValueError, match="exactly eight GPUs per node"):
        recipe._plan(args)
    assert not args.run_dir.exists()


def test_unknown_hardware_setting_is_rejected(tmp_path):
    recipe = _load("recipe")
    args = _args(tmp_path)
    args.hardware = "h200"
    with pytest.raises(ValueError, match="unknown hardware"):
        recipe._settings(args)


@pytest.mark.parametrize(
    "names",
    [
        ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 8,
        # Mixed labels are rejected even when every entry shares the RTX prefix.
        ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 7 + ["NVIDIA B200"],
        ["NVIDIA RTX PRO 6000 Blackwell"] * 8,
        # The Max-Q workstation part has a different inventory name.
        ["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"] * 8,
    ],
)
def test_rtx_pro_6000_inventory_accepts_only_eight_uniform_server_gpus(
    monkeypatch, names
):
    recipe = _fake_torch(monkeypatch, names)
    hardware = {"default": False, "label": "RTX PRO 6000 Blackwell Server Edition"}
    if names == ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 8:
        assert recipe._verify_gpus(8, hardware)["gpu_names"] == names
    else:
        with pytest.raises(
            ValueError, match="exactly 8 RTX PRO 6000 Blackwell Server Edition"
        ):
            recipe._verify_gpus(8, hardware)


@pytest.mark.parametrize(
    "names,hardware",
    [
        (
            ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * 8,
            {"label": "NVIDIA B200"},
        ),
        (["NVIDIA B200"] * 8, {"label": "RTX PRO 6000 Blackwell Server Edition"}),
    ],
)
def test_rtx_inventory_stays_closed_to_the_planned_family(monkeypatch, names, hardware):
    recipe = _fake_torch(monkeypatch, names)
    with pytest.raises(ValueError):
        recipe._verify_gpus(8, hardware)


@pytest.mark.parametrize("count", [4, 7, 9])
def test_rtx_pro_6000_rejects_wrong_visible_gpu_count(monkeypatch, count):
    recipe = _fake_torch(
        monkeypatch, ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] * count
    )
    hardware = {"default": False, "label": "RTX PRO 6000 Blackwell Server Edition"}
    with pytest.raises(
        ValueError, match="exactly 8 RTX PRO 6000 Blackwell Server Edition"
    ):
        recipe._verify_gpus(8, hardware)


def test_receipt_without_hardware_keeps_b200_validation(tmp_path, monkeypatch):
    recipe = _load("recipe")
    args = _args(tmp_path)
    recipe._plan(args)
    path = args.run_dir / "run.json"
    receipt = json.loads(path.read_text())
    assert "hardware" not in receipt
    settings = receipt
    assert recipe._topology(settings)["label"] == "NVIDIA B200"
    assert (
        recipe._environment(settings, args.run_dir)["NPA_WAM_HARDWARE"] == "NVIDIA B200"
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 8,
                get_device_name=lambda _: "NVIDIA B200",
            ),
            __version__="fixture",
            version=SimpleNamespace(cuda="fixture"),
        ),
    )
    assert len(recipe._verify_gpus(8, settings.get("hardware"))["gpu_names"]) == 8
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 8,
                get_device_name=lambda _: (
                    "NVIDIA RTX PRO 6000 Blackwell Server Edition"
                ),
            ),
            __version__="fixture",
            version=SimpleNamespace(cuda="fixture"),
        ),
    )
    with pytest.raises(ValueError, match="exactly 8 NVIDIA B200"):
        recipe._verify_gpus(8, settings.get("hardware"))


@pytest.mark.parametrize(
    "identities,world,devices,hardware,valid",
    [
        # An eight-rank single host is the RTX placement that passes.
        (
            [("host", n) for n in range(8)],
            8,
            8,
            "RTX PRO 6000 Blackwell Server Edition",
            True,
        ),
        # The same world split across two hosts is rejected under the RTX setting.
        (
            [("a", n) for n in range(4)] + [("b", n) for n in range(4)],
            8,
            8,
            "RTX PRO 6000 Blackwell Server Edition",
            False,
        ),
        # Two full nodes are also rejected under the RTX setting.
        (
            [("a", n) for n in range(8)] + [("b", n) for n in range(8)],
            16,
            8,
            "RTX PRO 6000 Blackwell Server Edition",
            False,
        ),
        # Four cards cannot satisfy the eight-device RTX setting.
        (
            [("host", n) for n in range(4)],
            4,
            4,
            "RTX PRO 6000 Blackwell Server Edition",
            False,
        ),
        # Four cards on one host still pass for B200.
        ([("host", n) for n in range(4)], 4, 4, None, True),
        # Two B200 nodes of eight cards still pass.
        (
            [("a", n) for n in range(8)] + [("b", n) for n in range(8)],
            16,
            8,
            None,
            True,
        ),
        # A repeated local rank differs from the planned GPU count.
        ([("host", n) for n in (0, 1, 2, 2, 4, 5, 6, 7)], 8, 8, None, False),
    ],
)
def test_rtx_and_b200_collective_placement(identities, world, devices, hardware, valid):
    check = _load("distributed_preflight")._verify_placement
    if valid:
        assert check(identities, world, devices, hardware) >= 1
    else:
        with pytest.raises(RuntimeError):
            check(identities, world, devices, hardware)


def test_rtx_collective_accepts_one_node_and_rejects_two():
    check = _load("distributed_preflight")._verify_placement
    one_node = [("host", n) for n in range(8)]
    two_nodes = [("a", n) for n in range(8)] + [("b", n) for n in range(8)]
    assert check(one_node, 8, 8, "RTX PRO 6000 Blackwell Server Edition") == 1
    with pytest.raises(RuntimeError, match="exactly one node"):
        check(two_nodes, 16, 8, "RTX PRO 6000 Blackwell Server Edition")
    assert check(two_nodes, 16, 8) == 2


@pytest.mark.parametrize(
    "failure",
    (
        None,
        "local-valid",
        "local-zero-peak",
        "local-wrong-plan",
        "local-wrong-native-batch",
        "local-wrong-native-accumulation",
        "local-wrong-native-shards",
        "local-missing-save-time",
        "local-stale-rank",
        "local-failed-rank",
        "local-extra-rank",
        "local-duplicate-uuid",
        "local-node-exit",
        "missing-update",
        "extra-update",
        "nonfinite",
        "worker-exit",
        "wrong-device",
        "stale-settings",
        "incomplete-checkpoint",
    ),
)
def test_rtx_smoke_report_requires_native_update_and_checkpoint_evidence(
    tmp_path, failure
):
    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    args.steps = 2
    local = failure is not None and failure.startswith("local-")
    if local:
        args.launcher = "torchrun"
    if failure == "local-wrong-plan":
        args.samples_per_rank = 32
    recipe._plan(args)
    run = args.run_dir
    proof = {
        "status": "passed",
        "backend": "nccl",
        "world_size": 8,
        "hosts": 1,
        "all_reduce_sum": 36,
    }
    (run / "distributed-preflight.json").write_text(json.dumps(proof))
    node = {
        "rank": 0,
        "returncode": 1 if failure == "worker-exit" else 0,
        "run_sha256": hashlib.sha256((run / "run.json").read_bytes()).hexdigest(),
        "hardware": {
            "gpu_names": [
                "NVIDIA B200"
                if failure == "wrong-device"
                else recipe.RTX_PRO_6000["names"][0]
            ]
            * 8
        },
        "train_process_seconds": 31.2,
        "preflight_seconds": 1.0,
        "trainer_started_unix": 1767225600.0,
    }
    if local:
        import uuid

        (run / "ranks").mkdir()
        for rank in range(8 + (failure == "local-extra-rank")):
            receipt = dict(
                rank=rank,
                local_rank=rank,
                host="fixture",
                run_sha256=node["run_sha256"],
                returncode=0,
                peak_allocated_bytes=0
                if rank == 7 and failure == "local-zero-peak"
                else 100,
                device_uuid=str(uuid.UUID(int=rank + 1)),
            )
            if rank == 7 and failure == "local-stale-rank":
                receipt["run_sha256"] = "0" * 64
            if rank == 7 and failure == "local-failed-rank":
                receipt["returncode"] = 17
            if rank == 7 and failure == "local-duplicate-uuid":
                receipt["device_uuid"] = str(uuid.UUID(int=1))
            (run / f"ranks/{rank}.json").write_text(json.dumps(receipt))
        if failure == "local-node-exit":
            node["returncode"] = 9
    if failure == "stale-settings":
        node["run_sha256"] = "0" * 64
    (run / "node-0.finished.json").write_text(json.dumps(node))
    log = "Iteration 1: Hit counter: 1/50 | Loss: 0.9900 | Time: 2.00s\n"
    if failure != "missing-update":
        loss = "nan" if failure == "nonfinite" else "0.9000"
        log += f"Iteration 2: Hit counter: 2/50 | Loss: {loss} | Time: 1.00s\n"
    if failure == "extra-update":
        log += "Iteration 3: Hit counter: 3/50 | Loss: 0.9010 | Time: 1.00s\n"
    log += "[01-01 00:00:05|INFO|trainer:301:train] Starting training...\nCheckpoint save completed: Time taken: 3.00 seconds\n"
    if failure == "local-missing-save-time":
        log = log.replace("Checkpoint save completed:", "unmeasured save:")
    (run / "node-0.log").write_text(log)
    job = run / "output/cosmos3_wam/libero_10" / args.name
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = job / "checkpoints/iter_000000002" / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_text("native metadata")
        if failure != "incomplete-checkpoint" or component != "trainer":
            (directory / "__0_0.distcp").write_text("native bytes")
    (job / "config.yaml").write_text(
        json.dumps(
            {
                "model": {
                    "config": {
                        "parallelism": {
                            "data_parallel_shard_degree": 4
                            if failure == "local-wrong-native-shards"
                            else 8,
                            "data_parallel_replicate_degree": 1,
                            "context_parallel_shard_degree": 1,
                        }
                    }
                },
                "trainer": {
                    "grad_accum_iter": 2
                    if failure == "local-wrong-native-accumulation"
                    else 4,
                    "max_iter": 2,
                },
                "dataloader_train": {
                    "max_samples_per_batch": 32
                    if failure == "local-wrong-native-batch"
                    else 64
                },
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-B", str(RECIPE / "smoke_report.py"), "--run-dir", str(run)],
        text=True,
        capture_output=True,
    )
    if failure not in (None, "local-valid"):
        assert result.returncode == 1, result.stderr
        assert result.stdout == ""
        return
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["completed_optimizer_updates"] == 2
    assert [update["step"] for update in report["updates"]] == [1, 2]
    assert len(report["checkpoint_files"]) == 8
    if local:
        assert len(report["trainer_ranks"]) == 8
        assert report["train_process_seconds"] == 31.2


@pytest.mark.parametrize("same_checkpoint", (True, False))
def test_native_reload_skips_trials_and_requires_exact_checkpoint(
    tmp_path, monkeypatch, same_checkpoint
):
    monkeypatch.syspath_prepend(str(RECIPE))
    evaluation = _load("evaluate")
    run = tmp_path / "run"
    job = run / "output/cosmos3_wam/libero_10/rtx-smoke"
    checkpoint = job / "checkpoints/iter_000000002"
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        directory.mkdir(parents=True)
        (directory / ".metadata").write_text("metadata")
        (directory / "__0_0.distcp").write_text("weights")
    (job / "config.yaml").write_text("job: smoke")
    (run / "run.json").write_text(json.dumps({"name": "rtx-smoke", "steps": 2}))
    args = SimpleNamespace(
        shared_root=tmp_path,
        run_dir=run,
        output_dir=tmp_path / "reload",
        step=2,
        workers=1,
        seed=42,
        record_rollouts=False,
        reload_only=True,
    )
    stopped = []
    fake_server = SimpleNamespace(poll=lambda: None)
    observed_path = checkpoint / ("model" if same_checkpoint else "other")
    monkeypatch.setattr(evaluation, "_validate", lambda args: checkpoint)
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *_, **__: fake_server)
    monkeypatch.setattr(
        evaluation, "_server_info", lambda _: {"checkpoint": str(observed_path)}
    )
    monkeypatch.setattr(evaluation, "_stop", stopped.append)
    monkeypatch.setattr(evaluation.socket, "socket", lambda: DummyPort())
    monkeypatch.setattr(
        evaluation,
        "_client_command",
        lambda *_: pytest.fail("reload launched a rollout"),
    )
    if same_checkpoint:
        evaluation._main(args)
        receipt = json.loads((args.output_dir / "reload.json").read_text())
        manifest = json.loads((args.output_dir / "checkpoint-hashes.json").read_text())
        assert (
            receipt["loaded_sha256"]
            == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        )
        assert not (args.output_dir / "quality.json").exists()
    else:
        with pytest.raises(ValueError, match="unexpected checkpoint"):
            evaluation._main(args)
        assert not (args.output_dir / "reload.json").exists()
    assert stopped == [fake_server]


@pytest.mark.parametrize(
    "trainer_code, preflight_failure", [(0, False), (17, False), (0, True)]
)
def test_local_worker_uses_standalone_torchrun(
    tmp_path, monkeypatch, trainer_code, preflight_failure
):
    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    args.launcher = "torchrun"
    args.steps = 2
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    recipe._plan(args)
    run = args.run_dir
    settings = json.loads((run / "run.json").read_text())
    assert settings["launcher"] == "torchrun"
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    assert settings["torch_compile_disable"] is True
    assert recipe._environment(settings, run)["TORCH_COMPILE_DISABLE"] == "1"
    assert (run / "trainer_rank.py").read_bytes() == (
        RECIPE / "trainer_rank.py"
    ).read_bytes()
    assert not (run / "train.sbatch").exists()
    monkeypatch.delenv("SLURM_NODEID", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "stale")
    fifo = tmp_path / "nccl.fifo"
    os.mkfifo(fifo)
    monkeypatch.setenv("NCCL_DEBUG_FILE", str(fifo))
    monkeypatch.setattr(recipe, "_verify_inputs", lambda *_: None)
    monkeypatch.setattr(
        recipe,
        "_verify_gpus",
        lambda *_: {"gpu_names": [recipe.RTX_PRO_6000["names"][0]] * 8},
    )
    calls = []

    def launch(argv, *, env, **kwargs):
        assert "MASTER_ADDR" not in env
        assert env["NCCL_DEBUG_FILE"] == str(run / "nccl-node-0-%p.log")
        calls.append(argv)
        if preflight_failure and len(calls) == 3:
            raise subprocess.CalledProcessError(9, argv)
        return SimpleNamespace(returncode=trainer_code)

    monkeypatch.setattr(recipe.subprocess, "run", launch)
    if preflight_failure:
        with pytest.raises(subprocess.CalledProcessError):
            recipe._run_worker(args, local=True)
        assert len(calls) == 3 and not (run / "node-0.finished.json").exists()
    else:
        assert recipe._run_worker(args, local=True) == trainer_code
        node = json.loads((run / "node-0.finished.json").read_text())
        assert node["returncode"] == trainer_code
        assert calls[3] == [
            *calls[2][:-1],
            str(run / "trainer_rank.py"),
            *recipe._native_options(settings, run),
        ]
    assert calls.pop(0)[1:] == [run / "recipe.py", "decode-worker", "--run-dir", run]
    assert calls[0][1:4] == ["-m", "cosmos_framework.scripts.train", "--dryrun"]
    assert calls[1][1:] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "--max_restarts=0",
        str(run / "distributed_preflight.py"),
    ]
    with pytest.raises(FileExistsError):
        recipe._run_worker(args, local=True)


def test_local_mode_is_explicit_and_claim_is_exclusive(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    recipe = _load("recipe")
    args = _rtx_args(tmp_path)
    assert args.launcher == "slurm"
    args.launcher = "torchrun"
    args.hardware = "b200"
    with pytest.raises(ValueError, match="rtx-pro"):
        recipe._plan(args)
    args.hardware = "rtx-pro-6000"
    recipe._plan(args)
    settings = json.loads((args.run_dir / "run.json").read_text())
    with pytest.raises(ValueError, match="launcher"):
        recipe._worker_rank(settings, args.run_dir, local=False)
    with pytest.raises(ValueError, match="launcher"):
        recipe._worker_rank({}, args.run_dir, local=True)

    def claim():
        try:
            recipe._worker_rank(settings, args.run_dir, local=True)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(claim) for _ in range(2)]
    assert sum(result.result() for result in results) == 1


@pytest.mark.parametrize("exit_code", [0, 17])
def test_trainer_wrapper_forwards_native_argv_and_records_rank(
    tmp_path, monkeypatch, exit_code
):
    import uuid

    wrapper = _load("trainer_rank")
    device = uuid.UUID(int=7)
    selected = []
    memory = [1800]
    cuda = SimpleNamespace(
        set_device=selected.append,
        get_device_properties=lambda index: SimpleNamespace(uuid=device),
        max_memory_allocated=lambda index: memory[0],
        reset_peak_memory_stats=lambda device=None: memory.__setitem__(0, 23),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    for key, value in {
        "NPA_WAM_RUN_DIR": str(tmp_path),
        "RANK": "3",
        "LOCAL_RANK": "3",
        "WORLD_SIZE": "8",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("COSMOS_EXIT_WITHOUT_FINALIZE", raising=False)
    (tmp_path / "run.json").write_text("{}")
    package = tmp_path / "cosmos_framework/scripts"
    package.mkdir(parents=True)
    for directory in (package, package.parent):
        (directory / "__init__.py").touch()
    (package / "train.py").write_text(
        'import sys\nif __name__ == "__main__":\n'
        '    assert sys.argv[1:] == ["--sft-toml", "config"]\n'
        "    import torch\n    torch.cuda.reset_peak_memory_stats()\n"
        f"    raise SystemExit({exit_code})\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # runpy imports the two containing packages. Restore them after the fixture.
    for name in ("cosmos_framework", "cosmos_framework.scripts"):
        monkeypatch.setitem(sys.modules, name, None)
        del sys.modules[name]
    monkeypatch.setattr(sys, "argv", ["trainer_rank.py", "--sft-toml", "config"])
    assert wrapper._rank_main() == exit_code
    receipt = json.loads((tmp_path / "ranks/3.json").read_text())
    assert selected == [3] and receipt["device_uuid"] == str(device)
    assert (
        receipt["returncode"] == exit_code and receipt["peak_allocated_bytes"] == 1800
    )
    assert receipt["run_sha256"] == hashlib.sha256(b"{}").hexdigest()
    with pytest.raises(FileExistsError):
        wrapper._rank_main()


class DummyPort:
    def setsockopt(self, *_):
        pass

    def listen(self, _):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def bind(self, _):
        pass


def test_four_gpu_report_counts_active_devices_and_compares_runtime(tmp_path):
    reporter = _load("report")
    report = reporter._summarize(_completed_run(tmp_path / "four", gpus_per_node=4), 50)
    baseline = reporter._summarize(_completed_run(tmp_path / "eight"), 50)
    assert report["training_gpu_hours"] == pytest.approx(4 * 200 / 3600)
    reporter._compare(report, baseline)
    assert report["speedup_vs_8_gpus"] == 1
    baseline["hardware"]["torch"] = "different-runtime"
    with pytest.raises(ValueError, match="same runtime"):
        reporter._compare(report, baseline)
