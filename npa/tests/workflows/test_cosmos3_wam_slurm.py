"""Exercise native WAM launch topology, malformed measurements, and Slurm propagation."""

import importlib.util
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import threading

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"wam_{name}", RECIPE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _completed_run(tmp_path, nodes=1):
    recipe = _load("recipe")
    args = _args(tmp_path, nodes)
    recipe._plan(args)
    run = args.run_dir
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
                        "gpu_names": ["NVIDIA B200"] * 8,
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
                "world_size": 8 * nodes,
                "hosts": nodes,
            }
        )
    )
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
        recipe, "_verify_gpus", lambda: {"gpu_names": ["NVIDIA B200"] * 8}
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
