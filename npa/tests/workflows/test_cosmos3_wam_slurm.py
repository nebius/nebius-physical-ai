"""Exercise native WAM launch topology, malformed measurements, and Slurm propagation."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

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
        "srun": 'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"',
    }.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "workers")
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


def _completed_run(tmp_path, nodes=1):
    recipe = _load("recipe")
    args = _args(tmp_path, nodes)
    recipe._plan(args)
    run = args.run_dir
    for rank in range(nodes):
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
