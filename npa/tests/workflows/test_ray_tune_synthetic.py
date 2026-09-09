"""Contracts for the guarded native Ray Tune synthetic reference."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml


EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-tune-synthetic"


def load(name: str):
    """Load one standalone example module from its file path."""
    module_name = "ray_tune_example_" + name.replace("/", "_")
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_files_and_runtime_contract_exist() -> None:
    expected = {
        "README.md",
        "inspect_results.py",
        "search.py",
        "cluster.yaml",
        "cluster/prepare.py",
        "cluster/requirements.txt",
        "cluster/start.sh",
    }
    assert expected <= {
        path.relative_to(EXAMPLE).as_posix() for path in EXAMPLE.rglob("*") if path.is_file()
    }
    requirements = (EXAMPLE / "cluster/requirements.txt").read_text().splitlines()
    assert requirements == ["ray[default,tune]==2.58.0"]
    readme = (EXAMPLE / "README.md").read_text()
    assert 'TUNE_RUNTIME="$HOME/.npa-ray-tune"' in readme
    assert "TUNE_RUNTIME=/root" not in readme


def test_skypilot_hosts_one_cpu_ray_service_without_cluster_autoscaling() -> None:
    spec = yaml.safe_load((EXAMPLE / "cluster.yaml").read_text())
    assert spec["num_nodes"] == 1
    assert "accelerators" not in spec["resources"]
    assert spec["resources"]["image_id"] == (
        "docker:docker.io/rayproject/ray@"
        "sha256:c3c9573c5c6bfe4127885f79622d6a32064d34cafc7d156ec728aab8657be250"
    )
    assert spec["config"]["kubernetes"]["pod_config"]["spec"] == {
        "serviceAccountName": "default",
        "automountServiceAccountToken": False,
    }
    assert spec["setup"].strip().endswith("/prepare.py")
    assert spec["run"].strip() == "bash /opt/npa-ray-tune-bootstrap/start.sh"


@pytest.mark.parametrize(
    "settings",
    [
        {"iterations": 0, "target": 0.2, "fail_value": None},
        {"iterations": 2, "target": float("nan"), "fail_value": None},
        {"iterations": 2, "target": 0.2, "fail_value": 0.9},
    ],
)
def test_invalid_search_settings_fail_before_ray(settings: dict) -> None:
    with pytest.raises(ValueError):
        load("search").validate_settings(settings)


@pytest.mark.parametrize("name", ["", ".", "../escape", "space name", "a" * 65])
def test_invalid_run_name_fails_before_ray(name: str) -> None:
    with pytest.raises(ValueError, match="run-name"):
        load("search").validate_run_name(name)


def test_summary_requires_all_trials_and_the_known_optimum() -> None:
    search = load("search")
    results = [
        SimpleNamespace(
            config={"step_size": value},
            metrics={"loss": (value - 0.2) ** 2, "iteration": 3, "resumed": False},
            error=None,
            path=f"trial-{index}",
            checkpoint=object(),
        )
        for index, value in enumerate(search.SEARCH_VALUES)
    ]
    summary = search.summarize(results, experiment_path="experiment")
    assert summary["best_step_size"] == 0.2
    assert summary["best_loss"] == 0.0
    assert summary["trial_count"] == len(search.SEARCH_VALUES)
    results[0].error = RuntimeError("trial failed")
    with pytest.raises(RuntimeError, match="trial failed"):
        search.summarize(results, experiment_path="experiment")


def test_relative_local_storage_is_normalized_for_arrow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load("search").normalize_storage_path("storage") == str(tmp_path / "storage")
    assert load("search").normalize_storage_path("s3://bucket/prefix") == "s3://bucket/prefix"


@pytest.mark.parametrize(
    "path",
    ["", "https://example.invalid/path", "s3://bucket", "s3://user@bucket/prefix", "s3://bucket/a/../b"],
)
def test_unsafe_remote_storage_is_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="storage-path"):
        load("search").normalize_storage_path(path)


def test_manifest_inspector_rejects_changed_bytes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    result = {"status": "succeeded", "ray_version": "2.58.0", "trial_count": 3}
    (output / "result.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    digest = hashlib.sha256((output / "result.json").read_bytes()).hexdigest()
    (output / "SHA256SUMS").write_text(f"{digest}  result.json\n")
    assert load("inspect_results").verify_manifest(output) == 1
    (output / "result.json").write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load("inspect_results").verify_manifest(output)


def test_inspector_rejects_self_consistent_false_trial_evidence(tmp_path: Path) -> None:
    search = load("search")
    output = tmp_path / "output"
    trials = [
        {
            "step_size": value,
            "loss": (value - 0.2) ** 2,
            "iterations": 3,
            "resumed": value == 0.3,
            "checkpoint_available": True,
        }
        for value in search.SEARCH_VALUES
    ]
    search._export(output, search._summary_record(trials, trials[1], "experiment"))
    trials[0]["loss"] = 99.0
    (output / "trials.json").write_text(json.dumps(trials, indent=2, sort_keys=True) + "\n")
    manifest = output / "SHA256SUMS"
    lines = []
    for name in search.ARTIFACT_NAMES:
        digest = hashlib.sha256((output / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    manifest.write_text("".join(lines))
    with pytest.raises(ValueError, match="trial metrics"):
        load("inspect_results").inspect(output)


def test_runtime_preparation_is_fresh_and_private(tmp_path: Path) -> None:
    prepare = load("cluster/prepare")
    root = tmp_path / "runtime"
    prepare._create_runtime(root)
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "exports").stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        prepare._create_runtime(root)


def test_runtime_preparation_rejects_shared_parent(tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    with pytest.raises(ValueError, match="shared writes"):
        load("cluster/prepare")._create_runtime(parent / "runtime")


def test_runtime_preparation_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "runtime"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        load("cluster/prepare")._create_runtime(link)
    assert list(target.iterdir()) == []


def test_service_start_is_scoped_and_blocking() -> None:
    script = (EXAMPLE / "cluster/start.sh").read_text()
    assert "unset RAY_ADDRESS RAY_API_SERVER_ADDRESS" in script
    assert "--dashboard-host=127.0.0.1" in script
    assert "--min-worker-port=10010 --max-worker-port=10999" in script
    assert " start --block --head " in script
    assert "ray stop" not in script


def test_real_local_ray_258_tune_execution(tmp_path: Path) -> None:
    ray = pytest.importorskip("ray")
    if ray.__version__ != "2.58.0":
        pytest.skip("the native reference requires Ray 2.58.0")
    storage = tmp_path / "storage"
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "search.py"),
            "--storage-path",
            str(storage),
            "--run-name",
            "local-contract",
            "--output-dir",
            str(output),
            "--fail-step-size",
            "0.3",
            "--local",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    assert receipt == {
        "best_loss": 0.0,
        "best_step_size": 0.2,
        "output_dir": str(output.resolve()),
        "status": "succeeded",
        "trial_count": 3,
    }
    inspected = load("inspect_results").inspect(output)
    assert inspected == {"artifacts_verified": 3, "resumed_trials": 1, "trials_verified": 3}
