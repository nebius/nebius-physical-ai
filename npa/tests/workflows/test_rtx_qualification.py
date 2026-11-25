"""Exercise the single-host qualification boundary and model-only receiver on CPU."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


def _module(name, monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    spec = importlib.util.spec_from_file_location(name, RECIPE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seconds", [0, -1, 2701])
def test_invalid_deadline_never_claims_run(seconds, tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        _module("qualify", monkeypatch)._qualify(tmp_path, seconds)
    assert not (tmp_path / "qualification").exists()


def test_deadline_failure_retains_evidence(tmp_path, monkeypatch):
    mod = _module("qualify", monkeypatch)

    def scope(argv, **kwargs):
        assert "systemd-run" == argv[0]
        assert "--property=RuntimeMaxSec=2700s" in argv
        assert "--property=KillMode=control-group" in argv
        assert "--property=KillSignal=SIGKILL" in argv
        return SimpleNamespace(returncode=137)

    monkeypatch.setattr(subprocess, "run", scope)
    with pytest.raises(RuntimeError, match="137"):
        mod._qualify(tmp_path, 2700)
    receipt = (tmp_path / "qualification/qualification.json").read_bytes()
    assert json.loads(receipt)["status"] == "failed"
    with pytest.raises(FileExistsError):
        mod._qualify(tmp_path, 2700)
    assert (tmp_path / "qualification/qualification.json").read_bytes() == receipt


def test_wrong_reloaded_checkpoint_fails_qualification(tmp_path, monkeypatch):
    mod = _module("qualify", monkeypatch)
    run = tmp_path
    evidence = run / "qualification"
    evidence.mkdir()
    settings = {
        "launcher": "torchrun",
        "hardware": {"label": "RTX PRO 6000 Blackwell Server Edition"},
        "nodes": 1,
        "gpus": 8,
        "gpus_per_node": 8,
        "samples_per_rank": 64,
        "gradient_accumulation": 4,
        "global_batch": 2048,
        "steps": 2,
        "seed": 42,
        "shared_root": str(tmp_path),
    }
    raw = json.dumps(settings).encode()
    (run / "run.json").write_bytes(raw)

    def launch(argv, **kwargs):
        if Path(argv[1]).name == "smoke_report.py":
            json.dump(
                {
                    "run_sha256": hashlib.sha256(raw).hexdigest(),
                    "checkpoint_sha256": "a" * 64,
                },
                kwargs["stdout"],
            )
        if Path(argv[1]).name == "evaluate.py":
            (evidence / "reload").mkdir()
            (evidence / "reload/reload.json").write_text(
                json.dumps({"step": 2, "loaded_sha256": "b" * 64})
            )

    monkeypatch.setattr(mod.subprocess, "run", launch)
    native = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **k: "0::/fixture.scope"
        if str(path) == "/proc/self/cgroup"
        else native(path, *a, **k),
    )
    with pytest.raises(ValueError, match="reload disagrees"):
        mod._worker(run, "fixture.scope")
    assert not (evidence / "verified.json").exists()


@pytest.mark.parametrize(
    "failure", [None, "report-hash", "local-reload", "run-id", "shard"]
)
def test_receive_only_promotes_qualified_model(failure, tmp_path, monkeypatch):
    mod = _module("receive_model", monkeypatch)
    src = tmp_path / "source"
    dst = tmp_path / "destination"
    evidence = src / "qualification"
    evidence.mkdir(parents=True)
    relative = Path("output/cosmos3_wam/libero_10/trial/checkpoints/iter_000000002")
    model = src / relative / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    shard = model / "__0_0.distcp"
    shard.write_bytes(b"trained")
    settings = {"name": "trial", "steps": 2, "launcher": "torchrun"}
    raw = json.dumps(settings).encode()
    (src / "run.json").write_bytes(raw)
    _, manifest = mod._checkpoint_identity(src / relative)
    files = {"model/" + k: v for k, v in manifest.items()}
    report = {
        "schema": "npa.cosmos3.wam-rtx-smoke.v1",
        "completed_optimizer_updates": 2,
        "terminal_step": 2,
        "run_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_files": files,
        "checkpoint_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }
    report_bytes = json.dumps(report).encode()
    (evidence / "native-training.json").write_bytes(report_bytes)
    sha = hashlib.sha256(report_bytes).hexdigest()
    (evidence / "qualification.json").write_text(
        json.dumps(
            {
                "status": "failed" if failure == "local-reload" else "passed",
                "run_sha256": report["run_sha256"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "source_training_report_sha256": sha,
            }
        )
    )
    if failure == "run-id":
        (src / "run.json").write_bytes(raw + b" ")
    if failure == "shard":
        shard.write_bytes(b"changed")
    if failure == "report-hash":
        sha = "0" * 64

    def copy(argv, check):
        sources = argv[argv.index("--") + 1 : -1]
        for source in map(Path, sources):
            target = Path(argv[-1]) / source.name
            if source.is_dir():
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target, follow_symlinks=False)

    monkeypatch.setattr(subprocess, "run", copy)
    if failure:
        with pytest.raises(ValueError):
            mod._receive(src, None, dst, sha)
        assert not (dst / relative / "model").exists()
        assert not (dst / "transfer.json").exists()
    else:
        receipt = mod._receive(src, None, dst, sha)
        assert receipt["model_bytes"] == 15 and receipt["model_files"] == 2
        assert (dst / relative / "model" / shard.name).read_bytes() == b"trained"
        with pytest.raises(FileExistsError):
            mod._receive(src, None, dst, sha)
