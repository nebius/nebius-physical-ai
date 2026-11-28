"""Check the evaluator's training-to-scoring identity before starting any simulator."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "run",
        "step",
        "full_digest",
        "shard",
        "symlink",
        "missing",
        "extra",
        "no_source",
        "transfer",
        "verified_copy",
        "missing_receipt",
        "model_bytes",
        "model_files",
        "model_transfer_seconds",
        "destination_hash_seconds",
        "total_receive_seconds",
        "reload_with_receipt",
    ],
)
def test_score_bound_to_training(tmp_path, monkeypatch, fault):
    spec = importlib.util.spec_from_file_location("evaluation", RECIPE / "evaluate.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    run = tmp_path / "run"
    checkpoint = run / "output/cosmos3_wam/libero_10/test/checkpoints/iter_000000002"
    model = checkpoint / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    shard = model / "rank0.distcp"
    shard.write_bytes(b"policy-a")
    settings = {
        "launcher": "torchrun",
        "torch_compile_disable": True,
        "name": "test",
        "seed": 42,
        "steps": 2,
        "run_id": "cpu-run",
        "sources": {"framework": "pin", "libero": "pin"},
    }
    raw = json.dumps(settings).encode()
    (run / "run.json").write_bytes(raw)
    (tmp_path / "prepared.json").write_text(
        json.dumps({"sources": settings["sources"]})
    )
    full = {
        "model/" + p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in model.iterdir()
    }
    full["optim/optimizer.distcp"] = {"sha256": "a" * 64, "bytes": 7}
    report = {
        "schema": "npa.cosmos3.wam-rtx-smoke.v1",
        "run_sha256": hashlib.sha256(raw).hexdigest(),
        "terminal_step": 2,
        "completed_optimizer_updates": 2,
        "checkpoint_files": full,
        "checkpoint_sha256": evaluation._manifest_digest(full),
    }
    if fault == "run":
        (run / "run.json").write_bytes(raw + b" ")
    elif fault == "step":
        report["terminal_step"] = 1
    elif fault == "shard":
        shard.write_bytes(b"policy-b")
    elif fault == "full_digest":
        report["checkpoint_sha256"] = "0" * 64
    elif fault == "symlink":
        other = tmp_path / "copy"
        shard.rename(other)
        shard.symlink_to(other)
    elif fault == "missing":
        shard.unlink()
    elif fault == "extra":
        (model / "extra").touch()
    elif fault == "transfer":
        (run / "transfer.json").write_text(
            json.dumps({"source_training_report_sha256": "0" * 64})
        )
    source = run / "training.json"
    source.write_text(json.dumps(report))
    required = fault in {
        "verified_copy",
        "missing_receipt",
        "model_bytes",
        "model_files",
        "model_transfer_seconds",
        "destination_hash_seconds",
        "total_receive_seconds",
        "reload_with_receipt",
    }
    if required and fault != "missing_receipt":
        expected = evaluation._source_model(report, raw, 2)
        receipt = {
            "source_training_report_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "run_sha256": report["run_sha256"],
            "model_manifest_sha256": evaluation._manifest_digest(expected),
            "model_bytes": 16,
            "model_files": 2,
            "model_transfer_seconds": 3,
            "destination_hash_seconds": 2,
            "total_receive_seconds": 6,
        }
        overrides = {
            "model_bytes": 15,
            "model_files": 1,
            "model_transfer_seconds": False,
            "destination_hash_seconds": float("nan"),
            "total_receive_seconds": 4,
        }
        if fault in overrides:
            receipt[fault] = overrides[fault]
        (run / "transfer.json").write_text(json.dumps(receipt))
    args = SimpleNamespace(
        shared_root=tmp_path,
        run_dir=run,
        output_dir=tmp_path / "score",
        seed=42,
        workers=None,
        envs=None,
        trials=None,
        record_rollouts=False,
        step=2,
        reload_only=fault == "reload_with_receipt",
        require_transfer_receipt=required,
        source_training_report=None if fault == "no_source" else source,
    )
    monkeypatch.setattr(evaluation, "_check_revision", lambda *_: None)

    def score(args, checkpoint):
        assert args.torch_compile_disable is True
        assert (args.workers, args.envs, args.trials) == (1, 1, 1)
        (args.output_dir / "worker-0").mkdir()
        (args.output_dir / "worker-0/completion.json").write_text(
            '{"rollout_seconds":0.1,"server_ready_seconds":0.01}'
        )
        return [{"successes": 0} for _ in range(10)]

    if fault not in (None, "verified_copy"):
        monkeypatch.setattr(
            evaluation,
            "_evaluate_tasks",
            lambda *_: pytest.fail("started scoring before verifying source"),
        )
        with pytest.raises(ValueError):
            evaluation._main(args)
        assert not (args.output_dir / "quality.json").exists()
    else:
        monkeypatch.setattr(evaluation, "_evaluate_tasks", score)
        evaluation._main(args)
        quality = json.loads((args.output_dir / "quality.json").read_text())
        assert quality["model_bytes"] == 16 and quality["trials"] == 10
        assert quality["model_manifest_sha256"] != quality["source_checkpoint_sha256"]
        assert quality["worker_timings"][0]["rollout_seconds"] == 0.1


def test_slurm_keeps_prior_evaluation_defaults(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("evaluation", RECIPE / "evaluate.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    model = (
        tmp_path / "output/cosmos3_wam/libero_10/trial/checkpoints/iter_000000500/model"
    )
    model.mkdir(parents=True)
    (model / ".metadata").touch()
    pins = {"framework": "p", "libero": "q"}
    (tmp_path / "run.json").write_text(
        json.dumps({"name": "trial", "steps": 2000, "seed": 42, "sources": pins})
    )
    (tmp_path / "prepared.json").write_text(json.dumps({"sources": pins}))
    args = SimpleNamespace(
        run_dir=tmp_path,
        shared_root=tmp_path,
        step=500,
        seed=42,
        workers=None,
        trials=None,
        envs=None,
        record_rollouts=False,
    )
    monkeypatch.setattr(evaluation, "_check_revision", lambda *_: None)
    assert evaluation._validate(args) == model.parent
    assert (args.workers, args.trials, args.envs) == (8, 50, 8)
