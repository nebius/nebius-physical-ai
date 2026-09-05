"""Exercise real checkpoint copies with a synthetic trainer and local artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from npa.workflows import rl_sweep


@pytest.mark.parametrize("last_step", [49, 149])
def test_train_variant_publishes_last_numeric_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, last_step: int
) -> None:
    monkeypatch.chdir(tmp_path)
    final_bytes = f"synthetic checkpoint at step {last_step}".encode()

    def trainer(argv: list[str]) -> subprocess.CompletedProcess[str]:
        run = Path("logs/rsl_rl/npa_rl_sweep/2026-01-01_12-00-00-policy")
        run.mkdir(parents=True)
        for step in range(last_step + 1):
            (run / f"model_{step}.pt").write_bytes(
                f"synthetic checkpoint at step {step}".encode()
            )
        return subprocess.CompletedProcess(
            argv, 0, stdout="Mean reward: 10.0\n", stderr=""
        )

    output = tmp_path / "published"
    result = rl_sweep.train_variant(
        variant="policy",
        output_uri=str(output),
        iterations=last_step + 1,
        overrides="agent.save_interval=1",
        run_id="synthetic",
        train_script="train.py",
        python_bin="unused-interpreter",
        runner=trainer,
    )

    assert result["status"] == "success"
    assert Path(result["checkpoint_uri"]).read_bytes() == final_bytes
    for name in (rl_sweep.METRICS_FILENAME, rl_sweep.SUMMARY_FILENAME):
        metrics = json.loads((output / name).read_text())
        assert metrics["checkpoint_uri"] == result["checkpoint_uri"]


def test_sparse_checkpoints_use_steps_not_write_order(tmp_path: Path) -> None:
    run = tmp_path / "logs" / "run"
    run.mkdir(parents=True)
    for step in (100, 99, 12, 8):
        (run / f"model_{step}.pt").write_bytes(str(step).encode())

    published = rl_sweep._publish_checkpoint(
        str(tmp_path / "published"), str(tmp_path / "logs")
    )

    assert Path(published).read_bytes() == b"100"


def test_latest_run_directory_takes_precedence_over_step(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    for run, step in (("2026-01-02-run", 9), ("2026-01-01-run", 999)):
        directory = logs / run
        directory.mkdir(parents=True)
        (directory / f"model_{step}.pt").write_bytes(run.encode())

    published = rl_sweep._publish_checkpoint(str(tmp_path / "published"), str(logs))

    assert Path(published).read_bytes() == b"2026-01-02-run"


def test_nested_directory_does_not_change_selected_run(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    nested = logs / "run" / "earlier-run"
    nested.mkdir(parents=True)
    (nested / "model_999.pt").write_bytes(b"nested run")
    (logs / "run" / "model_9.pt").write_bytes(b"selected run")

    published = rl_sweep._publish_checkpoint(str(tmp_path / "published"), str(logs))

    assert Path(published).read_bytes() == b"selected run"


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("model_0009.pt", "model_0049.pt"), "model_0049.pt"),
        (("model_best.pt", "model_latest.pt"), "model_latest.pt"),
    ],
)
def test_existing_checkpoint_filename_formats(
    tmp_path: Path, names: tuple[str, ...], expected: str
) -> None:
    run = tmp_path / "logs" / "run"
    run.mkdir(parents=True)
    for name in names:
        (run / name).write_bytes(name.encode())

    published = rl_sweep._publish_checkpoint(
        str(tmp_path / "published"), str(tmp_path / "logs")
    )

    assert Path(published).read_bytes() == expected.encode()


@pytest.mark.parametrize("create_root", [False, True])
def test_no_checkpoint_does_not_create_output(tmp_path: Path, create_root: bool) -> None:
    logs = tmp_path / "logs"
    if create_root:
        logs.mkdir()
    output = tmp_path / "published"

    assert rl_sweep._publish_checkpoint(str(output), str(logs)) == ""
    assert not output.exists()
