"""Checkpoint bytes published after a successful full runner invocation."""
import json
import subprocess
from pathlib import Path
import pytest
from npa.workflows import rl_sweep

@pytest.mark.parametrize('last_step', [49, 149])
def test_training_publishes_final_numeric_checkpoint(tmp_path, monkeypatch, last_step):
    monkeypatch.chdir(tmp_path)
    def runner(argv):
        log = Path('logs/rsl_rl/npa_rl_sweep/2026-01-01_12-00-00-policy')
        log.mkdir(parents=True)
        for step in range(last_step + 1):
            (log / f'model_{step}.pt').write_bytes(json.dumps({'iteration':step}).encode())
        return subprocess.CompletedProcess(argv, 0, stdout='Mean reward: 10.0\n', stderr='')
    output = tmp_path / 'published'
    result = rl_sweep.train_variant(variant='policy', output_uri=str(output), task='Isaac-Cartpole-v0', iterations=last_step+1, num_envs=4096, overrides='agent.save_interval=1', run_id='fixture', train_script='train.py', python_bin='unused-interpreter', runner=runner)
    assert result['status'] == 'success'
    assert json.loads(Path(result['checkpoint_uri']).read_bytes())['iteration'] == last_step
    assert json.loads((output / rl_sweep.METRICS_FILENAME).read_text())['checkpoint_uri'] == result['checkpoint_uri']


def test_latest_run_directory_still_takes_precedence(tmp_path):
    old = tmp_path / 'logs/2026-01-01-run'
    recent = tmp_path / 'logs/2026-01-02-run'
    old.mkdir(parents=True)
    recent.mkdir(parents=True)
    (old / 'model_999.pt').write_bytes(b'older-run')
    (recent / 'model_9.pt').write_bytes(b'current-run')
    output = rl_sweep._publish_checkpoint(str(tmp_path / 'published'), str(tmp_path / 'logs'))
    assert Path(output).read_bytes() == b'current-run'


def test_sparse_checkpoints_use_steps_not_write_order(tmp_path: Path) -> None:
    run = tmp_path / "logs" / "run"
    run.mkdir(parents=True)
    for step in (100, 99, 12, 8):
        (run / f"model_{step}.pt").write_bytes(str(step).encode())

    published = rl_sweep._publish_checkpoint(
        str(tmp_path / "published"), str(tmp_path / "logs")
    )

    assert Path(published).read_bytes() == b"100"


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
