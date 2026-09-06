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
