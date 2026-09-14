"""Verify portable studio dispatch without operator configuration or shell interpolation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from npa import studio


def test_init_and_relocated_dispatch_need_no_cloud_config(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / 'docs/demos/executive-film'
    original = tmp_path / 'original studio'
    assert studio.run(['init', '--directory', str(original), '--renderer', str(source)]) == 0
    assert studio.run(['init', '--directory', str(original), '--renderer', str(source)]) == 1
    moved = tmp_path / 'new user' / 'studio with spaces'
    shutil.move(original, moved)
    project = moved / 'projects' / 'inference'
    project.mkdir()
    (project / 'film-project.json').write_text(json.dumps(dict(storyboard='story.json', assets='assets.json', voice_dir='voice', output_dir='renders')))
    (project / 'story.json').write_text(json.dumps({'scenes': [{'id':'one', 'duration':7, 'title':['Measured intelligence']}]}))
    registry = moved / 'studio.json'
    registry.write_text(json.dumps({'renderer':'renderer','projects':{'inference':'projects/inference/film-project.json'}}))
    monkeypatch.chdir(tmp_path)
    command = [sys.executable, '-m', 'npa', 'studio', '--registry', str(registry), 'inference', 'scenes']
    result = subprocess.run(command, env={'PATH':str(Path(sys.executable).parent)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Measured intelligence' in result.stdout
    assert not (moved / 'credentials.yaml').exists()


def test_registry_rejects_cloud_configuration_before_subprocess(tmp_path, monkeypatch):
    registry = tmp_path / 'studio.json'
    registry.write_text(json.dumps({'projects':{}, 'credentials':{'key':'placeholder'}}))
    monkeypatch.setattr(studio.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not execute')))
    assert studio.run(['--registry', str(registry), 'list']) == 1


def test_help_is_available_without_a_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert studio.run(['--help']) == 0
    assert 'npa studio' in capsys.readouterr().out
