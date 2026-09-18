"""Verify portable studio dispatch without operator configuration or shell interpolation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from npa import studio


def test_init_and_relocated_dispatch_need_no_cloud_config(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[3] / 'npa/src/npa/studio_renderer'
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


def test_installed_renderer_creates_only_selected_media_and_moves_offline(tmp_path, monkeypatch):
    private_config = tmp_path / 'operator-configuration'
    private_config.mkdir()
    (private_config / 'config.yaml').write_text('unrelated-private-configuration')
    monkeypatch.setenv('NPA_CONFIG_DIR', str(private_config))
    original = tmp_path / 'studio'
    source = tmp_path / 'user image.png'
    Image.new('RGB', (64, 64), 'blue').save(source)
    assert studio.run(['init', '--directory', str(original)]) == 0
    registry = original / 'studio.json'
    command = ['--registry', str(registry), 'create', 'demo', '--input-path', str(source),
               '--duration', '7', '--prompt', 'Explain the measured result', '--audience', 'Researchers']
    assert studio.run(command) == 0
    before = registry.read_bytes()
    assert studio.run(command) == 1
    assert registry.read_bytes() == before
    project = original / 'projects/demo'
    asset = json.loads((project / 'assets.json').read_text())['source']
    assert (project / asset['path']).read_bytes() == source.read_bytes()
    assert asset['provenance'] == {'status': 'unattributed'}
    assert str(tmp_path) not in (project / 'assets.json').read_text()
    assert not list(original.rglob('config.yaml'))
    assert 'projects/' in (original / '.gitignore').read_text()
    moved = tmp_path / 'another user' / 'moved studio'
    shutil.move(original, moved)
    source.unlink()
    assert studio.run(['--registry', str(moved / 'studio.json'), 'demo', 'scenes']) == 0


@pytest.mark.parametrize('name', ['../outside', 'create', 'Bad Name', '/absolute'])
def test_create_rejects_invalid_names_without_registering_a_project(tmp_path, name):
    original = tmp_path / 'studio'
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16)).save(source)
    assert studio.run(['init', '--directory', str(original)]) == 0
    with pytest.raises(SystemExit):
        studio.run(['--registry', str(original / 'studio.json'), 'create', name,
                    '--input-path', str(source)])
    assert list((original / 'projects').iterdir()) == []


def test_failed_media_copy_preserves_registry_and_leaves_no_project(tmp_path, monkeypatch):
    from npa import studio_project

    original = tmp_path / 'studio'
    source = tmp_path / 'source.png'
    Image.new('RGB', (16, 16)).save(source)
    assert studio.run(['init', '--directory', str(original)]) == 0
    registry = original / 'studio.json'
    before = registry.read_bytes()

    def fail_copy(*args):
        raise OSError('media copy failed')

    monkeypatch.setattr(studio_project.shutil, 'copyfile', fail_copy)
    assert studio.run(['--registry', str(registry), 'create', 'demo', '--input-path', str(source)]) == 1
    assert registry.read_bytes() == before
    assert list((original / 'projects').iterdir()) == []
