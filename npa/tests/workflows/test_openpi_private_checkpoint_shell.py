"""Execute the rendered adapted-server shell across its Python quoting boundary."""

import json
import os
import shlex
import subprocess
import sys

import pytest

from npa.workflows.byof import openpi_service, openpi_service_lifecycle


def _private_server():
    container = {
        "env": [{"name": "OPENPI_CHECKPOINT_URI", "value": "s3://private/checkpoint"}],
        "command": [
            "/bin/bash",
            "-lc",
            openpi_service._policy_server_shell("pi05_droid_jointpos_polaris", "auth"),
        ],
    }
    return {
        "deployment": {
            "metadata": {"labels": {"app.kubernetes.io/managed-by": "npa"}},
            "spec": {"template": {"spec": {"containers": [container]}}},
        }
    }, container


def _credentials(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.invalid")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "redacted")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "redacted")


def _python_probe(tmp_path):
    path = tmp_path / "python-probe"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys,types\n"
        "if len(sys.argv)>2 and sys.argv[1]=='-c':\n"
        " program=sys.argv[2]\n"
        " code=compile(program,'<rendered-server-python>','exec')\n"
        " with pathlib.Path(os.environ['PROBE_RECORD']).open('a') as out:\n"
        "  out.write(json.dumps(program)+'\\n')\n"
        " if 'download.maybe_download' in program:\n"
        "  def download_checkpoint(uri,token):\n"
        "   assert uri=='/workspace/openpi-server-cache/adapted-checkpoint'\n"
        "   assert token=='anon'\n"
        "   return uri\n"
        "  module=types.ModuleType('openpi.shared')\n"
        "  module.download=types.SimpleNamespace(maybe_download=download_checkpoint)\n"
        "  sys.modules['openpi.shared']=module\n"
        "  exec(code,{})\n"
    )
    path.chmod(0o700)
    return path


def test_private_checkpoint_executes_valid_rendered_python(tmp_path, monkeypatch):
    _credentials(monkeypatch)
    manifests, container = _private_server()
    openpi_service_lifecycle._patch_private_s3_checkpoint(
        manifests, "s3://private/checkpoint", run_id="checkpoint", namespace="default"
    )
    openpi_service_lifecycle._inject_checkpoint_probe(manifests)
    shell = container["command"][2].replace(
        "/opt/venv/bin/python", shlex.quote(str(_python_probe(tmp_path)))
    )
    record = tmp_path / "programs.jsonl"
    environment = dict(os.environ)
    environment.update(
        {row["name"]: row["value"] for row in container["env"] if "value" in row}
    )
    assert environment["OPENPI_CHECKPOINT_URI"] == (
        "/workspace/openpi-server-cache/adapted-checkpoint"
    )
    assert environment["NPA_OPENPI_CHECKPOINT_SOURCE_URI"] == (
        "s3://private/checkpoint"
    )
    environment["PROBE_RECORD"] = str(record)
    result = subprocess.run(
        ["/bin/bash", "-c", shell], env=environment, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    programs = [json.loads(line) for line in record.read_text().splitlines()]
    assert sum("download.maybe_download" in program for program in programs) == 1
    assert any("adapted checkpoint prefix is empty" in program for program in programs)
    assert any("npa-openpi-loaded-checkpoint.json" in program for program in programs)


@pytest.mark.parametrize(
    "binding", [[], [{"name": "OPENPI_CHECKPOINT_URI", "value": "wrong"}]]
)
def test_private_checkpoint_rejects_missing_or_changed_binding(monkeypatch, binding):
    _credentials(monkeypatch)
    manifests, container = _private_server()
    container["env"] = binding
    with pytest.raises(openpi_service.OpenPIServiceError, match="environment binding"):
        openpi_service_lifecycle._patch_private_s3_checkpoint(
            manifests,
            "s3://private/checkpoint",
            run_id="checkpoint",
            namespace="default",
        )
