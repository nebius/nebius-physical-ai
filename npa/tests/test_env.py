from __future__ import annotations

import subprocess
import json
from pathlib import Path
import shlex
import sys

import pytest

from npa.clients.env import load_env_file_script, render_docker_env_file, render_shell_env_file, shell_quote_env_value


def test_shell_quote_env_value_escapes_shell_sensitive_chars() -> None:
    value = "abc$def!`cmd`'\"\\tail"

    assert shell_quote_env_value(value) == "'abc$def!`cmd`'\\''\"\\tail'"


def test_render_shell_env_file_round_trips_through_shell_source(tmp_path) -> None:
    value = "abc$def!`cmd`'\"\\tail"
    env_file = tmp_path / "env"
    env_file.write_text(render_shell_env_file({"S3_SECRET_KEY": value}))

    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"set -a; . {env_file}; set +a; python3 - <<'PY'\n"
            "import os\n"
            "print(os.environ['S3_SECRET_KEY'])\n"
            "PY",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.rstrip("\n") == value


def test_render_docker_env_file_preserves_shell_sensitive_chars_without_quotes() -> None:
    value = "abc$def!`cmd`'\"\\tail"

    rendered = render_docker_env_file({"S3_SECRET_KEY": value})

    assert rendered == f"S3_SECRET_KEY={value}\n"


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("prefix", ["", "true && "])
def test_literal_loader_preserves_credentials_and_never_executes_data(tmp_path, shell, prefix):
    marker = tmp_path / "must-not-exist"
    values = {
        "HF_TOKEN": f"$(touch {shlex.quote(str(marker))})",
        "AWS_SECRET_ACCESS_KEY": f"`touch {shlex.quote(str(marker))}`",
        "FIFTYONE_DATASET_NAME": "  literal '$HOME' \\\" é ; trailing  ",
        "NPA_QUOTED": "'quotes are data'",
        "NPA_EMPTY": "",
        "NPA_EQUALS": "a=b=c",
    }
    env_file = tmp_path / "environment with spaces"
    env_file.write_text("# operator data\n\n" + render_docker_env_file(values).rstrip("\n"))
    script = prefix + load_env_file_script(str(env_file))
    program = "import json,os; print(json.dumps({key:os.environ[key] for key in " + repr(list(values)) + "}))"
    script += "\n" + shlex.join([sys.executable, "-c", program])
    completed = subprocess.run([shell, "-c", "set -e\n" + script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == values
    assert not marker.exists()


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
def test_literal_loader_does_not_bypass_failed_preceding_command(tmp_path, shell):
    marker = tmp_path / "must-not-run"
    env_file = tmp_path / "env"
    env_file.write_text("NPA_LITERAL=value\n")
    # Native login hooks may already have defined the function. Failed venv
    # activation must still prevent both loading credentials and the workload.
    script = "npa_load_env_file() { return 0; }; false && "
    script += load_env_file_script(str(env_file)) + " && touch " + shlex.quote(str(marker))
    completed = subprocess.run([shell, "-c", script], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["not-an-assignment", "BAD-NAME=value", "1NAME=value", "BASH_FUNC_fn%%=value"])
def test_literal_loader_refuses_malformed_entries_without_running_next_command(tmp_path, entry):
    env_file = tmp_path / "env"
    env_file.write_text(entry + "\n")
    marker = tmp_path / "must-not-exist"
    script = load_env_file_script(str(env_file)) + " && touch " + shlex.quote(str(marker))
    completed = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not marker.exists()
    assert entry not in completed.stderr


def test_literal_loader_missing_file_obeys_required_contract(tmp_path):
    missing = str(tmp_path / "missing")
    assert subprocess.run(["/bin/sh", "-c", load_env_file_script(missing, required=False)]).returncode == 0
    assert subprocess.run(["/bin/sh", "-c", load_env_file_script(missing)], capture_output=True).returncode != 0


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
def test_literal_loader_propagates_export_failure_in_conditional_chain(tmp_path, shell):
    env_file = tmp_path / "env"
    env_file.write_text("NPA_READONLY=changed\nNPA_OTHER=value\n")
    marker = tmp_path / "must-not-run"
    script = "readonly NPA_READONLY=original; " + load_env_file_script(str(env_file))
    script += " && touch " + shlex.quote(str(marker))
    completed = subprocess.run([shell, "-c", script], capture_output=True, text=True)
    assert completed.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("value", ["bad\nline", "bad\rline", "bad\0byte"])
def test_docker_env_writer_refuses_unrepresentable_values(value):
    with pytest.raises(ValueError, match="newline or NUL"):
        render_docker_env_file({"HF_TOKEN": value})


def test_all_shared_vm_readers_use_literal_loader():
    source = Path(__file__).resolve().parents[1] / "src/npa"
    for relative in (
        "cli/cosmos/__init__.py", "cli/fiftyone/__init__.py",
        "cli/genesis/__init__.py", "cli/workbench/lerobot.py",
        "workflows/distill_two_vm.py",
        "deploy/terraform/cloud_init.yaml.tpl",
    ):
        text = (source / relative).read_text()
        for path in ("/opt/lerobot/.env", "/etc/npa-fiftyone/env"):
            assert f"source {path}" not in text
            assert f". {path}" not in text
