"""Regression tests for Linux argv-safe Isaac Job payload transport."""

from __future__ import annotations

import os
import random
import string
import subprocess
from pathlib import Path

import pytest

from npa.workflows.sim2real.isaac_job_payload import (
    compressed_bash_launch,
    decode_compressed_bash_args,
    embedded_base64_file_block,
    execute_manifest_container_inline,
)


def test_large_generated_script_round_trips_below_per_argument_limit() -> None:
    rng = random.Random(7)
    script = "".join(
        rng.choice(string.ascii_letters + string.digits) for _ in range(300_000)
    )
    command, args = compressed_bash_launch(script)

    assert command == ["/bin/bash", "-lc"]
    assert len(args) > 3
    assert max(map(len, args)) < 128 * 1024
    assert decode_compressed_bash_args(args) == script


def test_compressed_launch_executes_reconstructed_script() -> None:
    command, args = compressed_bash_launch("printf ISAAC_PAYLOAD_EXEC_OK")
    result = subprocess.run(
        [*command, *args],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == "ISAAC_PAYLOAD_EXEC_OK"


def test_embedded_base64_file_block_avoids_large_process_argument(
    tmp_path: Path,
) -> None:
    payload = "scenario-record-\n" * 200_000
    destination = tmp_path / "scenarios.jsonl"
    script = tmp_path / "materialize.sh"
    block = embedded_base64_file_block(
        payload,
        destination=str(destination),
        marker="NPA_TEST_SCENARIOS_B64",
    )
    script.write_text("set -euo pipefail\n" + block, encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(script)], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0
    assert destination.read_text(encoding="utf-8") == payload
    assert "--payload" not in block
    assert "base64 --decode >" in block


def test_embedded_base64_file_block_rejects_unsafe_marker() -> None:
    with pytest.raises(ValueError, match="uppercase shell token"):
        embedded_base64_file_block("x", destination="/tmp/x", marker="bad-marker")


def test_inline_payload_requires_and_attests_exact_workflow_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "b" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    monkeypatch.setenv("ACCEPT_EULA", "Y")
    command, args = compressed_bash_launch('test "$INLINE_MARKER" = yes')
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "image": image,
                            "command": command,
                            "args": args,
                            "env": [{"name": "INLINE_MARKER", "value": "yes"}],
                        }
                    ]
                }
            }
        }
    }

    proof = execute_manifest_container_inline(manifest)

    assert proof["mode"] == "npa_workflow_skypilot_task"
    assert proof["image"] == image


def test_inline_large_payload_avoids_process_wide_argv_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "f" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    monkeypatch.setenv("ACCEPT_EULA", "Y")
    rng = random.Random(11)
    script = "# " + "".join(
        rng.choice(string.ascii_letters + string.digits) for _ in range(2_500_000)
    )
    command, args = compressed_bash_launch(script)
    assert sum(map(len, [*command, *args])) > 2_000_000
    observed: dict[str, object] = {}

    def _run(argv: list[str], *, env: dict[str, str], check: bool) -> None:
        observed["argv"] = argv
        observed["script_path"] = argv[-1]
        observed["script"] = Path(argv[-1]).read_text(encoding="utf-8")
        assert env["ACCEPT_EULA"] == "Y"
        assert check is True

    monkeypatch.setattr(subprocess, "run", _run)
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"image": image, "command": command, "args": args}]
                }
            }
        }
    }

    execute_manifest_container_inline(manifest)

    assert observed["argv"] == [
        "/bin/bash",
        "-lc",
        'exec /bin/bash "$1"',
        "npa-isaac-payload",
        observed["script_path"],
    ]
    assert observed["script"] == script
    assert not Path(str(observed["script_path"])).exists()


def test_inline_payload_removes_temp_script_without_masking_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "0" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    command, args = compressed_bash_launch("exit 17")
    observed: dict[str, str] = {}

    def _fail(argv: list[str], *, env: dict[str, str], check: bool) -> None:
        observed["script_path"] = argv[-1]
        assert Path(argv[-1]).exists()
        raise subprocess.CalledProcessError(17, argv)

    monkeypatch.setattr(subprocess, "run", _fail)
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"image": image, "command": command, "args": args}]
                }
            }
        }
    }

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        execute_manifest_container_inline(manifest)

    assert exc_info.value.returncode == 17
    assert not Path(observed["script_path"]).exists()


def test_inline_cleanup_error_does_not_mask_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "1" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    command, args = compressed_bash_launch("exit 23")
    observed: dict[str, str] = {}

    def _fail(argv: list[str], *, env: dict[str, str], check: bool) -> None:
        observed["script_path"] = argv[-1]
        raise subprocess.CalledProcessError(23, argv)

    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"image": image, "command": command, "args": args}]
                }
            }
        }
    }
    with monkeypatch.context() as context:
        context.setattr(subprocess, "run", _fail)
        context.setattr(
            Path,
            "unlink",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("busy")),
        )
        context.setattr(
            os,
            "write",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("closed")),
        )
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            execute_manifest_container_inline(manifest)

    assert exc_info.value.returncode == 23
    Path(observed["script_path"]).unlink()


def test_inline_noncompressed_payload_preserves_original_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "2" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    observed: dict[str, object] = {}

    def _run(argv: list[str], *, env: dict[str, str], check: bool) -> None:
        observed["argv"] = argv
        assert check is True

    monkeypatch.setattr(subprocess, "run", _run)
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "image": image,
                            "command": ["/custom/entrypoint"],
                            "args": ["--flag", "value"],
                        }
                    ]
                }
            }
        }
    }

    execute_manifest_container_inline(manifest)

    assert observed["argv"] == ["/custom/entrypoint", "--flag", "value"]


def test_inline_payload_uses_silent_default_acceptance_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "c" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    monkeypatch.delenv("ACCEPT_EULA", raising=False)
    command, args = compressed_bash_launch("exit 99")
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{"image": image, "command": command, "args": args}]
                }
            }
        }
    }

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        execute_manifest_container_inline(manifest)
    assert exc_info.value.returncode == 99


@pytest.mark.parametrize("value", ["", "no", "FALSE"])
def test_inline_payload_refuses_explicit_opt_out_before_execution(
    monkeypatch: pytest.MonkeyPatch, mocker, value: str
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "d" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    monkeypatch.setenv("ACCEPT_EULA", value)
    run = mocker.patch("npa.workflows.sim2real.isaac_job_payload.subprocess.run")
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"image": image, "command": ["bash"], "args": ["-lc", "true"]}
                    ]
                }
            }
        }
    }

    with pytest.raises(RuntimeError, match="explicitly opted out"):
        execute_manifest_container_inline(manifest)
    run.assert_not_called()


def test_inline_payload_rejects_invalid_acceptance_before_execution(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "e" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    monkeypatch.setenv("ACCEPT_EULA", "maybe")
    run = mocker.patch("npa.workflows.sim2real.isaac_job_payload.subprocess.run")
    manifest = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"image": image, "command": ["bash"], "args": ["-lc", "true"]}
                    ]
                }
            }
        }
    }

    with pytest.raises(ValueError, match="Invalid ACCEPT_EULA"):
        execute_manifest_container_inline(manifest)
    run.assert_not_called()
