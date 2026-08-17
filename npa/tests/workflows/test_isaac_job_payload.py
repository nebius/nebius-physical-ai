"""Regression tests for Linux argv-safe Isaac Job payload transport."""

from __future__ import annotations

import random
import string
import subprocess

import pytest

from npa.workflows.sim2real.isaac_job_payload import (
    compressed_bash_launch,
    decode_compressed_bash_args,
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
