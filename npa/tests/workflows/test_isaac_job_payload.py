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
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "YES")
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


@pytest.mark.parametrize(
    "missing",
    ["OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"],
)
def test_inline_payload_fails_before_kit_without_explicit_operator_eula(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    image = "registry.example/npa/isaac@sha256:" + "c" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    for name in ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"):
        monkeypatch.setenv(name, "" if name == missing else "YES")
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

    with pytest.raises(RuntimeError, match=missing):
        execute_manifest_container_inline(manifest)
