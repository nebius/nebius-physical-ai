"""Regression tests for Linux argv-safe Isaac Job payload transport."""

from __future__ import annotations

import random
import string
import subprocess

from npa.workflows.sim2real.isaac_job_payload import (
    compressed_bash_launch,
    decode_compressed_bash_args,
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
