from __future__ import annotations

import subprocess

from npa.clients.env import render_docker_env_file, render_shell_env_file, shell_quote_env_value


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
