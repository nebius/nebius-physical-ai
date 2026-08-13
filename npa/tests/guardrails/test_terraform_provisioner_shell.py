"""The agent Terraform embeds bash in HCL; keep it syntactically valid.

`null_resource.wait_for_cloud_init` runs a multi-branch bash script through
`local-exec`. A syntax error there only surfaces at deploy time, after a VM has
been created, and Terraform reports it by dumping the whole script body.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MAIN_TF = REPO_ROOT / "npa" / "src" / "npa" / "deploy" / "terraform" / "main.tf"


def _provisioner_scripts() -> list[tuple[str, str]]:
    """Return ``(resource, script)`` for every local-exec heredoc in main.tf."""
    text = MAIN_TF.read_text(encoding="utf-8")
    scripts: list[tuple[str, str]] = []
    for match in re.finditer(r'resource\s+"null_resource"\s+"(\w+)"', text):
        name = match.group(1)
        tail = text[match.end() :]
        marker = "command     = <<-EOT"
        if marker not in tail:
            continue
        body = tail[tail.index(marker) + len(marker) :]
        scripts.append((name, body[: body.index("\n    EOT")]))
    return scripts


def _as_shell(script: str) -> str:
    """Replace Terraform interpolations with literals, honoring `$${` escapes."""
    escaped = script.replace("$${", "\x00")
    resolved = re.sub(r"\$\{[^}]*\}", "terraform-value", escaped)
    return resolved.replace("\x00", "${")


def test_local_exec_scripts_are_valid_bash() -> None:
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is present on CI and dev machines
        pytest.skip("bash is not available")
    scripts = _provisioner_scripts()
    assert scripts, "no local-exec scripts found; did main.tf move?"
    for name, script in scripts:
        with tempfile.NamedTemporaryFile("w", suffix=".sh") as handle:
            handle.write(_as_shell(script))
            handle.flush()
            result = subprocess.run(
                [bash, "-n", handle.name], capture_output=True, text=True, timeout=30
            )
        assert result.returncode == 0, f"{name} is not valid bash: {result.stderr}"


def test_ssh_wait_reports_progress_and_bounds_its_window() -> None:
    """A silent 15-minute wait reads as a hang; keep the window and the progress."""
    scripts = dict(_provisioner_scripts())
    script = scripts["wait_for_cloud_init"]

    assert "still waiting (attempt" in script
    assert "ConnectTimeout=5" in script
    # The window is bounded by attempts x sleep; keep both explicit and small.
    assert "attempts=30" in script
    # A port that never opens is reported as reachability, not as a key problem.
    assert "never opened from this machine" in script
    assert "never authenticated" in script
