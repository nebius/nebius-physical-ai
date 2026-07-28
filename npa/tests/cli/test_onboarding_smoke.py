"""Guards for the documented first-time-user onboarding path.

These tests defend the copy-pasteable quickstart so docs that "look right"
cannot silently rot: the setup guidance must stay placeholder-only (public
hygiene), and the advertised first real success must keep working offline.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from typer.testing import CliRunner

from npa.cli.main import app
from npa.workbench.vlm_eval import DEFAULT_MODEL, DEFAULT_SAMPLE_BENCHMARK_PATH


runner = CliRunner()

# Matches any dotted-quad IPv4 literal, e.g. 203.0.113.10.
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_setup_guidance_points_to_single_configure_entrypoint() -> None:
    """Onboarding guidance should not require separate nebius CLI steps first."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        lowered = result.output.lower()
        assert "npa configure --interactive" in result.output
        assert "nebius profile create" not in lowered
        assert "get-access-token" not in lowered


def test_setup_guidance_contains_no_raw_ip_address() -> None:
    """Setup guidance must use placeholders, never a literal host/IP."""
    for command in ("configure", "init"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        match = _IPV4.search(result.output)
        assert match is None, (
            f"`npa {command}` guidance leaks a literal IP {match.group(0)!r}; "
            "use a placeholder such as <your-byovm-host> instead."
        )
        assert "<your-byovm-host>" in result.output


def test_npa_version_emits_no_syntax_warning(tmp_path) -> None:
    """The README verify step `npa --version` must be warning-clean.

    It once printed ``SyntaxWarning: invalid escape sequence '\\s'`` from an
    embedded f-string in ``npa/src/npa/cli/agent.py`` (which `npa --version`
    imports). Reproduce the first-run experience in a subprocess with a fresh
    bytecode cache so every ``npa`` module compiles from source and any escape
    warning would actually surface, then assert none originates from the
    package. Runs the module entrypoint directly so it works without ``npa``
    being on ``PATH``.
    """
    env = dict(os.environ)
    # Fresh, isolated bytecode cache -> our modules recompile from source, so a
    # stray invalid escape re-appears here instead of being masked by a warm
    # ``__pycache__``. Show every SyntaxWarning rather than the default "once".
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    env["PYTHONWARNINGS"] = "always::SyntaxWarning"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv = ['npa', '--version']; "
            "from npa.cli.main import app_entry; app_entry()",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    # A SyntaxWarning attributable to the shipped package (path under src/npa/);
    # third-party dependency warnings live under site-packages and are ignored.
    package_warnings = [
        line
        for line in combined.splitlines()
        if "SyntaxWarning" in line and re.search(r"[\\/]src[\\/]npa[\\/]", line)
    ]
    assert not package_warnings, "npa emitted SyntaxWarning(s):\n" + "\n".join(package_warnings)
    assert "invalid escape sequence" not in combined, combined
    assert proc.returncode == 0, combined
    assert "npa" in proc.stdout


def test_npa_version_fast_path_skips_heavy_imports() -> None:
    """``npa --version`` must not import the full command tree.

    The console-script entry (``npa.cli.entry:main``) answers a bare version
    request before importing ``npa.cli.main``, which transitively pulls in heavy
    dependencies (boto3, paramiko, rerun, numpy). Guard that this stays fast so
    ``npa --version`` does not regress back to a multi-hundred-millisecond
    import. Runs in a subprocess so the check sees a clean interpreter.
    """
    probe = (
        "import sys; sys.argv = ['npa', '--version']; "
        "from npa.cli.entry import main; main(); "
        "heavy = [m for m in "
        "('npa.cli.main', 'boto3', 'paramiko', 'rerun', 'numpy') "
        "if m in sys.modules]; "
        "print('HEAVY:' + ','.join(heavy))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "npa " in proc.stdout
    assert "HEAVY:\n" in proc.stdout or proc.stdout.rstrip().endswith("HEAVY:"), (
        "npa --version fast path imported heavy modules: " + proc.stdout
    )


def test_quickstart_first_success_fixture_is_packaged() -> None:
    """The fixture the quickstart points at must ship inside the package."""
    assert DEFAULT_SAMPLE_BENCHMARK_PATH.exists(), (
        "Quickstart first-success benchmark fixture is missing: "
        f"{DEFAULT_SAMPLE_BENCHMARK_PATH}"
    )


def test_quickstart_benchmark_command_produces_real_result(tmp_path) -> None:
    """Run the exact documented first-success command end to end, offline."""
    output_path = tmp_path / "vlm-eval-benchmark.json"

    result = runner.invoke(
        app,
        [
            "workbench",
            "vlm-eval",
            "benchmark",
            "--dataset",
            str(DEFAULT_SAMPLE_BENCHMARK_PATH),
            "--output",
            str(output_path),
            "--backend",
            "stub",
            "--thresholds",
            "0.5,0.8,0.9",
            "--rubrics",
            "default,strict",
            "--models",
            DEFAULT_MODEL,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # A real scoring pass over the shipped labeled rollout set, no GPU or creds.
    assert payload["best_config"]["metrics"]["accuracy"] == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["item_count"] == 4
