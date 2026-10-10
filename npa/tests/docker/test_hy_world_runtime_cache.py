"""Regression tests for the HY-World runtime-fetch cache layout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "docker" / "workbench" / "hy-world" / "hy_world_runtime.sh"


def _bash(
    script: str, *, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
    )


def test_default_runtime_root_is_frozen_before_hf_cache_isolated(
    tmp_path: Path,
) -> None:
    """Normal image defaults cannot nest source/venv after HF_HOME is repointed."""

    shared_hf = tmp_path / "shared-hf"
    shared_ref = shared_hf / "hub" / "models--unrelated--model" / "refs"
    shared_ref.mkdir(parents=True)
    (shared_ref / "main").write_text("unrelated-main\n", encoding="utf-8")
    result_path = tmp_path / "result.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "configure_runtime_hf_cache",
                'model_cache="$(hub_cache_directory "$HY_WORLD_MODEL")"',
                'mkdir -p "$model_cache/snapshots/$HY_WORLD_MODEL_REF"',
                'register_hub_main_ref "$HY_WORLD_MODEL" "$HY_WORLD_MODEL_REF"',
                f'printf "%s\\n%s\\n%s\\n" "$(cache_root)" "$HF_HOME" "$(cat "$model_cache/refs/main")" > {result_path}',
                'test "$(source_tree)" = "$(cache_root)/source-$SOURCE_REF"',
            )
        ),
        environment={
            "HF_HOME": str(shared_hf),
            "NPA_HY_WORLD_RUNTIME_CACHE": "",
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    root, isolated_hf, main_ref = result_path.read_text(encoding="utf-8").splitlines()
    assert root == f"{shared_hf}/hy-world/runtime"
    assert isolated_hf.startswith(f"{root}/hf-")
    assert "/hf-" not in root
    assert main_ref == "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"
    assert (shared_ref / "main").read_text(encoding="utf-8") == "unrelated-main\n"


def test_completed_cache_layout_is_checked_for_relocated_console_scripts(
    tmp_path: Path,
) -> None:
    """A legacy .complete cache with an .incomplete script cannot be reused."""

    tree = tmp_path / "runtime" / "source-df9988efb87bfc0f4947eb3889411cf957478b06"
    venv = tree / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    stale = venv / "bin" / "legacy-console"
    stale.write_text(f"#!{tree}.incomplete/.venv/bin/python\n", encoding="utf-8")
    stale.chmod(0o755)

    result = _bash(
        f"source {RUNTIME}; verify_runtime_layout {tree}",
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(tmp_path / "runtime"),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode != 0
    assert "still references staging tree" in result.stderr


def test_ensure_revalidates_an_existing_complete_runtime_before_reuse(
    tmp_path: Path,
) -> None:
    """The complete-cache fast path invokes the layout guard before returning."""

    cache_root = tmp_path / "runtime"
    checked = tmp_path / "checked.txt"
    result = _bash(
        "\n".join(
            (
                f"source {RUNTIME}",
                "health() { :; }",
                "validate_runtime_paths() { :; }",
                "source_is_complete() { return 0; }",
                f'verify_runtime_layout() {{ printf "%s\\n" "$1" > {checked}; }}',
                "ensure",
            )
        ),
        environment={
            "NPA_HY_WORLD_RUNTIME_CACHE": str(cache_root),
            "NPA_HY_WORLD_RUNTIME_LIBRARY": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert checked.read_text(encoding="utf-8").strip() == (
        f"{cache_root}/source-df9988efb87bfc0f4947eb3889411cf957478b06"
    )
