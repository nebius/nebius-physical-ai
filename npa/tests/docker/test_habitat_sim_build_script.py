"""Exercise the local-only, full-SHA Habitat image build boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa/docker/workbench/habitat-sim/build.sh"


def _run(
    *args: object, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), *(str(arg) for arg in args)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_builder_requires_one_new_owner_selected_oci_path(tmp_path: Path) -> None:
    missing = _run()
    assert missing.returncode == 2
    assert "usage:" in missing.stderr

    output = tmp_path / "existing.oci.tar"
    output.write_bytes(b"owner data")
    existing = _run(output)
    assert existing.returncode == 2
    assert "refusing to overwrite" in existing.stderr
    assert output.read_bytes() == b"owner data"


def test_builder_passes_exact_git_sha_to_local_attested_oci_export(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker-argv"
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$DOCKER_ARGV_LOG"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_ARGV_LOG": str(log),
    }
    output = tmp_path / "candidate.oci.tar"

    result = _run(output, env=env)

    assert result.returncode == 0, result.stderr
    argv = log.read_text(encoding="utf-8").splitlines()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert argv[:2] == ["buildx", "build"]
    assert f"NPA_SOURCE_SHA={revision}" in argv
    assert f"type=oci,dest={output}" in argv
    assert "--provenance=mode=max" in argv
    assert "--sbom=true" in argv
    assert "--push" not in argv and "--load" not in argv
    assert argv[-1] == "npa"
