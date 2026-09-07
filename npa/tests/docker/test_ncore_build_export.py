"""Exercise the local NCore build command without invoking a builder."""

import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def test_scratch_root_registers_retained_cpython_shared_library():
    dockerfile = (ROOT / "npa/docker/workbench/ncore/Dockerfile").read_text()
    configure = dockerfile.index("/public-root/etc/ld.so.conf")
    refresh = dockerfile.index("ldconfig -r /public-root")
    probe = dockerfile.index("chroot --userspec=1000:1000")
    assert configure < refresh < probe
    assert "'/usr/local/lib'" in dockerfile[configure - 80:configure]


def test_oci_export_preserves_attestations_without_pushing(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    capture = tmp_path / "docker-argv.json"
    docker = tools / "docker"
    docker.write_text(
        f"#!{ROOT / 'npa/.venv/bin/python'}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['DOCKER_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    docker.chmod(0o755)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = tmp_path / "candidate.oci.tar"
    metadata = tmp_path / "build-metadata.json"
    result = subprocess.run(
        [
            "bash", str(ROOT / "npa/docker/workbench/ncore/build.sh"),
            "--source-sha", sha,
            "--oci-output", str(archive),
            "--metadata-file", str(metadata),
            "--builder", "test-builder",
        ],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}", "DOCKER_ARGV": str(capture)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())
    assert "--load" not in argv and "--push" not in argv
    assert "--provenance=mode=max" in argv and "--sbom=true" in argv
    assert argv[argv.index("--output") + 1] == f"type=oci,dest={archive}"
    assert argv[argv.index("--metadata-file") + 1] == str(metadata)
    assert argv[argv.index("--builder") + 1] == "test-builder"
    assert argv[argv.index("--build-arg") + 1] == f"SOURCE_SHA={sha}"
