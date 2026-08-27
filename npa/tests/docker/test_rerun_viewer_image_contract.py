"""Static contract checks for the Rerun SkyPilot visualization worker."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "rerun-viewer" / "Dockerfile"
ENTRYPOINT = ROOT / "npa" / "docker" / "workbench" / "rerun-viewer" / "entrypoint.sh"


def test_rerun_viewer_satisfies_skypilot_non_root_setup_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "@sha256:" in text
    assert "apt-get upgrade -y --no-install-recommends" in text
    for package in ("openssh-server", "rsync", "sudo"):
        assert package in text
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "USER ubuntu" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text
    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"'
        in text
    )


def test_rerun_viewer_entrypoint_preserves_orchestrator_argv() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert (
        'ENTRYPOINT ["/opt/npa/docker/workbench/rerun-viewer/entrypoint.sh"]'
        in dockerfile
    )
    assert 'exec "$@"' in entrypoint
    assert (
        "rerun --serve-web --web-viewer --bind 0.0.0.0 "
        "--web-viewer-port 9090 --port 9876"
    ) in entrypoint


def test_rerun_viewer_installs_exact_sdk_in_owned_venv() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "python -m venv /opt/rerun/venv" in text
    assert '"rerun-sdk==${RERUN_SDK_VERSION}"' in text
    assert "--no-cache-dir" in text
