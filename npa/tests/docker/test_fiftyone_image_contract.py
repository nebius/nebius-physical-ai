"""Static contract checks for the FiftyOne SkyPilot job image."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "fiftyone" / "Dockerfile"
ENTRYPOINT = ROOT / "npa" / "docker" / "workbench" / "fiftyone" / "entrypoint.sh"


def test_fiftyone_image_satisfies_skypilot_non_root_setup_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "apt-get upgrade -y --no-install-recommends" in text
    assert "openssh-server" in text
    assert "rsync" in text
    assert "sudo" in text
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "USER ubuntu" in text
    assert "install -d -m 0755 /run/sshd" in text
    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"'
        in text
    )


def test_fiftyone_entrypoint_executes_the_kubernetes_command() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["/opt/npa/docker/workbench/fiftyone/entrypoint.sh"]' in dockerfile
    assert 'exec "$@"' in entrypoint
    assert 'ENTRYPOINT ["/bin/bash"]' not in dockerfile


def test_fiftyone_keeps_bundled_database_and_brain_smoke() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "MONGODB_VERSION=" in text
    assert 'cp "mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/bin/mongod"' in text
    assert "smoke_functional.py" in text
