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
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text


def test_fiftyone_entrypoint_executes_the_kubernetes_command() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert (
        'ENTRYPOINT ["/opt/npa/docker/workbench/fiftyone/entrypoint.sh"]' in dockerfile
    )
    assert 'exec "$@"' in entrypoint
    assert 'ENTRYPOINT ["/bin/bash"]' not in dockerfile


def test_fiftyone_keeps_bundled_database_and_brain_smoke() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "MONGODB_VERSION=" in text
    assert (
        'cp "/tmp/mongodb-linux-x86_64-ubuntu2204-${MONGODB_VERSION}/bin/mongod"'
        in text
    )
    assert "smoke_functional.py" in text


def test_fiftyone_pins_native_lerobot_temporal_release() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    smoke = (DOCKERFILE.parent / "smoke_env.py").read_text(encoding="utf-8")

    assert "ARG FIFTYONE_VERSION=1.22.0" in dockerfile
    assert '"fiftyone[multimodal]==${FIFTYONE_VERSION}"' in dockerfile
    assert 'getattr(fo.types, "LeRobotDataset", None)' in smoke
    assert "TemporalTag" in smoke


def test_fiftyone_pins_graphql_core_compatible_with_its_strawberry_release() -> None:
    """A resolver update must not recreate the first-import failure in the image."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    pin = (
        "graphql-core @ https://files.pythonhosted.org/packages/1f/b4/"
        "b52a324b0b297ddc49b8ea31a51d39fae63132f694a267b9381d94e2e0af/"
        "graphql_core-3.2.13-py3-none-any.whl#sha256="
        "b0eb04f2c31556b2310a77c8fb53c74e8b56570f8fea594d89c6ef7827dbb497"
    )
    assert pin in dockerfile
    assert dockerfile.count('"graphql-core @ ') == 1


def test_fiftyone_retains_a_root_owned_dependency_correction_receipt() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    validation = (DOCKERFILE.parent / "validation_checks.py").read_text(
        encoding="utf-8"
    )

    receipt = "/usr/share/doc/npa-fiftyone/dependency-source-correction.json"
    assert (
        "install -d -m 0755 -o root -g root /usr/share/doc/npa-fiftyone" in dockerfile
    )
    assert "sudo -n install -o root -g root -m 0644" not in dockerfile
    root_receipt_layer = (
        "USER root\n\n"
        "RUN install -o root -g root -m 0644 \\\n"
        "      /opt/fiftyone/dependency-source-correction.json \\\n"
        "      /usr/share/doc/npa-fiftyone/dependency-source-correction.json"
    )
    assert root_receipt_layer in dockerfile
    assert (
        "USER ubuntu\n\nRUN curl --fail --location --silent --show-error" in dockerfile
    )
    assert receipt in dockerfile
    assert "dependency-source-correction" in validation
    assert "Correction receipt is mutable" in validation
    assert 'find_spec("skimage.data._fetchers")' in validation
    assert 'module_sha256 == receipt["sanitized_sha256"]' in validation
