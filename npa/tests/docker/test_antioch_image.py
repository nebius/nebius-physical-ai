from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "docker" / "workbench" / "antioch"


def test_antioch_image_pins_base_dependencies_and_source_revision() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text()
    requirements = (IMAGE / "requirements.txt").read_text().splitlines()

    assert "python:3.12-slim-bookworm@sha256:" in dockerfile
    assert "ARG NPA_SOURCE_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in dockerfile
    assert 'npa.version="${NPA_VERSION}"' in dockerfile
    assert requirements == [
        "boto3==1.42.91",
        "fastapi==0.136.1",
        "numpy==2.4.4",
        "pyarrow==24.0.0",
        "pydantic==2.13.4",
        "uvicorn==0.38.0",
    ]


def test_antioch_image_keeps_vendor_runtime_out_of_build_layers() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text()

    assert "pip install antioch-sim" not in dockerfile
    assert "COPY auth.json" not in dockerfile
    assert "USER 10001" in dockerfile
    assert "EXPOSE 8789" in dockerfile
