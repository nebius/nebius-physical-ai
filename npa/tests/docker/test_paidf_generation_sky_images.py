from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from npa.workflows.paidf_upstream import upstream_contract


ROOT = Path(__file__).resolve().parents[3]
CASES = (
    ("paidf-image-edit-sky", "image-attribute-augmentation"),
    ("paidf-event-video-sky", "event-video-generation"),
)


@pytest.mark.parametrize("image,variant", CASES)
def test_generation_wrapper_preserves_reviewed_parent(image: str, variant: str) -> None:
    recipe = ROOT / "npa/docker/workbench" / image / "Dockerfile"
    source = recipe.read_text()
    parent = next(
        item
        for item in upstream_contract(variant)["npa_integration"]["components"][
            "reference_runtime_images"
        ]
        if item.startswith("docker.io/vllm/")
    )
    assert f"FROM {parent}" in source
    assert "ARG BASE_IMAGE" not in source
    for prerequisite in ("openssh-server", "rsync", "sudo", "ssh-keygen -A"):
        assert prerequisite in source
    assert "USER ubuntu" in source
    assert "PasswordAuthentication no" in source
    assert "PermitRootLogin no" in source
    assert "rm -f /etc/ssh/ssh_host_*" in source
    # A source recipe is not a completed bootstrap attestation.
    assert "org.nebius.npa.skypilot-bootstrap-contract" not in source
    spec = yaml.safe_load(
        (ROOT / "npa/workflows/workbench/npa-workflows" / f"paidf-{variant}.yaml")
        .read_text()
    )
    assert spec["config"]["generation_image"].startswith(
        f"registry.example.invalid/npa-{image}@sha256:"
    )
    assert any(
        profile.get("image") == "{{config.generation_image}}"
        for profile in spec["resources"].values()
    )


@pytest.mark.parametrize("image,variant", CASES)
def test_generation_entrypoint_preserves_arguments_and_exit(image: str, variant: str) -> None:
    entrypoint = ROOT / "npa/docker/workbench" / image / "entrypoint.sh"
    result = subprocess.run(
        ["/bin/sh", str(entrypoint), "/bin/sh", "-c", 'printf "%s" "$1"; exit 17',
         "command", "two words"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == "two words"
    assert result.returncode == 17
