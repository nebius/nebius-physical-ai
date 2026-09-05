"""Keep the vendor CLI intact while making its worker argv safe."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from npa.deploy.images import RESTRICTED_PUBLICATION_TOOLS
from npa.workflows.paidf_upstream import upstream_contract


ROOT = Path(__file__).resolve().parents[3]
CASES = (
    ("paidf-detection-sky", "detection", "detection-and-tracking-rfdetr", "ubuntu"),
    ("paidf-captioning-sky", "captioning", "captioning", "appuser"),
    ("paidf-visual-qa-sky", "visual_qa", "visual-qa", "appuser"),
    (
        "paidf-attribute-search-sky", "attribute_search",
        "event-and-person-attribute-search", "appuser",
    ),
)


@pytest.mark.parametrize("image,config_key,vendor,user", CASES)
def test_labeling_wrapper_preserves_vendor_boundary(
    image: str, config_key: str, vendor: str, user: str
) -> None:
    source = (ROOT / "npa/docker/workbench" / image / "Dockerfile").read_text()
    parents = upstream_contract("event-video-generation")["npa_integration"][
        "components"
    ]["reference_runtime_images"]
    parent = next(p for p in parents if f"/paidf-{vendor}-service@" in p)
    assert f"FROM {parent}" in source
    assert "ARG BASE_IMAGE" not in source
    assert f"USER {user}" in source
    assert "test -x /app/.venv/bin/main" in source
    assert "/usr/bin/python3 -m venv /opt/npa-venv" in source
    assert f"chown -R {user}:{user} /opt/npa-venv" in source
    assert 'ENV PATH="/opt/npa-venv/bin:$PATH"' in source
    assert "/etc/profile.d/npa-runtime-python.sh" in source
    assert "rm -f /etc/ssh/ssh_host_*" in source
    assert "PasswordAuthentication no" in source
    assert "PermitRootLogin no" in source
    assert "org.nebius.npa.skypilot-bootstrap-contract" not in source
    assert image in RESTRICTED_PUBLICATION_TOOLS
    spec = yaml.safe_load(
        (ROOT / "npa/workflows/workbench/npa-workflows/paidf-event-video-generation.yaml")
        .read_text()
    )
    assert spec["config"][config_key + "_image"] == (
        f"registry.example.invalid/npa-{image}@sha256:" + "0" * 64
    )
    profile = spec["resources"][config_key.replace("_", "-")]
    assert profile["image"] == "{{config." + config_key + "_image}}"
    if config_key != "detection":
        assert "accelerators" not in profile


@pytest.mark.parametrize("image,config_key,vendor,user", CASES)
def test_labeling_entrypoint_does_not_reparse_worker_arguments(
    image: str, config_key: str, vendor: str, user: str
) -> None:
    entrypoint = ROOT / "npa/docker/workbench" / image / "entrypoint.sh"
    literal = "two words; $HOME $(printf unexpected)"
    result = subprocess.run(
        ["/bin/sh", str(entrypoint), "/bin/sh", "-c",
         'printf "%s" "$1"; exit 17', "worker", literal],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == literal
    assert result.returncode == 17
