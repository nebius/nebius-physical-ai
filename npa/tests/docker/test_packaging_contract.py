"""Enforce workbench container packaging contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"
WORKBENCH_DOCKER = ROOT / "npa" / "docker" / "workbench"


def _load_contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _final_user(dockerfile_text: str) -> str | None:
    users = re.findall(r"(?im)^\s*USER\s+(.+?)\s*$", dockerfile_text)
    if not users:
        return None
    return users[-1].strip().strip("\"'")


def _entrypoints(dockerfile_text: str) -> list[str]:
    return re.findall(r"(?im)^\s*ENTRYPOINT\s+(.+?)\s*$", dockerfile_text)


def _cmds(dockerfile_text: str) -> list[str]:
    return re.findall(r"(?im)^\s*CMD\s+(.+?)\s*$", dockerfile_text)


def _runtime_commands(dockerfile_text: str) -> list[str]:
    """ENTRYPOINT preferred; bare CMD is accepted for service images."""
    return _entrypoints(dockerfile_text) or _cmds(dockerfile_text)


def _exposes(dockerfile_text: str) -> list[int]:
    ports: list[int] = []
    for match in re.finditer(r"(?im)^\s*EXPOSE\s+(.+?)\s*$", dockerfile_text):
        for token in match.group(1).split():
            if token.isdigit():
                ports.append(int(token))
    return ports


def test_packaging_contract_file_exists() -> None:
    assert CONTRACT_PATH.is_file()
    contract = _load_contract()
    assert contract["version"] == 1
    assert "service" in contract["tiers"]
    assert "job" in contract["tiers"]
    assert "interactive" in contract["tiers"]
    assert contract["images"]


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_image_matches_packaging_contract(image_name: str) -> None:
    contract = _load_contract()
    entry = contract["images"][image_name]
    dockerfile = WORKBENCH_DOCKER / entry["dockerfile"]
    assert dockerfile.is_file(), f"missing Dockerfile for {image_name}: {dockerfile}"
    text = dockerfile.read_text(encoding="utf-8")
    tier_name = entry["tier"]
    tier = contract["tiers"][tier_name]

    if contract["security"]["require_non_root_user"]:
        expected_user = entry.get("final_user")
        final_user = _final_user(text)
        if expected_user:
            assert final_user == expected_user, f"{image_name}: expected USER {expected_user}, got {final_user}"
        else:
            allowed = set(contract["security"]["allowed_final_users"])
            assert final_user is not None, f"{image_name}: missing final USER"
            # Allow USER $NPA_RUNTIME_USER style only when default is documented non-root.
            if final_user.startswith("$"):
                assert "ubuntu" in text or "NPA_RUNTIME_USER" in text
            else:
                assert final_user in allowed, f"{image_name}: final USER {final_user!r} not in {allowed}"

    runtime_cmds = _runtime_commands(text)
    if tier.get("entrypoint_must_not_be_bash"):
        assert runtime_cmds, (
            f"{image_name}: {tier_name} images must declare ENTRYPOINT or CMD"
        )
        joined = " ".join(runtime_cmds).lower()
        assert "/bin/bash" not in joined and '["bash"]' not in joined, (
            f"{image_name}: {tier_name} ENTRYPOINT/CMD must not be /bin/bash"
        )

    declared_ports = entry.get("ports") or []
    if declared_ports:
        exposed = set(_exposes(text))
        for port in declared_ports:
            assert port in exposed, f"{image_name}: missing EXPOSE {port}"

    for pattern in contract["security"].get("secret_patterns", []):
        assert re.search(pattern, text) is None, f"{image_name}: Dockerfile matches secret pattern {pattern}"


def test_packaging_doc_exists() -> None:
    doc = ROOT / "docs" / "workbench" / "container-packaging.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "Packaging tiers" in text
    assert "Security baseline" in text
    assert "packaging-contract.yaml" in text
