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


def _strip_comments(dockerfile_text: str) -> str:
    """Return only the Dockerfile's instructions, with comment lines removed."""

    return "\n".join(
        line for line in dockerfile_text.splitlines() if not line.lstrip().startswith("#")
    )


def _bakes_omniverse(dockerfile_text: str, markers: list[str]) -> bool:
    """Whether a Dockerfile's INSTRUCTIONS bake NVIDIA Omniverse Kit (Isaac Sim).

    Markers are matched against instructions only. A comment that merely names
    Isaac Sim (isaac-lab/Dockerfile carries a ``# Tag: nvcr.io/nvidia/isaac-lab``
    header) bakes nothing, so matching raw text would let prose force an image to
    ``restricted``.
    """

    instructions = _strip_comments(dockerfile_text)
    return any(marker in instructions for marker in markers)


def _base_image_refs(dockerfile_text: str) -> list[str]:
    """Image refs this Dockerfile builds on: ``FROM`` targets plus ``ARG *BASE*``
    defaults (most workbench images parameterize their parent as
    ``ARG BASE_IMAGE=<ref>`` and then ``FROM ${BASE_IMAGE}``)."""
    refs = [
        match.group(1)
        for match in re.finditer(r"(?im)^\s*FROM\s+(?:--\S+\s+)*(\S+)", dockerfile_text)
    ]
    refs.extend(
        match.group(1)
        for match in re.finditer(r"(?im)^\s*ARG\s+\w*BASE\w*=(\S+)", dockerfile_text)
    )
    return refs


def _workbench_parents(dockerfile_text: str, contract_images: dict) -> set[str]:
    """Contract image names this Dockerfile inherits from (``npa-<name>`` refs)."""
    parents: set[str] = set()
    for ref in _base_image_refs(dockerfile_text):
        name = ref.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
        if not name.startswith("npa-"):
            continue
        parent = name[len("npa-") :]
        if parent in contract_images:
            parents.add(parent)
    return parents


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


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_image_declares_redistribution_class(image_name: str) -> None:
    contract = _load_contract()
    entry = contract["images"][image_name]
    classes = contract["redistribution"]["classes"]
    cls = entry.get("redistribution")
    assert cls in classes, (
        f"{image_name}: redistribution must be one of {sorted(classes)}, got {cls!r}"
    )


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_omniverse_images_are_restricted(image_name: str) -> None:
    """An image that bakes NVIDIA Omniverse Kit (Isaac Sim) is NOT freely
    redistributable and must be classified ``restricted`` so it is never
    published to a public registry. This detects the marker in the Dockerfile so
    a newly added Omniverse-baking image cannot silently be marked ``public``.
    """
    contract = _load_contract()
    entry = contract["images"][image_name]
    dockerfile = WORKBENCH_DOCKER / entry["dockerfile"]
    text = dockerfile.read_text(encoding="utf-8")
    markers = contract["redistribution"]["omniverse_markers"]
    bakes_omniverse = _bakes_omniverse(text, markers)

    if bakes_omniverse:
        assert entry.get("redistribution") == "restricted", (
            f"{image_name}: Dockerfile bakes Omniverse Kit (Isaac Sim); it must be "
            f"redistribution: restricted (NVIDIA proprietary — public redistribution "
            f"needs an NVIDIA AI Enterprise license)"
        )
        return

    # An image built FROM a restricted workbench image inherits that parent's baked
    # Omniverse Kit even though its own Dockerfile carries no marker (e.g.
    # sonic-mujoco derives from sonic), so it must be restricted too.
    restricted_parents = sorted(
        parent
        for parent in _workbench_parents(text, contract["images"])
        if contract["images"][parent].get("redistribution") == "restricted"
    )
    assert not restricted_parents or entry.get("redistribution") == "restricted", (
        f"{image_name}: builds FROM restricted workbench image(s) "
        f"{restricted_parents} and inherits their baked Omniverse Kit; it must be "
        f"redistribution: restricted"
    )


def test_omniverse_marker_detection_ignores_comments() -> None:
    """Prose naming Isaac Sim must not read as baking it; instructions still must."""

    contract = _load_contract()
    markers = contract["redistribution"]["omniverse_markers"]

    comment_only = (
        "# Base is deliberately NOT nvcr.io/nvidia/isaac-lab, and no isaacsim wheel\n"
        "  # is installed. ISAACSIM_ACCEPT_EULA/OMNI_KIT_ACCEPT_EULA are unset.\n"
        "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04\n"
        "RUN pip install --no-cache-dir numpy\n"
    )
    assert not _bakes_omniverse(comment_only, markers), (
        "comment-only prose must not count as baking Omniverse Kit"
    )

    for instruction in (
        "FROM nvcr.io/nvidia/isaac-lab:2.3.2\n",
        "ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2\n",
        'RUN pip install --no-cache-dir "isaacsim==4.5.0"\n',
        'RUN pip install "isaaclab[isaacsim,all]==2.3.2"\n',
        "ENV ISAACSIM_ACCEPT_EULA=YES OMNI_KIT_ACCEPT_EULA=YES\n",
    ):
        assert _bakes_omniverse("# an explanatory header\n" + instruction, markers), instruction


@pytest.mark.parametrize("image_name", ["groot", "isaac-lab", "sonic"])
def test_first_party_omniverse_images_match_a_marker_in_instructions(image_name: str) -> None:
    """Comment-stripping must not weaken the guard.

    Every image that genuinely bakes Omniverse Kit has to remain detectable from
    its instructions alone, so this pins that the markers do not depend on prose.
    """

    contract = _load_contract()
    entry = contract["images"][image_name]
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    markers = contract["redistribution"]["omniverse_markers"]
    assert _bakes_omniverse(text, markers), (
        f"{image_name}: bakes Omniverse Kit but no marker matches its instructions"
    )


def test_packaging_doc_exists() -> None:
    doc = ROOT / "docs" / "workbench" / "container-packaging.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "Packaging tiers" in text
    assert "Security baseline" in text
    assert "packaging-contract.yaml" in text
