"""Enforce workbench container packaging contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from pathlib import Path

import pytest
import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS,
    SKYPILOT_BOOTSTRAP_RUNTIME_PROBED_TOOLS,
)

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


def _build_contract_text(dockerfile: Path) -> str:
    """Include copied common installers that materially construct the image."""

    text = dockerfile.read_text(encoding="utf-8")
    parts = [text]
    common = WORKBENCH_DOCKER / "common"
    for script in sorted(common.glob("*.sh")):
        if script.name in text:
            parts.append(script.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _normalize_dockerfile(dockerfile_text: str) -> str:
    """Return a Dockerfile's instructions, one logical instruction per line.

    Two normalisations, both load-bearing for the bake detector:

    * **Comment lines are stripped.** The Isaac Dockerfiles now legitimately discuss
      what they do *not* bake, and prose must never force an image to ``restricted``.
    * **Line continuations are joined.** The sonic Dockerfile used to spread a single
      ``pip install`` over 27 backslash-continued lines, so a per-line matcher would
      have missed the real installer while still tripping on a comment. Inline
      backtick shell comments are dropped for the same reason.
    """

    lines = [
        line
        for line in dockerfile_text.splitlines()
        if not line.lstrip().startswith("#")
    ]
    joined: list[str] = []
    buffer = ""
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += stripped
        joined.append(buffer)
        buffer = ""
    if buffer:
        joined.append(buffer)
    # Drop `# ...` shell comments used inside continued RUN lines.
    return "\n".join(re.sub(r"`#[^`]*`", " ", line) for line in joined)


_SUDOERS_ENTRY_RE = re.compile(
    r"(?P<principal>[A-Za-z_][A-Za-z0-9_.-]*)\s+ALL\s*=\s*"
    r"\(\s*ALL\s*\)\s+NOPASSWD\s*:\s*(?P<commands>ALL|/[^\s'\"\\]+)"
)


def _passwordless_root_grants(text: str) -> set[str]:
    """Return normalized unconditional sudoers grants from executable source."""

    instructions = _normalize_dockerfile(text)
    return {
        f"{match.group('principal')} ALL=(ALL) NOPASSWD:{match.group('commands')}"
        for match in _SUDOERS_ENTRY_RE.finditer(instructions)
    }


def _validate_structured_passwordless_root_exemption(
    image_name: str,
    exemption: object,
    *,
    source_texts: Mapping[str, str] | None = None,
) -> None:
    """Bind one exemption to its exact build sources and exact sudoers grants."""

    assert isinstance(exemption, Mapping), image_name
    rationale = str(exemption.get("rationale") or "").strip()
    assert len(rationale) >= 80, f"{image_name}: passwordless-root rationale is missing"
    sources = [str(value) for value in exemption.get("sources") or []]
    assert sources and len(sources) == len(set(sources)), (
        f"{image_name}: passwordless-root sources must be non-empty and unique"
    )
    declared = {
        (str(item.get("source") or ""), str(item.get("sudoers_entry") or ""))
        for item in exemption.get("grants") or []
        if isinstance(item, Mapping)
    }
    assert declared and all(source and grant for source, grant in declared), (
        f"{image_name}: passwordless-root grants must name source and sudoers_entry"
    )
    assert {source for source, _grant in declared} <= set(sources), (
        f"{image_name}: declared grants must belong to the reviewed source set"
    )

    actual: set[tuple[str, str]] = set()
    for source in sources:
        path = WORKBENCH_DOCKER / source
        text = (
            source_texts[source]
            if source_texts is not None and source in source_texts
            else path.read_text(encoding="utf-8")
        )
        assert path.is_file() or source_texts is not None, (
            f"{image_name}: passwordless-root source not found: {source}"
        )
        actual.update((source, grant) for grant in _passwordless_root_grants(text))
    assert actual == declared, (
        f"{image_name}: passwordless-root grants differ from the reviewed contract; "
        f"added={sorted(actual - declared)}, removed={sorted(declared - actual)}"
    )


def _validate_derived_bootstrap_source(
    image_name: str,
    entry: Mapping[str, object],
    *,
    source_text: str | None = None,
) -> None:
    """Verify the exact derived Dockerfile that declares the bootstrap label."""

    derived = entry.get("derived_skypilot_bootstrap_contract")
    assert isinstance(derived, Mapping), image_name
    assert derived.get("version") == "skypilot-0.12.2-v1", image_name
    assert derived.get("verification") == "runtime_probe_required", image_name
    dockerfile = WORKBENCH_DOCKER / str(derived.get("dockerfile") or "")
    assert dockerfile != WORKBENCH_DOCKER / str(entry.get("dockerfile") or ""), (
        f"{image_name}: derived contract must not attest the canonical Dockerfile"
    )
    text = (
        source_text
        if source_text is not None
        else dockerfile.read_text(encoding="utf-8")
    )
    version = str(derived["version"])
    assert f'org.nebius.npa.skypilot-bootstrap-contract="{version}"' in text
    for token in (
        "openssh-server",
        "rsync",
        "sudo",
        "ubuntu ALL=(ALL) NOPASSWD:ALL",
        "ssh-keygen -A",
        "rm -f /etc/ssh/ssh_host_*",
        "PasswordAuthentication no",
        "PermitRootLogin no",
        r"exec \"$@\"",
    ):
        assert token in text, f"{image_name}: derived source missing {token!r}"
    assert _final_user(text) == "ubuntu", image_name
    assert text.index("ssh-keygen -A") < text.index("rm -f /etc/ssh/ssh_host_*"), (
        f"{image_name}: runtime host-key generation must precede build-key removal"
    )


def _bake_matches(dockerfile_text: str, patterns: list[dict]) -> list[str]:
    """Return the ``kind``s of every bake pattern this Dockerfile trips.

    This is the heart of the redesigned guard. The old version matched substrings
    (``isaacsim``, ``OMNI_KIT_ACCEPT_EULA``, ...) against the instructions, which was
    right while the images baked Isaac Sim. Now that they fetch it at run time, their
    Dockerfiles still legitimately *mention* those names - in bootstrap plumbing, in
    version pins, and in comments explaining what is deliberately absent - so a
    substring match would fire on all four. Weakening it to make the build pass would
    have thrown the guard away; instead it now distinguishes **baked at build time**
    from **referenced for run time**.
    """

    instructions = _normalize_dockerfile(dockerfile_text)
    return [
        entry["kind"] for entry in patterns if re.search(entry["pattern"], instructions)
    ]


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
            number = token.split("/", 1)[0]
            if number.isdigit():
                ports.append(int(number))
    return ports


def test_packaging_contract_file_exists() -> None:
    assert CONTRACT_PATH.is_file()
    contract = _load_contract()
    assert contract["version"] == 1
    assert "service" in contract["tiers"]
    assert "job" in contract["tiers"]
    assert "interactive" in contract["tiers"]
    assert contract["images"]


def test_images_that_install_npa_copy_forced_workflow_package_data() -> None:
    """Hatch metadata generation must see every force-included workflow YAML.

    ``pyproject.toml`` force-includes files below ``workflows/``. A Dockerfile that
    copies the project metadata and installs ``/opt/npa`` therefore cannot copy only
    ``src/npa``: pip fails before it can build editable or regular package metadata.
    """

    missing: list[str] = []
    for dockerfile in sorted(WORKBENCH_DOCKER.rglob("Dockerfile*")):
        if not dockerfile.is_file():
            continue
        instructions = _normalize_dockerfile(dockerfile.read_text(encoding="utf-8"))
        installs_npa = any(
            re.search(pattern, instructions)
            for pattern in (
                r"\b(?:uv\s+)?pip\s+install\b[^\n]*/opt/npa",
                r"\bpython\s+-m\s+pip\s+install\b[^\n]*/opt/npa",
            )
        )
        if not installs_npa or "/opt/npa/pyproject.toml" not in instructions:
            continue
        if not re.search(
            r"\bCOPY\b[^\n]*\b(?:npa/)?workflows\s+/opt/npa/workflows\b",
            instructions,
        ):
            missing.append(str(dockerfile.relative_to(ROOT)))
    assert not missing, "missing workflow package data: " + ", ".join(missing)


def test_declared_skypilot_images_enforce_the_versioned_build_contract() -> None:
    contract = _load_contract()
    for name, item in contract["images"].items():
        version = item.get("skypilot_bootstrap_contract")
        if not version:
            continue
        dockerfile = WORKBENCH_DOCKER / item["dockerfile"]
        dockerfile_text = dockerfile.read_text(encoding="utf-8")
        text = _build_contract_text(dockerfile)
        assert version == "skypilot-0.12.2-v1", name
        assert (
            f'org.nebius.npa.skypilot-bootstrap-contract="{version}"'
            in dockerfile_text
        ), name
        for package in ("openssh-server", "rsync", "sudo"):
            assert package in text, f"{name}: missing {package}"
        assert "NOPASSWD" in text or _final_user(text) in {None, "root", "0"}, name
        entrypoints = _entrypoints(dockerfile_text)
        assert entrypoints, f"{name}: contract images need a forwarding entrypoint"
        script_matches = re.findall(r'ENTRYPOINT\s+\["([^"]+)"\]', dockerfile_text)
        assert script_matches, name
        entrypoint_path = script_matches[-1]
        script = dockerfile.parent / Path(entrypoint_path).name
        if not script.is_file():
            copy_match = re.search(
                rf"(?im)^COPY\s+(?:--\S+\s+)*(?P<src>\S+)\s+"
                rf"{re.escape(entrypoint_path)}\s*$",
                _normalize_dockerfile(dockerfile_text),
            )
            if copy_match:
                script = ROOT / "npa" / copy_match.group("src")
        assert script.is_file(), f"{name}: entrypoint source not found: {script}"
        entrypoint_text = script.read_text(encoding="utf-8")
        assert (
            'exec "$@"' in entrypoint_text or 'exec "$MODE" "$@"' in entrypoint_text
        ), name


def test_packaged_skypilot_attestation_inventory_matches_contract() -> None:
    contract = _load_contract()
    declared = {
        name
        for name, item in contract["images"].items()
        if item.get("skypilot_bootstrap_contract")
    }
    assert SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS == declared


def test_runtime_probed_bootstrap_inventory_matches_exact_derived_sources() -> None:
    contract = _load_contract()
    declared = {
        name
        for name, item in contract["images"].items()
        if (
            isinstance(item.get("derived_skypilot_bootstrap_contract"), Mapping)
            and item["derived_skypilot_bootstrap_contract"].get("verification")
            == "runtime_probe_required"
        )
    }
    assert SKYPILOT_BOOTSTRAP_RUNTIME_PROBED_TOOLS == declared
    for name in sorted(declared):
        _validate_derived_bootstrap_source(name, contract["images"][name])


@pytest.mark.parametrize(
    "removed_token",
    [
        "openssh-server",
        "ssh-keygen -A",
        "rm -f /etc/ssh/ssh_host_*",
        r"exec \"$@\"",
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"',
    ],
)
def test_derived_bootstrap_source_guard_rejects_contract_mutations(
    removed_token: str,
) -> None:
    entry = _load_contract()["images"]["groot"]
    dockerfile = (
        WORKBENCH_DOCKER / entry["derived_skypilot_bootstrap_contract"]["dockerfile"]
    )
    text = dockerfile.read_text(encoding="utf-8")
    assert removed_token in text

    with pytest.raises(AssertionError):
        _validate_derived_bootstrap_source(
            "groot", entry, source_text=text.replace(removed_token, "removed")
        )


def test_derived_bootstrap_contract_cannot_be_redirected_to_canonical_source() -> None:
    entry = deepcopy(_load_contract()["images"]["groot"])
    entry["derived_skypilot_bootstrap_contract"]["dockerfile"] = "groot/Dockerfile"

    with pytest.raises(AssertionError, match="must not attest the canonical"):
        _validate_derived_bootstrap_source("groot", entry)


def test_fiftyone_image_has_skypilot_kubernetes_prerequisites() -> None:
    """The workflow image must survive SkyPilot's non-root pod bootstrap."""
    text = (WORKBENCH_DOCKER / "fiftyone" / "Dockerfile").read_text(encoding="utf-8")
    for package in ("netcat-openbsd", "openssh-server", "patch", "rsync", "sudo"):
        assert re.search(rf"(?m)^\s+{re.escape(package)}\s*\\\\?$", text), package
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text
    assert "PasswordAuthentication no" in text
    assert "PermitRootLogin no" in text


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_public_images_explain_passwordless_root(image_name: str) -> None:
    """A public image cannot silently acquire an unrestricted sudo grant."""
    contract = _load_contract()
    entry = contract["images"][image_name]
    if entry.get("redistribution") != "public":
        return
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    grants_passwordless_root = bool(
        re.search(r"(?im)^\s*[^#\n]+\bNOPASSWD\s*:\s*(?:ALL|/)", text)
    )
    exemption = entry.get("passwordless_root_exemption")
    if isinstance(exemption, Mapping):
        _validate_structured_passwordless_root_exemption(image_name, exemption)
    elif grants_passwordless_root:
        rationale = str(exemption or "").strip()
        assert len(rationale) >= 80, (
            f"{image_name}: public image grants passwordless root without a narrow "
            "passwordless_root_exemption in packaging-contract.yaml"
        )


def test_groot_passwordless_root_contract_is_mutation_sensitive() -> None:
    exemption = _load_contract()["images"]["groot"]["passwordless_root_exemption"]
    assert exemption["sources"] == [
        "groot/Dockerfile",
        "common/install_isaac_runtime_base.sh",
        "groot/Dockerfile.k8s-prereqs",
    ]
    source_texts = {
        source: (WORKBENCH_DOCKER / source).read_text(encoding="utf-8")
        for source in exemption["sources"]
    }
    _validate_structured_passwordless_root_exemption(
        "groot", exemption, source_texts=source_texts
    )

    added_grant = dict(source_texts)
    added_grant["groot/Dockerfile"] += (
        "\nRUN printf 'root ALL=(ALL) NOPASSWD:ALL\\n' > /etc/sudoers.d/root\n"
    )
    with pytest.raises(AssertionError, match="grants differ"):
        _validate_structured_passwordless_root_exemption(
            "groot", exemption, source_texts=added_grant
        )

    for source in (
        "common/install_isaac_runtime_base.sh",
        "groot/Dockerfile.k8s-prereqs",
    ):
        removed_grant = dict(source_texts)
        removed_grant[source] = removed_grant[source].replace(
            "ubuntu ALL=(ALL) NOPASSWD:ALL", "ubuntu ALL=(ALL) PASSWD:ALL", 1
        )
        with pytest.raises(AssertionError, match="grants differ"):
            _validate_structured_passwordless_root_exemption(
                "groot", exemption, source_texts=removed_grant
            )

    missing_rationale = deepcopy(exemption)
    missing_rationale["rationale"] = ""
    with pytest.raises(AssertionError, match="rationale is missing"):
        _validate_structured_passwordless_root_exemption(
            "groot", missing_rationale, source_texts=source_texts
        )


def test_groot_uses_a_fixed_consistent_linux_headers_snapshot() -> None:
    """GR00T must fix inherited headers without leaving dpkg inconsistent."""
    text = (WORKBENCH_DOCKER / "groot" / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG GROOT_UBUNTU_SNAPSHOT=20260827T000000Z" in text
    assert "ARG GROOT_LINUX_LIBC_DEV_VERSION=5.15.0-190.200" in text
    assert "NPA_UBUNTU_SNAPSHOT=${GROOT_UBUNTU_SNAPSHOT}" in text
    assert (
        "NPA_LINUX_LIBC_DEV_VERSION=${GROOT_LINUX_LIBC_DEV_VERSION}" in text
    )
    assert '"linux-libc-dev=${GROOT_LINUX_LIBC_DEV_VERSION}"' in text
    assert "dpkg --purge --force-depends linux-libc-dev" not in text


def test_sim2real_control_root_exception_is_finite_and_task_scoped() -> None:
    entry = _load_contract()["images"]["sim2real-control"]
    exception = entry["privilege_exception"]
    assert exception == {
        "id": "sim2real-control-skypilot-0.12.2-root-bootstrap",
        "expires_with": "skypilot-rootless-kubernetes-bootstrap",
        "scope": "ephemeral-standard-workflow-task-pod",
        "prohibited_uses": ["service", "public-ingress", "baked-credentials"],
    }
    rationale = entry["passwordless_root_exemption"]
    for boundary in ("CPU-only", "uid 1000", "not started", "trust boundary"):
        assert boundary in rationale


def test_first_class_pinned_dockerfiles_are_covered_by_contract() -> None:
    """A pin plus an in-tree same-name Dockerfile may not bypass legal review.

    Derived aliases such as ``envgen`` intentionally map to a differently named
    Dockerfile and are covered through their parent contract entry. A first-class
    ``workbench/<tool>/Dockerfile`` has no such excuse: adding its pin must add a
    packaging classification in the same change.
    """

    contract_images = set(_load_contract()["images"])
    first_class = {
        tool
        for tool in CONTAINER_IMAGE_NAMES
        if (WORKBENCH_DOCKER / tool / "Dockerfile").is_file()
    }
    assert first_class <= contract_images, (
        "pinned first-class Dockerfiles missing from packaging-contract.yaml: "
        f"{sorted(first_class - contract_images)}"
    )


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
            assert final_user == expected_user, (
                f"{image_name}: expected USER {expected_user}, got {final_user}"
            )
        else:
            allowed = set(contract["security"]["allowed_final_users"])
            assert final_user is not None, f"{image_name}: missing final USER"
            # Allow USER $NPA_RUNTIME_USER style only when default is documented non-root.
            if final_user.startswith("$"):
                assert "ubuntu" in text or "NPA_RUNTIME_USER" in text
            else:
                assert final_user in allowed, (
                    f"{image_name}: final USER {final_user!r} not in {allowed}"
                )

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
        assert re.search(pattern, text) is None, (
            f"{image_name}: Dockerfile matches secret pattern {pattern}"
        )


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
def test_dockerfiles_that_bake_omniverse_are_restricted(image_name: str) -> None:
    """An image that BAKES NVIDIA Omniverse Kit (Isaac Sim) is not freely
    redistributable and must be ``restricted``, so it is never published publicly.

    Baking is what matters, not mentioning. The Isaac images fetch Isaac Sim / Isaac
    Lab at first run under the operator's own EULA acceptance, so their Dockerfiles
    reference isaacsim, isaaclab and the EULA variable names without shipping a byte
    of any of them.
    """
    contract = _load_contract()
    entry = contract["images"][image_name]
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    patterns = contract["redistribution"]["omniverse_bake_patterns"]
    matched = _bake_matches(text, patterns)

    if matched:
        assert entry.get("redistribution") == "restricted", (
            f"{image_name}: Dockerfile bakes Omniverse Kit at build time "
            f"({', '.join(matched)}); it must be redistribution: restricted "
            f"(NVIDIA proprietary - public redistribution needs an NVIDIA AI "
            f"Enterprise license). If the intent was to fetch Isaac at RUN time, use "
            f"npa/docker/workbench/common/isaac_bootstrap.sh instead."
        )


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_images_derived_from_restricted_are_restricted(image_name: str) -> None:
    """Restriction is inherited, and a child shows no marker of its own.

    This is not redundant with the bake check: sonic-mujoco builds FROM npa-sonic and
    would carry its parent's payload while its own Dockerfile looked clean.
    """
    contract = _load_contract()
    entry = contract["images"][image_name]
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    restricted_parents = sorted(
        parent
        for parent in _workbench_parents(text, contract["images"])
        if contract["images"][parent].get("redistribution") == "restricted"
    )
    assert not restricted_parents or entry.get("redistribution") == "restricted", (
        f"{image_name}: builds FROM restricted workbench image(s) {restricted_parents} "
        f"and inherits whatever they bake; it must be redistribution: restricted"
    )


# --------------------------------------------------------------------------------------
# Mutation tests for the bake detector, in BOTH directions.
#
# The detector was redesigned from "mentions Isaac" to "bakes Isaac" precisely so the
# runtime-fetch images could stop being restricted. That redesign is only trustworthy if
# it still catches every way an image can bake Isaac, so both directions are pinned:
# MUST_DETECT covers every historical baking form plus the two the runtime-fetch design
# itself introduces, and MUST_NOT_DETECT covers every legitimate reference that now
# appears in the shipped Dockerfiles.
# --------------------------------------------------------------------------------------

MUST_DETECT = {
    "nvcr isaac-lab base": "FROM nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc80\n",
    "nvcr isaac-sim base": "FROM nvcr.io/nvidia/isaac-sim:4.5.0\n",
    "nvcr base via ARG": (
        "ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2\n"
        "FROM --platform=linux/amd64 ${BASE_IMAGE}\n"
    ),
    "pip isaacsim": 'RUN pip install --no-cache-dir "isaacsim==5.1.0.0"\n',
    "pip isaaclab": 'RUN pip install --no-deps "isaaclab==2.3.2.post1"\n',
    "pip isaaclab with isaacsim extra": 'RUN pip install "isaaclab[isaacsim,all]==2.3.2.post1"\n',
    # The real sonic install spanned 27 backslash-continued lines; a per-line matcher
    # missed it entirely while still tripping on the comment above it.
    "pip isaacsim across line continuations": (
        "RUN python -m pip install --no-cache-dir --no-deps \\\n"
        '      "isaacsim-kernel==5.1.0.0" \\\n'
        '      "isaacsim-extscache-kit==5.1.0.0" \\\n'
        "      --extra-index-url https://pypi.nvidia.com\n"
    ),
    "baked ENV OMNI_KIT_ACCEPT_EULA": "ENV OMNI_KIT_ACCEPT_EULA=YES\n",
    "baked ENV ISAACSIM_ACCEPT_EULA": "ENV ISAACSIM_ACCEPT_EULA=YES\n",
    "baked ENV ACCEPT_EULA": "ENV ACCEPT_EULA=Y\n",
    "baked ENV PRIVACY_CONSENT": "ENV PRIVACY_CONSENT=Y\n",
    "baked ARG EULA acceptance": "ARG OMNI_KIT_ACCEPT_EULA=YES\n",
    "baked EULA in a continued ENV block": (
        "ENV ACCEPT_EULA=Y \\\n    OMNI_KIT_ACCEPT_EULA=YES \\\n    PYTHONUNBUFFERED=1\n"
    ),
    "COPY of a Kit tree": "COPY --from=vendor /isaac-sim/kit/ /opt/kit/\n",
    # The two below are introduced by the runtime-fetch design itself: either would
    # materialise the whole install into a layer, and both are easy to add by accident.
    "bootstrap run at build time": "RUN /opt/npa/bin/isaac-bootstrap ensure\n",
    "bootstrap warm at build time": "RUN isaac_bootstrap.sh warm\n",
    "isaac shim invoked at build time": 'RUN /isaac-sim/python.sh -c "import isaaclab"\n',
    "isaac-python invoked at build time": "RUN isaac-python -m pip install foo\n",
    "hash-locked OVRTX wheel install": (
        "RUN uv pip install --python /opt/ovrtx/bin/python "
        "-r pylock.ovrtx-runtime.toml --require-hashes --no-deps\n"
    ),
    "OVRTX provision-only bootstrap at build time": (
        "RUN python -m world_understanding.functions.graphics.render_ovrtx "
        "--provision-only\n"
    ),
}

MUST_NOT_DETECT = {
    "prose about what is not baked": (
        "# This image is deliberately NOT built FROM nvcr.io/nvidia/isaac-lab, and no\n"
        "# isaacsim or isaaclab wheel is installed. ISAACSIM_ACCEPT_EULA and\n"
        "# OMNI_KIT_ACCEPT_EULA are left unset on purpose: acceptance is the operator's.\n"
        "FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04\n"
    ),
    "copying the bootstrap in": "COPY docker/workbench/common /opt/npa/docker/workbench/common\n",
    "pointing ISAAC_LAB_PYTHON at the shim": "ENV ISAAC_LAB_PYTHON=/isaac-sim/python.sh\n",
    "declaring the cache dir": "ENV NPA_ISAAC_CACHE_DIR=/opt/isaac-cache\n",
    "pinning the versions to fetch": (
        "ENV ISAAC_SIM_VERSION=5.1.0.0 \\\n    ISAAC_LAB_VERSION=2.3.2.post1\n"
    ),
    "naming the wheel manifest": (
        "ENV NPA_ISAAC_WHEELS_FILE=/opt/npa/docker/workbench/common/isaac-nvidia-wheels.txt\n"
    ),
    "running the base installer": (
        "RUN /opt/npa/docker/workbench/common/install_isaac_runtime_base.sh\n"
    ),
    "asking the bootstrap for status": "RUN /opt/npa/bin/isaac-bootstrap status\n",
    "installing OSS deps": "RUN pip install -r isaac-oss-deps.txt\n",
    "copying an isaac-lab smoke script": (
        "COPY docker/workbench/isaac-lab/smoke_functional.py /opt/npa/smoke_functional.py\n"
    ),
    "pip install via the image python arg": (
        'RUN "${NPA_IMAGE_PYTHON}" -m pip install --no-cache-dir "mujoco>=3.3,<3.4"\n'
    ),
    "pinning the Isaac Lab source commit": "ENV ISAAC_LAB_SRC_COMMIT=37ddf626871758333d6e\n",
}


@pytest.mark.parametrize("name", sorted(MUST_DETECT))
def test_bake_detector_catches_every_way_of_baking_isaac(name: str) -> None:
    patterns = _load_contract()["redistribution"]["omniverse_bake_patterns"]
    snippet = "# an explanatory header comment\n" + MUST_DETECT[name]
    matched = _bake_matches(snippet, patterns)
    assert matched, f"bake form not detected: {name}\n{snippet}"


@pytest.mark.parametrize("name", sorted(MUST_NOT_DETECT))
def test_bake_detector_allows_runtime_fetch_references(name: str) -> None:
    patterns = _load_contract()["redistribution"]["omniverse_bake_patterns"]
    matched = _bake_matches(MUST_NOT_DETECT[name], patterns)
    assert not matched, (
        f"legitimate runtime-fetch reference wrongly flagged as baking: {name} "
        f"({', '.join(matched)})\n{MUST_NOT_DETECT[name]}"
    )


@pytest.mark.parametrize("image_name", ["groot", "isaac-lab", "sonic", "sonic-mujoco"])
def test_reintroducing_a_baked_install_fails_the_guard(image_name: str) -> None:
    """Mutation test against the REAL Dockerfiles, not just synthetic snippets.

    Re-inserting a baked install into each shipped Dockerfile must trip the detector.
    Without this the detector could pass simply because the real Dockerfiles happen not
    to resemble the synthetic cases.
    """
    contract = _load_contract()
    patterns = contract["redistribution"]["omniverse_bake_patterns"]
    text = (WORKBENCH_DOCKER / contract["images"][image_name]["dockerfile"]).read_text(
        encoding="utf-8"
    )
    assert not _bake_matches(text, patterns), f"{image_name} unexpectedly bakes Isaac"

    for mutation in (
        'RUN pip install "isaacsim==5.1.0.0"\n',
        "ENV OMNI_KIT_ACCEPT_EULA=YES\n",
        "FROM nvcr.io/nvidia/isaac-lab:2.3.2\n",
        "RUN /opt/npa/bin/isaac-bootstrap ensure\n",
    ):
        assert _bake_matches(text + mutation, patterns), (
            f"{image_name}: re-adding {mutation.strip()!r} must trip the bake guard"
        )


def test_content_agents_runtime_fetch_guard_is_mutation_tested() -> None:
    contract = _load_contract()
    entry = contract["images"]["content-agents"]
    assert entry["ovrtx_runtime_fetch"] is True
    patterns = contract["redistribution"]["omniverse_bake_patterns"]
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    assert not _bake_matches(text, patterns)

    mutations = (
        "RUN uv pip install -r pylock.ovrtx-runtime.toml --require-hashes\n",
        "RUN python -m world_understanding.functions.graphics.render_ovrtx "
        "--provision-only\n",
    )
    for mutation in mutations:
        assert _bake_matches(text + mutation, patterns), mutation


@pytest.mark.parametrize(
    "image_name",
    sorted(
        name
        for name, entry in _load_contract()["images"].items()
        if entry.get("isaac_runtime_fetch")
    ),
)
def test_isaac_runtime_fetch_images_wire_the_bootstrap(image_name: str) -> None:
    """Declaring ``isaac_runtime_fetch`` must mean the bootstrap is actually reachable.

    Otherwise an image could claim the architecture, bake nothing, and simply fail at
    run time - or worse, quietly resolve Isaac some other way.
    """
    contract = _load_contract()
    entry = contract["images"][image_name]
    text = (WORKBENCH_DOCKER / entry["dockerfile"]).read_text(encoding="utf-8")
    instructions = _normalize_dockerfile(text)

    parents = _workbench_parents(text, contract["images"])
    if any(contract["images"][parent].get("isaac_runtime_fetch") for parent in parents):
        # A derived image inherits the wiring; requiring it to re-COPY the bootstrap
        # would be wrong (sonic-mujoco intentionally adds no Isaac logic of its own).
        return

    assert "docker/workbench/common" in instructions, (
        f"{image_name}: declares isaac_runtime_fetch but never COPYs "
        f"docker/workbench/common (the bootstrap and its pinned wheel manifest)"
    )
    assert "install_isaac_runtime_base.sh" in instructions, (
        f"{image_name}: declares isaac_runtime_fetch but never runs "
        f"install_isaac_runtime_base.sh, which installs the shim"
    )


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_no_image_bakes_eula_acceptance(image_name: str) -> None:
    """The refusal to run without operator-supplied acceptance is the legal mechanism.

    Baking any acceptance variable removes the operator from the licensing decision, so
    this is checked for EVERY image, not only the Isaac ones.
    """
    contract = _load_contract()
    text = (WORKBENCH_DOCKER / contract["images"][image_name]["dockerfile"]).read_text(
        encoding="utf-8"
    )
    instructions = _normalize_dockerfile(text)
    offenders = re.findall(
        r"(?im)^\s*(?:ENV|ARG)\s+[^\n]*\b((?:OMNI_KIT_|ISAACSIM_)?ACCEPT_EULA"
        r"|PRIVACY_CONSENT)\s*=\s*\S+",
        instructions,
    )
    assert not offenders, (
        f"{image_name}: bakes EULA acceptance ({sorted(set(offenders))}). Acceptance "
        f"must be supplied by the operator at run time; see "
        f"npa/docker/workbench/common/isaac_bootstrap.sh."
    )


@pytest.mark.parametrize("image_name", sorted(_load_contract()["images"]))
def test_no_image_builds_from_an_nvcr_base(image_name: str) -> None:
    """No workbench image may pull from NVIDIA's credentialed registry.

    An nvcr.io base both bakes proprietary content and makes the build depend on an NGC
    login, so build-your-own stops working for anyone without NGC credentials.
    """
    contract = _load_contract()
    text = (WORKBENCH_DOCKER / contract["images"][image_name]["dockerfile"]).read_text(
        encoding="utf-8"
    )
    for base in _base_image_refs(_normalize_dockerfile(text)):
        assert "nvcr.io" not in base, f"{image_name}: builds FROM {base}"


def test_packaging_doc_exists() -> None:
    doc = ROOT / "docs" / "workbench" / "container-packaging.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "Packaging tiers" in text
    assert "Security baseline" in text
    assert "packaging-contract.yaml" in text
