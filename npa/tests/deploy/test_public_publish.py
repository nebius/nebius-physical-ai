"""License-guarded public-development to supported-release GHCR publishing.

These tests lock the license
boundary: whatever is classified non-redistributable must never be selected for a public
registry, and the selector must stay in sync with the packaging contract's
``redistribution:`` fields.

The Isaac images are no longer restricted — they were
re-architected to fetch Isaac Sim / Isaac Lab at first run under the operator's own EULA
acceptance instead of baking it. Cosmos3 serving is now a zero-payload runtime
bootstrap and SONIC MuJoCo is independently rebuilt on a public base. That makes
the boundary tests the delicate ones: asserting "nothing is restricted" would pass just
as well against a guard that had been deleted. So the tests that exercise the refusal
monkeypatch a synthetic restricted catalog tool in, proving the mechanism still bites.
"""

from __future__ import annotations

import json
import copy
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.deploy import images
from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    DEFAULT_PUBLIC_CONTAINER_REGISTRY,
    RESTRICTED_DERIVED_IMAGES,
    RESTRICTED_PUBLICATION_TOOLS,
    UNVALIDATED_PUBLICATION_TOOLS,
    container_image_for_tool,
    is_public_registry,
    is_publicly_redistributable,
    public_container_registry,
    publicly_publishable_tools,
    restricted_image_names,
)
from npa.deploy.publish_public import (
    PublishItem,
    _pin_publication_sources as REAL_PUBLICATION_SOURCE_PIN,
    build_publish_plan,
    verify_bootstrap_publication_source as REAL_BOOTSTRAP_PUBLICATION_GATE,
    verify_gpu_accepted_publication_source as REAL_GPU_ACCEPTANCE_GATE,
    verify_ltx_publication_source as REAL_LTX_PUBLICATION_GATE,
    verify_wan_publication_source as REAL_WAN_PUBLICATION_GATE,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


@pytest.fixture(autouse=True)
def _avoid_registry_attestation_reads_in_unrelated_publish_tests(monkeypatch) -> None:
    """Keep generic publish tests focused while the dedicated Wan tests exercise the gate."""
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public,
        "verify_wan_publication_source",
        lambda item: (True, "test fixture: Wan gate verified"),
    )
    monkeypatch.setattr(
        publish_public,
        "verify_ltx_publication_source",
        lambda item: (True, "test fixture: LTX gate verified"),
    )
    monkeypatch.setattr(
        publish_public,
        "verify_content_agents_publication_source",
        lambda item: (True, "test fixture: Content Agents gate verified"),
    )
    monkeypatch.setattr(
        publish_public,
        "_pin_publication_sources",
        lambda plan: (list(plan), []),
    )
    monkeypatch.setattr(
        publish_public,
        "verify_bootstrap_publication_source",
        lambda item: (True, "test fixture: bootstrap gate verified"),
    )
    monkeypatch.setattr(
        publish_public,
        "verify_gpu_accepted_publication_source",
        lambda item: (True, "test fixture: GPU acceptance gate verified"),
    )


def test_gpu_accepted_publication_gate_binds_exact_digest(monkeypatch) -> None:
    from npa.deploy import publish_public

    item = PublishItem(
        tool="cosmos3-serving",
        source_ref="ghcr.io/example/npa-cosmos3-serving:dev-source",
        target_ref="ghcr.io/example/npa-cosmos3-serving:release",
    )
    accepted = images.GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS[item.tool]
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda _ref, **_: (True, accepted)
    )
    assert REAL_GPU_ACCEPTANCE_GATE(item) == (
        True,
        f"exact GPU-accepted digest {accepted}",
    )

    mutation = "sha256:" + "f" * 64
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda _ref, **_: (True, mutation)
    )
    ok, detail = REAL_GPU_ACCEPTANCE_GATE(item)
    assert not ok
    assert mutation in detail
    assert accepted in detail


def test_wan_source_tag_is_frozen_before_preflight_and_copy(monkeypatch) -> None:
    """A later tag retarget cannot change the bytes selected for publication."""

    from npa.deploy import publish_public

    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, digest)
    )
    plan, failures = REAL_PUBLICATION_SOURCE_PIN(
        [
            PublishItem(
                tool="wan2-2",
                source_ref="source.example/npa-wan2-2:accepted",
                target_ref="target.example/npa-wan2-2:accepted",
            )
        ]
    )

    assert failures == []
    assert plan[0].source_ref == f"source.example/npa-wan2-2@{digest}"


def test_every_publication_source_is_frozen_before_any_gate(monkeypatch) -> None:
    from npa.deploy import publish_public

    digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, digest)
    )
    plan, failures = REAL_PUBLICATION_SOURCE_PIN(
        [
            PublishItem(
                tool="cosmos2-transfer",
                source_ref="source.example/npa-cosmos2-transfer:release",
                target_ref="target.example/npa-cosmos2-transfer:release",
            )
        ]
    )
    assert failures == []
    assert plan[0].source_ref.endswith(f"@{digest}")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ({}, "missing bootstrap-contract attestation"),
        (
            {"org.nebius.npa.skypilot-bootstrap-contract": "stale-contract"},
            "attestation version mismatch",
        ),
    ],
)
def test_publication_refuses_missing_or_stale_bootstrap_attestation(
    monkeypatch, labels, expected
) -> None:
    from npa.deploy import publish_public

    digest = "sha256:" + "c" * 64
    monkeypatch.setattr(
        publish_public,
        "_crane_json",
        lambda args: {"config": {"Labels": labels}},
    )
    ok, detail = REAL_BOOTSTRAP_PUBLICATION_GATE(
        PublishItem(
            tool="cosmos2-transfer",
            source_ref=f"source.example/npa-cosmos2-transfer@{digest}",
            target_ref="target.example/npa-cosmos2-transfer:release",
        )
    )
    assert not ok
    assert expected in detail


def test_publication_accepts_exact_digest_bootstrap_attestation(monkeypatch) -> None:
    from npa.deploy import publish_public
    from npa.orchestration.skypilot.image_bootstrap_contract import (
        ATTESTATION_LABEL,
        CONTRACT_VERSION,
    )

    digest = "sha256:" + "d" * 64
    monkeypatch.setattr(
        publish_public,
        "_crane_json",
        lambda args: {"config": {"Labels": {ATTESTATION_LABEL: CONTRACT_VERSION}}},
    )
    ok, detail = REAL_BOOTSTRAP_PUBLICATION_GATE(
        PublishItem(
            tool="cosmos2-transfer",
            source_ref=f"source.example/npa-cosmos2-transfer@{digest}",
            target_ref="target.example/npa-cosmos2-transfer:release",
        )
    )
    assert ok
    assert digest in detail


@pytest.mark.parametrize("tool", ["cosmos", "groot"])
def test_preflight_skips_bootstrap_gate_for_uncontracted_image(
    monkeypatch, tool: str
) -> None:
    from npa.deploy import publish_public

    item = PublishItem(
        tool=tool,
        source_ref=f"source.example/npa-{tool}:release",
        target_ref=f"target.example/npa-{tool}:release",
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )

    def unexpected_gate(_item):  # pragma: no cover - must not run
        raise AssertionError("uncontracted image reached the bootstrap gate")

    monkeypatch.setattr(
        publish_public, "verify_bootstrap_publication_source", unexpected_gate
    )

    assert publish_public.preflight_sources([item]) == []


def test_preflight_runs_bootstrap_gate_for_contracted_image(monkeypatch) -> None:
    from npa.deploy import publish_public

    item = PublishItem(
        tool="fiftyone",
        source_ref="source.example/npa-fiftyone@sha256:" + "a" * 64,
        target_ref="target.example/npa-fiftyone:release",
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )
    checked: list[PublishItem] = []

    def gate(candidate: PublishItem) -> tuple[bool, str]:
        checked.append(candidate)
        return True, "attested"

    monkeypatch.setattr(publish_public, "verify_bootstrap_publication_source", gate)

    assert publish_public.preflight_sources([item]) == []
    assert checked == [item]


def test_isaac_images_are_no_longer_restricted() -> None:
    """Removing baked Omniverse Kit made the Isaac images publishable.

    They now fetch Isaac Sim / Isaac Lab at first run from pypi.nvidia.com under the
    operator's own EULA acceptance and ship no NVIDIA Isaac bytes, verified against the
    built image by npa/scripts/scan_image_omniverse_payload.py (isaac-lab: 83,043 entries
    scanned; sonic: 125,655 entries; both VERDICT clean).
    """
    for tool in ("isaac-lab", "sonic", "groot"):
        assert is_publicly_redistributable(tool), tool


def test_rebuilt_cosmos3_serving_and_sonic_mujoco_are_gpu_accepted() -> None:
    """Clean bytes and exact GPU evidence earn release eligibility.

    Omniverse Kit was only the first: sonic also baked gated model weights (git-LFS
    smudging) and NVIDIA Omniverse 3D assets (the RoboCasa asset library under
    decoupled_wbc/dexmg). Both were found by scanning the built image, and neither was
    visible in the Dockerfile. The scan that clears it:
    npa-sonic:0.1.2-rtfetch-rc5, 125,655 entries, 16 allowlisted paths, VERDICT clean.
    """
    assert RESTRICTED_PUBLICATION_TOOLS == frozenset()
    assert RESTRICTED_DERIVED_IMAGES == frozenset()
    for tool in ("isaac-lab", "sonic", "groot", "cosmos3-serving", "sonic-mujoco"):
        assert is_publicly_redistributable(tool), tool
    assert UNVALIDATED_PUBLICATION_TOOLS == frozenset()
    assert set(images.GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS) == {
        "cosmos3-ray-serve",
        "cosmos3-serving",
        "sonic-mujoco",
    }


def test_public_set_excludes_every_restricted_tool(monkeypatch) -> None:
    """The exclusion remains effective for canonical tools too."""
    monkeypatch.setattr(
        images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis", "cosmos"})
    )
    public = set(publicly_publishable_tools())
    assert public.isdisjoint({"genesis", "cosmos"})
    for tool in ("genesis", "cosmos"):
        assert not is_publicly_redistributable(tool)
    assert "lerobot" in public, "unrelated tools must stay publishable"


def test_public_set_includes_the_oss_tools() -> None:
    public = set(publicly_publishable_tools())
    for tool in (
        "lerobot",
        "genesis",
        "cosmos",
        "fiftyone",
        "lancedb",
        "rerun-viewer",
        "lichtblick",
        # Newly publishable: no baked Omniverse Kit, weights or assets.
        "isaac-lab",
        "sonic",
        "groot",
        "cosmos3-serving",
        "cosmos3-ray-serve",
        "sonic-mujoco",
    ):
        assert tool in public, tool
    assert public == (
        set(CONTAINER_IMAGE_NAMES)
        - RESTRICTED_PUBLICATION_TOOLS
        - images.PUBLICATION_QUARANTINE_TOOLS
    )


def test_publish_plan_now_includes_the_isaac_images() -> None:
    """The point of the re-architecture: these are publishable at last."""
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    names = {item.source_ref.rsplit("/", 1)[-1].split(":", 1)[0] for item in plan}
    for image in (
        "npa-isaac-lab",
        "npa-sonic",
        "npa-groot",
    ):
        assert image in names, image
    assert "npa-sonic-mujoco" in names
    assert "npa-cosmos3-serving" in names
    for item in plan:
        assert item.target_ref.startswith("ghcr.io/example/workbench/")


def test_publish_plan_still_refuses_a_restricted_image(monkeypatch) -> None:
    """The hard refusal inside build_publish_plan is defence in depth around the selector.

    Monkeypatching the set is also what pins that the refusal reads it through the module
    rather than through a from-import: a stale binding made this guard disagree with
    publicly_publishable_tools(), so a tool the selector considered publishable tripped the
    refusal and the whole plan raised. A defence-in-depth check holding a stale copy of the
    thing it defends is worse than no check.
    """
    monkeypatch.setattr(images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis"}))
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    names = {item.source_ref.rsplit("/", 1)[-1].split(":", 1)[0] for item in plan}
    assert "npa-genesis" not in names
    # sonic is publishable under this monkeypatched set, so the plan must contain it -
    # proving the refusal followed the patched set instead of a captured one.
    assert "npa-sonic" in names


def test_publish_plan_requires_a_target() -> None:
    with pytest.raises(ValueError):
        build_publish_plan(target_registry="")


def test_publish_plan_promotes_dev_sha_to_release_tag() -> None:
    sha = "1" * 40
    plan = build_publish_plan(
        target_registry="ghcr.io/example/workbench",
        development_git_sha=sha,
    )
    assert plan
    accepted_shas = {
        tool: images.accepted_publication_development_sha(tool)
        for tool in (
            "ltx2",
            "wan2-2",
            "cosmos3-serving",
            "cosmos3-ray-serve",
            "sonic-mujoco",
        )
    }
    assert len(set(accepted_shas.values())) == 5
    for item in plan:
        source_image = item.source_ref.rsplit("/", 1)[-1]
        target_image = item.target_ref.rsplit("/", 1)[-1]
        assert source_image.split(":", 1)[0] == target_image.split(":", 1)[0], item
        expected_sha = accepted_shas.get(item.tool) or sha
        assert source_image.endswith(f":dev-{expected_sha}"), item
        assert not target_image.endswith(f":dev-{expected_sha}"), item


def test_accepted_images_use_distinct_exact_development_sources_and_digests() -> None:
    plan = build_publish_plan(
        target_registry="ghcr.io/nebius/nebius-physical-ai",
        development_git_sha="1" * 40,
    )
    by_tool = {item.tool: item for item in plan}
    manifest = images.public_release_manifest()["releases"]

    for tool in (
        "ltx2",
        "wan2-2",
        "cosmos3-serving",
        "cosmos3-ray-serve",
        "sonic-mujoco",
    ):
        entry = manifest[tool]
        assert by_tool[tool].source_ref.endswith(
            f":dev-{entry['development_sha']}"
        )
        if tool == "ltx2":
            accepted_digest = images.ltx2_accepted_image_manifest()["oci_digest"]
        elif tool == "wan2-2":
            accepted_digest = images.wan_accepted_image_manifest()["oci_digest"]
        else:
            accepted_digest = images.GPU_ACCEPTED_PUBLIC_IMAGE_DIGESTS[tool]
        assert entry["published_digest"] == accepted_digest


def test_publish_plan_uses_the_public_sonic_pin_not_the_default_variant() -> None:
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    sonic = next(item for item in plan if item.tool == "sonic")
    expected = (
        "npa-sonic:cuda13-b300-0.1.2-k8s-runtime-"
        "sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )
    assert sonic.source_ref.endswith(f"npa-sonic:dev-{'0' * 40}") is False
    assert sonic.target_ref.endswith(expected)


def test_public_registry_defaults_to_ghcr(monkeypatch) -> None:
    monkeypatch.delenv("NPA_PUBLIC_REGISTRY", raising=False)
    assert public_container_registry() == DEFAULT_PUBLIC_CONTAINER_REGISTRY
    assert DEFAULT_PUBLIC_CONTAINER_REGISTRY.startswith("ghcr.io/")


def test_public_registry_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/example/workbench")
    assert public_container_registry() == "ghcr.io/example/workbench"


def test_official_channel_overrides_must_remain_on_ghcr(monkeypatch) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "registry.example/workbench/releases")
    with pytest.raises(ValueError, match="public release registry must be a GHCR"):
        public_container_registry()

    with pytest.raises(ValueError, match="public development registry must be a GHCR"):
        images.development_image_for_tool(
            "genesis",
            git_sha="c" * 40,
            registry="registry.example/workbench/development",
        )


def test_publish_plan_targets_public_registry_by_default() -> None:
    plan = build_publish_plan(target_registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY)
    # Derived from the contract rather than a magic number, so adding a freely
    # redistributable image does not silently drift this gate. (main's form, kept over an
    # earlier hardcoded 19 from this branch -- which main's 20th tool, foxglove-embed,
    # would have broken immediately.)
    # Two independent gates remove tools from the plan: licence restriction, and
    # having no built/validated artifact to publish. Both are subtracted from the
    # contract-derived total rather than hardcoded, so adding a freely
    # redistributable image does not silently drift this gate.
    assert len(plan) == len(publicly_publishable_tools()) - len(
        set(publicly_publishable_tools()) & set(UNVALIDATED_PUBLICATION_TOOLS)
    )
    # And, since the Isaac re-architecture emptied the restricted set: every image the repo
    # builds and has validated is publishable. This is the assertion that would catch a
    # tool silently dropping out of the plan, which the derived equality above cannot.
    assert len(plan) == len(CONTAINER_IMAGE_NAMES) - len(
        set(CONTAINER_IMAGE_NAMES)
        & (set(RESTRICTED_PUBLICATION_TOOLS) | set(UNVALIDATED_PUBLICATION_TOOLS))
    )
    for item in plan:
        assert item.target_ref.startswith(DEFAULT_PUBLIC_CONTAINER_REGISTRY + "/npa-")
    # npa-foxglove-embed carries only MIT (@foxglove/embed) + Apache-2.0 (Caddy)
    # content plus our own assets, so it belongs in the public set.
    assert "foxglove-embed" in {item.tool for item in plan}


def test_restricted_image_names_cover_every_contract_restricted_image() -> None:
    """The operator-facing excluded list must name every restricted image, derived
    variants included, without any caller hardcoding them.

    The code and packaging contract must agree about what may not be published.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_restricted = {
        name
        for name, entry in contract["images"].items()
        if entry.get("redistribution") == "restricted"
    }
    names = restricted_image_names()
    assert names == sorted(names), "names must be stable/sorted for operator output"
    assert contract_restricted <= set(names), sorted(contract_restricted - set(names))
    assert set(RESTRICTED_DERIVED_IMAGES).isdisjoint(CONTAINER_IMAGE_NAMES)
    assert set(RESTRICTED_DERIVED_IMAGES).isdisjoint(
        publicly_publishable_tools()
    )


def test_contract_marks_active_isaac_images_public_and_runtime_fetch() -> None:
    """The contract must record BOTH facts: publishable, and what earns it.

    `redistribution: public` on its own would look like someone relabelled four restricted
    images; `isaac_runtime_fetch: true` is the claim that earns it, and
    npa/tests/docker/test_packaging_contract.py checks the Dockerfiles implement it.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    for name in ("isaac-lab", "sonic", "groot"):
        entry = contract["images"][name]
        assert entry["redistribution"] == "public", name
        assert entry.get("isaac_runtime_fetch") is True, name
    mujoco = contract["images"]["sonic-mujoco"]
    assert mujoco["redistribution"] == "public"
    assert "runtime-fetch" in mujoco["notes"]


def test_the_restriction_mechanism_still_exists() -> None:
    """The general refusal API remains even with no current restricted image."""
    assert hasattr(images, "OMNIVERSE_RESTRICTED_TOOLS")
    assert hasattr(images, "OMNIVERSE_RESTRICTED_DERIVED_IMAGES")
    assert restricted_image_names() == []
    for symbol in (
        "is_publicly_redistributable",
        "restricted_image_names",
        "omniverse_restricted_image_names",
        "publicly_publishable_tools",
        "is_public_registry",
    ):
        assert callable(getattr(images, symbol)), symbol
    assert (
        "restricted"
        in yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))["redistribution"][
            "classes"
        ]
    ), "the restricted class must remain enforced"


def test_selector_matches_packaging_contract_classification() -> None:
    """Every image the packaging contract marks ``restricted`` must resolve to a
    tool that the selector also treats as non-public (kept in sync).

    Vacuous while nothing is restricted; kept so that classifying something restricted
    again immediately re-arms the sync check.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    # contract image keys that map onto canonical tool keys
    for image_name, entry in contract["images"].items():
        if entry.get("redistribution") != "restricted":
            continue
        if image_name == "sonic-mujoco":
            assert image_name in RESTRICTED_DERIVED_IMAGES
            continue
        tool = image_name
        if tool in CONTAINER_IMAGE_NAMES:
            assert not is_publicly_redistributable(tool), image_name
        else:
            # A future non-canonical restricted image must map to a
            # restricted canonical tool
            assert tool in RESTRICTED_PUBLICATION_TOOLS, image_name


# --- Resolution guard: a restricted tool must never resolve from a public registry ----
#
# The docs tell external consumers to point NPA_REGISTRY at the public release channel. Asking
# for a restricted tool in that state used to silently produce a public image reference
# for something we must never publish. Private registries are unaffected —
# build-your-own is the licensed path, whichever registry that is.


@pytest.mark.parametrize(
    "registry",
    [
        "ghcr.io/nebius/nebius-physical-ai",
        "docker.io/nebius/workbench",
        "quay.io/nebius/workbench",
        "public.ecr.aws/nebius/workbench",
    ],
)
def test_restricted_tools_refuse_to_resolve_from_a_public_registry(
    monkeypatch, registry
) -> None:
    monkeypatch.setattr(images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis"}))
    with pytest.raises(ValueError, match="not publicly redistributable"):
        container_image_for_tool("genesis", registry=registry)


def test_restricted_tools_still_resolve_from_an_operators_own_registry(
    monkeypatch,
) -> None:
    """Build-your-own into a private registry is the licensed path; do not block it."""
    monkeypatch.setattr(images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis"}))
    ref = container_image_for_tool("genesis", registry="registry.example/example")
    assert ref.startswith("registry.example/example/npa-genesis:")


def test_public_registry_detection() -> None:
    assert is_public_registry("ghcr.io/nebius/nebius-physical-ai")
    assert not is_public_registry("GHCR.IO/Operator/Private-Package")
    assert not is_public_registry("registry.example/e00example")
    assert not is_public_registry("")


def test_public_release_override_is_treated_as_public(monkeypatch) -> None:
    """The configured GHCR release namespace is public; other GHCR packages are not."""
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/example/workbench")
    assert is_public_registry("ghcr.io/example/workbench")
    assert is_public_registry(DEFAULT_PUBLIC_CONTAINER_REGISTRY)
    assert not is_public_registry("ghcr.io/example/private")


def test_restricted_tool_refuses_default_official_namespace_after_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/example/workbench")
    monkeypatch.setattr(images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis"}))

    with pytest.raises(ValueError, match="not publicly redistributable"):
        container_image_for_tool(
            "genesis", registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY
        )


def test_restricted_tool_allows_deliberate_operator_ghcr_namespace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/example/workbench")
    monkeypatch.setattr(images, "RESTRICTED_PUBLICATION_TOOLS", frozenset({"genesis"}))

    ref = container_image_for_tool(
        "genesis", registry="ghcr.io/operator/build-your-own"
    )
    assert ref.startswith("ghcr.io/operator/build-your-own/npa-genesis:")


def test_oss_tools_resolve_from_the_public_release_normally() -> None:
    """The guard must not get in the way of the images that ARE publishable."""
    ref = container_image_for_tool(
        "lerobot", registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY
    )
    assert ref.startswith(DEFAULT_PUBLIC_CONTAINER_REGISTRY + "/npa-lerobot:")


# --------------------------------------------------------------------------------------
# Anonymous pullability
#
# Pushing to GHCR is not publishing: a new container package is PRIVATE, and a package
# linked to a repository inherits that repository's access *permissions* but explicitly NOT
# its visibility -- so even a public repo yields private packages. GitHub has no REST API
# to change visibility for organisation-owned packages, so it cannot be automated. Without
# a verification step the publish job copies every image, exits 0 and looks successful
# while nothing is actually pullable, which is a silent false success on the one action in
# this repo that cannot be undone.
# --------------------------------------------------------------------------------------


def test_registry_host_is_split_off_correctly() -> None:
    from npa.deploy.publish_public import _registry_host

    assert (
        _registry_host("ghcr.io/nebius/nebius-physical-ai/npa-lerobot:0.5.1")
        == "ghcr.io"
    )
    assert _registry_host("registry.example/abc/npa-lerobot:0.5.1") == (
        "registry.example"
    )


def test_verify_public_reports_every_private_image(monkeypatch) -> None:
    from npa.deploy import publish_public

    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    private = {plan[0].target_ref, plan[1].target_ref}

    def fake_check(ref: str, **_: object) -> tuple[bool, str]:
        return (
            (False, "HTTP 403 (package is private)")
            if ref in private
            else (True, "HTTP 200")
        )

    monkeypatch.setattr(publish_public, "anonymous_pull_ok", fake_check)
    failures = publish_public.verify_public(plan)

    assert {item.target_ref for item, _ in failures} == private
    assert all("403" in detail for _, detail in failures)


def test_verify_public_exits_non_zero_when_anything_is_private(
    monkeypatch, capsys
) -> None:
    """The whole point: a publish that produced private packages must FAIL the run."""
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (False, "HTTP 403")
    )
    rc = publish_public.main(
        ["--target", "ghcr.io/example/workbench", "--verify-public"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "must be anonymously pullable" in captured.err


def test_verify_public_exits_zero_when_everything_is_public(monkeypatch) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "HTTP 200")
    )
    assert (
        publish_public.main(
            ["--target", "ghcr.io/example/workbench", "--verify-public"]
        )
        == 0
    )


def test_verify_public_does_not_copy_anything(monkeypatch) -> None:
    """--verify-public must never be a publish path in disguise."""
    from npa.deploy import publish_public

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"--verify-public must not copy {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "ok")
    )
    assert (
        publish_public.main(
            ["--target", "ghcr.io/example/workbench", "--verify-public"]
        )
        == 0
    )


def test_verify_parity_reports_missing_and_mismatched_targets(monkeypatch) -> None:
    from npa.deploy import publish_public

    plan = [
        PublishItem(
            tool="cosmos",
            source_ref="source.example/npa-cosmos@sha256:" + "a" * 64,
            target_ref="target.example/npa-cosmos:release",
        ),
        PublishItem(
            tool="lerobot",
            source_ref="source.example/npa-lerobot@sha256:" + "b" * 64,
            target_ref="target.example/npa-lerobot:release",
        ),
        PublishItem(
            tool="groot",
            source_ref="source.example/npa-groot@sha256:" + "c" * 64,
            target_ref="target.example/npa-groot:release",
        ),
    ]
    digests = {
        plan[0].source_ref: "sha256:" + "a" * 64,
        plan[0].target_ref: "sha256:" + "a" * 64,
        plan[1].source_ref: "sha256:" + "b" * 64,
        plan[1].target_ref: "sha256:" + "d" * 64,
        plan[2].source_ref: "sha256:" + "c" * 64,
    }

    def digest(ref: str, **_: object) -> tuple[bool, str]:
        value = digests.get(ref)
        return (True, value) if value else (False, "MANIFEST_UNKNOWN")

    monkeypatch.setattr(publish_public, "_crane_digest", digest)
    failures = publish_public.verify_parity(plan)

    assert failures == [
        (
            plan[1],
            "digest mismatch — source sha256:"
            + "b" * 64
            + "; target sha256:"
            + "d" * 64,
        ),
        (plan[2], "target digest unreadable — MANIFEST_UNKNOWN"),
    ]


def test_verify_parity_is_read_only(monkeypatch) -> None:
    from npa.deploy import publish_public

    plan = [
        PublishItem(
            tool="cosmos",
            source_ref="source.example/npa-cosmos@sha256:" + "a" * 64,
            target_ref="target.example/npa-cosmos:release",
        )
    ]
    monkeypatch.setattr(publish_public, "build_publish_plan", lambda **_: plan)
    monkeypatch.setattr(publish_public, "_preflight_or_explain", lambda *a, **k: plan)
    monkeypatch.setattr(publish_public, "verify_parity", lambda items: [])

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"--verify-parity must not copy {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    assert (
        publish_public.main(["--target", "target.example/workbench", "--verify-parity"])
        == 0
    )


def test_accepted_release_plan_partitions_published_and_pending_tools() -> None:
    from npa.deploy import publish_public

    manifest = images.public_release_manifest()
    plan = publish_public.accepted_release_plan(
        target_registry="ghcr.io/nebius/nebius-physical-ai"
    )

    assert len(plan) == 30
    assert set(manifest["releases"]) | set(manifest["publication_pending"]) == set(
        publicly_publishable_tools()
    )
    assert set(manifest["publication_pending"]) == {"leisaac"}
    for item in plan:
        recorded = manifest["releases"][item.tool]["published_digest"]
        assert item.source_ref.endswith(f"@{recorded}")


def test_verify_accepted_releases_compares_anonymous_live_and_recorded_digests(
    monkeypatch,
) -> None:
    from npa.deploy import publish_public

    plan = publish_public.accepted_release_plan(
        target_registry="ghcr.io/nebius/nebius-physical-ai"
    )[:2]
    expected = {
        item.target_ref: item.source_ref.rpartition("@")[2] for item in plan
    }
    drifted = plan[1]

    def resolve(ref: str, **_: object) -> tuple[bool, str]:
        if ref == drifted.target_ref:
            return True, "sha256:" + "f" * 64
        return True, expected[ref]

    monkeypatch.setattr(publish_public, "anonymous_digest", resolve)

    assert publish_public.verify_accepted_releases(plan) == [
        (
            drifted,
            "published digest mismatch — recorded "
            + expected[drifted.target_ref]
            + "; live sha256:"
            + "f" * 64,
        )
    ]


def test_verify_accepted_releases_mode_never_reads_development_sources(
    monkeypatch,
) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public,
        "anonymous_digest",
        lambda ref, **_: (
            True,
            next(
                item.source_ref.rpartition("@")[2]
                for item in publish_public.accepted_release_plan(
                    target_registry="ghcr.io/nebius/nebius-physical-ai"
                )
                if item.target_ref == ref
            ),
        ),
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("accepted release verification must not read dev tags")

    monkeypatch.setattr(publish_public, "_preflight_or_explain", explode)
    assert (
        publish_public.main(
            [
                "--target",
                "ghcr.io/nebius/nebius-physical-ai",
                "--verify-accepted-releases",
            ]
        )
        == 0
    )


def test_anonymous_check_sends_no_credentials_for_a_private_registry(
    monkeypatch,
) -> None:
    """It must test the UNAUTHENTICATED path, or a private package reads as public.

    Using plain HTTP rather than a crane/docker call is deliberate: those would happily
    reuse an ambient login from an earlier step in the same job.
    """
    from npa.deploy import publish_public

    seen: dict[str, object] = {}

    class FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        url = request if isinstance(request, str) else request.full_url
        seen["url"] = url
        if not isinstance(request, str):
            seen["auth"] = request.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr(publish_public.urllib.request, "urlopen", fake_urlopen)
    ok, detail = publish_public.anonymous_pull_ok(
        "registry.example/abc/npa-lerobot:0.5.1"
    )

    assert ok, detail
    assert seen["url"].startswith(
        "https://registry.example/v2/abc/npa-lerobot/manifests/"
    )
    assert seen.get("auth") is None, (
        "no Authorization header may be sent for a non-GHCR host"
    )


def test_a_token_endpoint_refusal_is_reported_as_a_verdict_not_a_glitch(
    monkeypatch,
) -> None:
    """GHCR can refuse at the token endpoint when a package is private or absent.

    Reporting that as "could not obtain an anonymous token" reads like a network problem and
    invites a pointless retry; it is actually the answer.
    """
    from npa.deploy import publish_public

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        raise publish_public.urllib.error.HTTPError(
            url="https://ghcr.io/token", code=403, msg="Forbidden", hdrs=None, fp=None
        )

    monkeypatch.setattr(publish_public.urllib.request, "urlopen", fake_urlopen)
    ok, detail = publish_public.anonymous_pull_ok(
        "ghcr.io/example/workbench/npa-lerobot:1.0"
    )

    assert not ok
    assert "private or does not exist yet" in detail
    assert "could not obtain" not in detail


# --------------------------------------------------------------------------------------
# Source preflight and the manual-visibility click-through
#
# The copy loop is a sequence of independent `crane copy` subprocesses with no transaction
# around it, and `crane auth login` never contacts the registry, so both a dead credential
# and a missing pinned tag used to surface as "image 7 of 22 failed" with six packages
# already created. These pin the read-only failure that now happens first.
# --------------------------------------------------------------------------------------


def test_the_copy_path_writes_nothing_when_a_source_is_unreadable(
    monkeypatch, capsys, tmp_path
) -> None:
    from npa.deploy import publish_public

    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(
            f"nothing may be copied after a failed preflight: {item.target_ref}"
        )

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(
        publish_public,
        "_crane_manifest_readable",
        lambda ref, **_: (False, "UNAUTHORIZED: authentication required"),
    )

    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])

    assert rc == 1
    assert "nothing was copied" in capsys.readouterr().err
    assert not github_output.exists()


def test_preflight_reports_the_registrys_own_reason(monkeypatch) -> None:
    """UNAUTHORIZED (dead token) and MANIFEST_UNKNOWN (absent tag) need different fixes."""
    from npa.deploy import publish_public

    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "Error: fetching manifest\nMANIFEST_UNKNOWN: manifest unknown"

    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public.subprocess, "run", lambda *a, **k: FakeCompleted()
    )

    ok, detail = publish_public._crane_manifest_readable(
        "cr.example/abc/npa-lerobot:1.0"
    )

    assert not ok
    assert detail == "MANIFEST_UNKNOWN: manifest unknown"


def test_the_preflight_flag_never_copies(monkeypatch) -> None:
    from npa.deploy import publish_public

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"--preflight must not copy {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )

    assert (
        publish_public.main(["--target", "ghcr.io/example/workbench", "--preflight"])
        == 0
    )


@pytest.mark.parametrize("recorded_total", [27, 26, None])
def test_wan_publication_gate_binds_live_trivy_to_recorded_totals(
    monkeypatch, recorded_total: int | None
) -> None:
    from npa.deploy import publish_public

    accepted = copy.deepcopy(images.wan_accepted_image_manifest())
    if recorded_total is None:
        accepted["vulnerability_scan"].pop("critical_total")
        accepted["vulnerability_scan"].pop("critical_unfixed")
    else:
        accepted["vulnerability_scan"]["critical_total"] = recorded_total
        accepted["vulnerability_scan"]["critical_unfixed"] = recorded_total
    monkeypatch.setattr(
        publish_public.images, "wan_accepted_image_manifest", lambda: accepted
    )
    digest = accepted["oci_digest"]

    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, digest)
    )
    monkeypatch.setattr(
        publish_public,
        "_crane_json",
        lambda args: (
            {"config": {}, "layers": [{}]}
            if args[0] == "manifest"
            else {"architecture": "amd64", "os": "linux"}
        ),
    )
    monkeypatch.setattr(
        publish_public,
        "_github_attestation_predicates",
        lambda **_: set(accepted["attestations"]["required_predicates"]),
    )

    class CleanScan:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str):
            self.stdout = stdout

    def clean_scan(args, **_kwargs):
        if args[0] == "/usr/bin/trivy":
            return CleanScan(
                json.dumps(
                    {
                        "Results": [
                            {
                                "Vulnerabilities": [
                                    {
                                        "VulnerabilityID": f"CVE-test-{index}",
                                        "Severity": "CRITICAL",
                                        "FixedVersion": "",
                                    }
                                    for index in range(27)
                                ]
                            }
                        ]
                    }
                )
            )
        return CleanScan(json.dumps({"status": "pass", "findings": []}))

    monkeypatch.setattr(
        publish_public.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(publish_public.subprocess, "run", clean_scan)
    ok, detail = REAL_WAN_PUBLICATION_GATE(
        PublishItem(
            tool="wan2-2",
            source_ref=("ghcr.io/nebius/nebius-physical-ai/npa-wan2-2@" + digest),
            target_ref="ghcr.io/nebius/nebius-physical-ai/npa-wan2-2:accepted",
        )
    )

    if recorded_total == 27:
        assert ok, detail
        assert digest in detail
        assert "residual unfixed CRITICAL findings disclosed: 27" in detail
    else:
        assert not ok
        assert (
            "differs from the accepted manifest" in detail
            or "totals are missing or inconsistent" in detail
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            {
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-fixed",
                        "Severity": "CRITICAL",
                        "FixedVersion": "1.2.3",
                    }
                ]
            },
            "fixed CRITICAL vulnerabilities",
        ),
        ({"Secrets": [{"RuleID": "private-key"}]}, "secret findings"),
    ],
)
def test_wan_live_trivy_gate_fails_closed(
    monkeypatch, result: dict, message: str
) -> None:
    from npa.deploy import publish_public

    completed = subprocess.CompletedProcess(
        ["trivy"], 0, stdout=json.dumps({"Results": [result]}), stderr=""
    )
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/trivy")
    monkeypatch.setattr(
        publish_public.subprocess, "run", lambda *args, **kwargs: completed
    )

    with pytest.raises(RuntimeError, match=message):
        publish_public._scan_wan_trivy_exact_digest(
            "source.example/npa-wan2-2@sha256:" + "1" * 64
        )


def test_wan_live_trivy_gate_uses_digest_pinned_container_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    from npa.deploy import publish_public

    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    (docker_config / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.setattr(
        publish_public.shutil,
        "which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    invoked: list[str] = []

    def clean_scan(args, **_kwargs):
        invoked.extend(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"Results": []}),
            stderr="",
        )

    monkeypatch.setattr(publish_public.subprocess, "run", clean_scan)

    result = publish_public._scan_wan_trivy_exact_digest(
        "source.example/npa-wan2-2@sha256:" + "1" * 64
    )

    assert result == {
        "critical_total": 0,
        "critical_with_fix": 0,
        "critical_unfixed": 0,
        "secrets": 0,
    }
    assert invoked[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert f"{docker_config.resolve()}:/root/.docker:ro" in invoked
    assert publish_public._TRIVY_CONTAINER_IMAGE in invoked
    assert invoked.index(publish_public._TRIVY_CONTAINER_IMAGE) < invoked.index("image")


def test_wan_live_trivy_gate_fails_closed_without_scanner_runtime(
    monkeypatch,
) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(publish_public.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="docker is unavailable"):
        publish_public._scan_wan_trivy_exact_digest(
            "source.example/npa-wan2-2@sha256:" + "1" * 64
        )


def test_wan_publication_gate_refuses_digest_not_bound_to_gpu_proofs(
    monkeypatch,
) -> None:
    from npa.deploy import publish_public

    accepted_digest = "sha256:" + "a" * 64
    observed_digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, observed_digest)
    )
    monkeypatch.setattr(
        publish_public.images,
        "wan_accepted_image_manifest",
        lambda: {"oci_digest": accepted_digest},
    )

    ok, detail = REAL_WAN_PUBLICATION_GATE(
        PublishItem(
            tool="wan2-2",
            source_ref="source.example/npa-wan2-2:retagged",
            target_ref="ghcr.io/example/npa-wan2-2:retagged",
        )
    )

    assert not ok
    assert accepted_digest in detail


def test_ltx_publication_gate_binds_zero_payload_gpu_and_attestation_proofs(
    monkeypatch,
) -> None:
    from npa.deploy import publish_public

    accepted = images.ltx2_accepted_image_manifest()
    digest = accepted["oci_digest"]
    repository = "ghcr.io/nebius/nebius-physical-ai/npa-ltx2"
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, digest)
    )
    monkeypatch.setattr(
        publish_public,
        "_github_attestation_predicates",
        lambda **_: set(accepted["attestations"]["required_predicates"]),
    )
    monkeypatch.setattr(
        publish_public,
        "_crane_json",
        lambda args: {"architecture": "amd64", "os": "linux"},
    )
    monkeypatch.setattr(
        publish_public,
        "_scan_wan_trivy_exact_digest",
        lambda ref: {
            "critical_total": 0,
            "critical_with_fix": 0,
            "critical_unfixed": 0,
            "secrets": 0,
        },
    )
    monkeypatch.setattr(
        publish_public.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps({"status": "pass", "findings": []}), stderr=""
        ),
    )

    ok, detail = REAL_LTX_PUBLICATION_GATE(
        PublishItem(
            tool="ltx2",
            source_ref=f"{repository}@{digest}",
            target_ref=f"{repository}:2.5-rtfetch-20260817",
        )
    )

    assert ok, detail
    assert digest in detail
    assert "SPDX+SLSA" in detail


def test_ltx_publication_gate_refuses_a_nonaccepted_digest(monkeypatch) -> None:
    from npa.deploy import publish_public

    observed = "sha256:" + "a" * 64
    accepted = images.ltx2_accepted_image_manifest()["oci_digest"]
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, observed)
    )

    ok, detail = REAL_LTX_PUBLICATION_GATE(
        PublishItem(
            tool="ltx2",
            source_ref="ghcr.io/nebius/nebius-physical-ai/npa-ltx2:retagged",
            target_ref="ghcr.io/nebius/nebius-physical-ai/npa-ltx2:release",
        )
    )

    assert not ok
    assert accepted in detail


def test_wan_publication_gate_refuses_an_extra_unscanned_platform(monkeypatch) -> None:
    from npa.deploy import publish_public

    index_digest = "sha256:" + "1" * 64
    platform_digest = "sha256:" + "2" * 64
    attestation_digest = "sha256:" + "3" * 64
    proof = {
        "run_id": "run",
        "observed_image_id_digest": index_digest,
        "mp4_sha256": "4" * 64,
        "rrd_sha256": "5" * 64,
        "rrd_manifest_sha256": "6" * 64,
    }
    accepted = {
        "oci_digest": index_digest,
        "amd64_manifest": platform_digest,
        "runtime_requirements_sha256": "7" * 64,
        "source": {"revision": "source-ref"},
        "model": {"revision": "model-ref"},
        "tokenizer": {"revision": "tokenizer-ref"},
        "runtime_acceptance": {"manifest_sha256": "9" * 64},
        "payload_scan": {
            "report_sha256": "a" * 64,
            "archives_scanned": 2,
            "findings": 0,
        },
        "single_gpu_proof": {**proof, "gpu_count": 1},
        "distributed_proof": {**proof, "gpu_count": 4},
        "vulnerability_scan": {
            "report_sha256": "b" * 64,
            "critical_total": 0,
            "critical_with_fix": 0,
            "secrets": 0,
        },
    }
    index = {
        "manifests": [
            {
                "digest": platform_digest,
                "platform": {"architecture": "amd64", "os": "linux"},
            },
            {
                "digest": attestation_digest,
                "annotations": {
                    "vnd.docker.reference.type": "attestation-manifest",
                    "vnd.docker.reference.digest": platform_digest,
                },
            },
            {
                "digest": "sha256:" + "8" * 64,
                "platform": {"architecture": "arm64", "os": "linux"},
            },
        ]
    }
    monkeypatch.setattr(
        publish_public, "_crane_digest", lambda ref, **_: (True, index_digest)
    )
    monkeypatch.setattr(
        publish_public.images, "wan_accepted_image_manifest", lambda: accepted
    )
    monkeypatch.setattr(publish_public, "_crane_json", lambda args: index)

    ok, detail = REAL_WAN_PUBLICATION_GATE(
        PublishItem(
            tool="wan2-2",
            source_ref="source.example/npa-wan2-2:accepted",
            target_ref="ghcr.io/example/npa-wan2-2:accepted",
        )
    )

    assert not ok
    assert "one scanned platform manifest" in detail


def test_wan_publication_gate_blocks_copy_before_any_write(monkeypatch, capsys) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )
    monkeypatch.setattr(
        publish_public,
        "verify_wan_publication_source",
        lambda item: (False, "SLSA provenance is absent"),
    )

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"publication gate must prevent copying {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])

    assert rc == 1
    assert "SLSA provenance is absent" in capsys.readouterr().err


def test_a_successful_copy_still_fails_while_the_packages_are_private(
    monkeypatch, capsys, tmp_path
) -> None:
    """Copying every image and exiting 0 would be the silent false success we guard against."""
    from npa.deploy import publish_public

    copied: list[str] = []
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )
    monkeypatch.setattr(
        publish_public, "_crane_copy", lambda item: copied.append(item.target_ref)
    )
    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (False, "HTTP 403")
    )

    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])
    captured = capsys.readouterr()

    assert rc == 1
    assert copied, "the copy itself must still have happened"
    assert github_output.read_text(encoding="utf-8") == "copy_phase_completed=true\n"
    assert "release-tag copy completed" in captured.err, (
        "must not read as a failed copy"
    )
    # The click-through list is the whole point: no hunting for 20-odd packages by hand.
    assert "NOT PUBLIC" in captured.out


def test_a_copy_exits_zero_only_once_the_packages_are_public(monkeypatch) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )
    monkeypatch.setattr(publish_public, "_crane_copy", lambda item: None)
    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "HTTP 200")
    )

    assert publish_public.main(["--target", "ghcr.io/example/workbench"]) == 0


def test_crane_copy_skips_a_target_with_the_exact_source_digest(
    monkeypatch, capsys
) -> None:
    """Repeat publishes must prove equality without invoking the registry write path."""
    from npa.deploy import publish_public
    from npa.deploy.publish_public import PublishItem

    item = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot@sha256:" + "a" * 64,
        target_ref="target.example/npa-lerobot:1.0",
    )
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public,
        "_crane_digest",
        lambda ref, **_: (True, "sha256:identical"),
    )

    def no_copy(*args, **kwargs) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"matching digests must not invoke crane copy: {args}")

    monkeypatch.setattr(publish_public.subprocess, "run", no_copy)

    assert publish_public._crane_copy(item) is False
    assert "Already current; skipping copy" in capsys.readouterr().out


def test_crane_digest_returns_the_registry_digest(monkeypatch) -> None:
    from npa.deploy import publish_public

    class Result:
        returncode = 0
        stdout = "sha256:current\n"
        stderr = ""

    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")

    def run(args, **kwargs):  # noqa: ANN001, ANN202 - subprocess test double
        calls.append((args, kwargs))
        return Result()

    monkeypatch.setattr(publish_public.subprocess, "run", run)

    assert publish_public._crane_digest("registry.example/image:1") == (
        True,
        "sha256:current",
    )
    assert calls[0][0] == ["/usr/bin/crane", "digest", "registry.example/image:1"]
    assert calls[0][1]["check"] is False


def test_crane_digest_preserves_the_registry_error(monkeypatch) -> None:
    from npa.deploy import publish_public

    class Result:
        returncode = 1
        stdout = ""
        stderr = "Error: fetching digest\nMANIFEST_UNKNOWN: manifest unknown\n"

    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public.subprocess, "run", lambda *args, **kwargs: Result()
    )

    assert publish_public._crane_digest("registry.example/image:missing") == (
        False,
        "MANIFEST_UNKNOWN: manifest unknown",
    )


def test_crane_copy_updates_a_target_with_a_different_digest(
    monkeypatch, capsys
) -> None:
    from npa.deploy import publish_public
    from npa.deploy.publish_public import PublishItem

    item = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot@sha256:" + "a" * 64,
        target_ref="target.example/npa-lerobot:1.0",
    )
    target_reads = iter([(True, "sha256:old"), (True, "sha256:new")])
    calls: list[list[str]] = []
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public,
        "_crane_digest",
        lambda ref, **_: (
            (True, "sha256:new") if ref == item.source_ref else next(target_reads)
        ),
    )
    monkeypatch.setattr(
        publish_public.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args),
    )

    assert publish_public._crane_copy(item) is True
    assert calls == [["/usr/bin/crane", "copy", item.source_ref, item.target_ref]]
    assert "Digest changed; copying" in capsys.readouterr().out


@pytest.mark.parametrize(
    "target_error",
    [
        "MANIFEST_UNKNOWN: manifest unknown",
        "NAME_UNKNOWN: repository name not known",
    ],
)
def test_crane_copy_creates_a_missing_target(monkeypatch, target_error: str) -> None:
    """Only authoritative absence permits a release-tag write."""
    from npa.deploy import publish_public
    from npa.deploy.publish_public import PublishItem

    item = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot@sha256:" + "a" * 64,
        target_ref="target.example/npa-lerobot:1.0",
    )
    target_reads = iter([(False, target_error), (True, "sha256:new")])
    calls: list[list[str]] = []
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public,
        "_crane_digest",
        lambda ref, **_: (
            (True, "sha256:new") if ref == item.source_ref else next(target_reads)
        ),
    )
    monkeypatch.setattr(
        publish_public.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args),
    )

    assert publish_public._crane_copy(item) is True
    assert calls == [["/usr/bin/crane", "copy", item.source_ref, item.target_ref]]


def test_every_copy_refuses_a_mutable_source_tag(monkeypatch) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    with pytest.raises(RuntimeError, match="pinned by exact OCI digest"):
        publish_public._crane_copy(
            PublishItem(
                tool="cosmos2-transfer",
                source_ref="source.example/npa-cosmos2-transfer:mutable",
                target_ref="target.example/npa-cosmos2-transfer:mutable",
            )
        )


def test_crane_copy_refuses_an_unknown_target_digest_failure(monkeypatch) -> None:
    """A transient read failure must not turn into a blind repeat write."""
    from npa.deploy import publish_public
    from npa.deploy.publish_public import PublishItem

    item = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot@sha256:" + "a" * 64,
        target_ref="target.example/npa-lerobot:1.0",
    )
    digests = {
        item.source_ref: (True, "sha256:new"),
        item.target_ref: (False, "timed out after 60s"),
    }
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(
        publish_public,
        "_crane_digest",
        lambda ref, **_: digests[ref],
    )

    with pytest.raises(RuntimeError, match="refusing to copy"):
        publish_public._crane_copy(item)


def test_repeat_publish_skips_all_matching_copies_but_still_verifies(
    monkeypatch, capsys
) -> None:
    """Incrementality removes writes, not the final anonymous-public assertion."""
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    verified: list[str] = []
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok")
    )
    monkeypatch.setattr(publish_public, "_crane_copy", lambda item: False)

    def public(ref: str, **_: object) -> tuple[bool, str]:
        verified.append(ref)
        return True, "HTTP 200"

    monkeypatch.setattr(publish_public, "anonymous_pull_ok", public)

    assert publish_public.main(["--target", "ghcr.io/example/workbench"]) == 0
    output = capsys.readouterr().out
    assert "Copied 0 image(s)." in output
    assert f"Skipped {len(plan)} already-current image(s)." in output
    assert verified == [item.target_ref for item in plan]


def test_a_wholesale_unauthorized_preflight_blames_the_credential(
    monkeypatch, capsys
) -> None:
    """All reads failing points to the GHCR package credential, not one package grant."""
    from npa.deploy import publish_public

    monkeypatch.setattr(
        publish_public,
        "_crane_manifest_readable",
        lambda ref, **_: (
            False,
            "UNAUTHORIZED: authentication required: failed to get profile",
        ),
    )

    rc = publish_public.main(["--target", "ghcr.io/example/workbench", "--preflight"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "Every read was denied" in err
    assert "GITHUB_TOKEN" in err
    assert "crane auth login ghcr.io" in err
    assert "lacks\nviewer" not in err, "a per-repository role hint would misdirect here"


def test_a_partial_preflight_failure_blames_the_role_or_the_tag(
    monkeypatch, capsys
) -> None:
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    broken = plan[0].source_ref

    monkeypatch.setattr(
        publish_public,
        "_crane_manifest_readable",
        lambda ref, **_: (
            (False, "MANIFEST_UNKNOWN: manifest unknown")
            if ref == broken
            else (True, "ok")
        ),
    )

    rc = publish_public.main(["--target", "ghcr.io/example/workbench", "--preflight"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "MANIFEST_UNKNOWN" in err
    assert "Every read was denied" not in err


# --------------------------------------------------------------------------------------
# Images that were never built
#
# The plan comes from the packaging contract, which records what this repo BUILDS. The
# registry holds what someone actually pushed. Run #2 of the workflow found five images in
# the gap: npa-cosmos-curate, npa-cosmos-evaluator, npa-cosmos3 and npa-foxglove-embed had
# no repository at all (NAME_UNKNOWN -- landed with a Dockerfile, a contract entry and a pin,
# but were never built), and npa-cosmos2-transfer had a repository but not the pinned tag
# (MANIFEST_UNKNOWN). Eighteen images were ready and none of them could be published.
#
# Absence is a legitimate state for a young tool, so it must be skippable. A denial never is:
# skipping it would quietly shrink the published set while reporting success.
# --------------------------------------------------------------------------------------

# Historical provider error strings keep the classifier pinned against real output.
_NAME_UNKNOWN = (
    "NAME_UNKNOWN: repository name not known to registry: Entity Folder not found for "
    "registry e00example"
)
_MANIFEST_UNKNOWN = (
    "MANIFEST_UNKNOWN: manifest unknown: Tag not found for manifest npa-x:null"
)
_FAILED_TO_GET_PROFILE = "UNAUTHORIZED: authentication required: failed to get profile"


@pytest.mark.parametrize(
    ("detail", "kind"),
    [
        (_NAME_UNKNOWN, "missing"),
        (_MANIFEST_UNKNOWN, "missing"),
        (_FAILED_TO_GET_PROFILE, "denied"),
        ("DENIED: requested access to the resource is denied", "denied"),
        ("timed out after 60s", "other"),
        ("crane exited 137", "other"),
    ],
)
def test_preflight_failures_are_classified_by_what_they_require(detail, kind) -> None:
    from npa.deploy.publish_public import classify_preflight_failure

    assert classify_preflight_failure(detail) == kind


def test_a_denial_that_also_says_name_unknown_is_never_treated_as_absence() -> None:
    """A registry may answer NAME_UNKNOWN for a repository the identity cannot see.

    Reading that as "not built yet" would silently drop a publishable release,
    so denial has to win over absence.
    """
    from npa.deploy.publish_public import classify_preflight_failure

    assert (
        classify_preflight_failure("UNAUTHORIZED: NAME_UNKNOWN: not visible")
        == "denied"
    )


def _run2_readability(plan):
    """The catalog fixture has 19 readable images, 4 absent repos, and 1 absent tag."""
    never_built = {
        "npa-cosmos-curate",
        "npa-cosmos-evaluator",
        "npa-cosmos3",
        "npa-foxglove-embed",
    }
    unpushed_tag = {"npa-cosmos2-transfer"}

    def readable(ref: str, **_: object) -> tuple[bool, str]:
        image = ref.rsplit("/", 1)[-1].split(":", 1)[0]
        if image in never_built:
            return False, _NAME_UNKNOWN
        if image in unpushed_tag:
            return False, _MANIFEST_UNKNOWN
        return True, "ok"

    return readable


def test_unbuilt_images_block_the_publish_by_default(monkeypatch, capsys) -> None:
    """Silently publishing a subset would be the wrong default: a pin regression that
    dropped an image would look exactly like success."""
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", _run2_readability(plan)
    )

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError("nothing may be copied without --skip-missing")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)

    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])
    err = capsys.readouterr().err

    assert rc == 1
    assert f"5 of {len(plan)}" in err
    # Both codes must survive into the explanation: they need different fixes, and an
    # operator greps for the registry's own wording.
    assert "NAME_UNKNOWN" in err and "never been pushed" in err
    assert "MANIFEST_UNKNOWN" in err and "unpushed build" in err
    assert "--skip-missing" in err, (
        "the way forward has to be named where it is discovered"
    )


def test_skip_missing_publishes_the_ready_images_and_names_the_skipped(
    monkeypatch, capsys
) -> None:
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", _run2_readability(plan)
    )
    copied: list[str] = []
    monkeypatch.setattr(
        publish_public, "_crane_copy", lambda item: copied.append(item.target_ref)
    )
    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "HTTP 200")
    )

    rc = publish_public.main(
        ["--target", "ghcr.io/example/workbench", "--skip-missing"]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert len(copied) == len(plan) - 5
    for image in ("npa-cosmos3", "npa-foxglove-embed", "npa-cosmos2-transfer"):
        assert not any(f"/{image}:" in ref for ref in copied), image
        # Skipping quietly would leave a hole in the mirror nobody knew about.
        assert image in captured.err, image
    assert any("/npa-lerobot:" in ref for ref in copied), (
        "ready images must still publish"
    )
    assert f"Copied {len(plan) - 5} image(s)." in captured.out


def test_skip_missing_never_skips_past_a_denial(monkeypatch, capsys) -> None:
    """The one case that must still stop the run: mixing absence with a permission fault.

    Skipping the denied image would publish a smaller set and exit 0, which is the silent
    false success this whole path exists to prevent.
    """
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    denied = plan[3].source_ref

    def readable(ref: str, **_: object) -> tuple[bool, str]:
        if ref == denied:
            return False, _FAILED_TO_GET_PROFILE
        return (False, _NAME_UNKNOWN) if ref == plan[0].source_ref else (True, "ok")

    monkeypatch.setattr(publish_public, "_crane_manifest_readable", readable)

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError("a denial must stop the run even with --skip-missing")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)

    rc = publish_public.main(
        ["--target", "ghcr.io/example/workbench", "--skip-missing"]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "NOT absence" in err
    assert denied in err, "name the image that blocked the run"
    assert "lacks" in err and "viewer" in err


def test_skip_missing_preflight_alone_reports_success(monkeypatch) -> None:
    """So a dry run can confirm the publish would proceed before anything is written."""
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", _run2_readability(plan)
    )

    assert (
        publish_public.main(
            ["--target", "ghcr.io/example/workbench", "--skip-missing", "--preflight"]
        )
        == 0
    )


def test_verify_public_with_skip_missing_ignores_the_unpublished(
    monkeypatch, capsys
) -> None:
    """The checklist must not list packages nobody tried to publish — those links 404."""
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", _run2_readability(plan)
    )
    monkeypatch.setattr(
        publish_public, "anonymous_pull_ok", lambda ref, **_: (False, "HTTP 403")
    )

    rc = publish_public.main(
        [
            "--target",
            "ghcr.io/example/workbench",
            "--skip-missing",
            "--verify-public",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert captured.out.count("NOT PUBLIC") == len(plan) - 5
    # Match whole package names: "npa-cosmos3-reason" contains "npa-cosmos3" as a substring
    # and IS readable, so a substring check would fail for the wrong reason.
    listed = set(
        re.findall(
            r"^\s+ghcr\.io/example/workbench/(npa-[^:]+):.*NOT PUBLIC",
            captured.out,
            re.MULTILINE,
        )
    )
    assert "npa-cosmos3" not in listed
    assert "npa-cosmos3-reason" in listed, (
        "a readable image whose name shares a prefix stays"
    )
    assert listed.isdisjoint(
        {
            "npa-cosmos3",
            "npa-cosmos-curate",
            "npa-cosmos-evaluator",
            "npa-foxglove-embed",
            "npa-cosmos2-transfer",
        }
    )


def test_the_post_copy_verification_only_covers_what_was_copied(
    monkeypatch, capsys
) -> None:
    """Verifying the skipped ones too would fail a publish that did everything asked of it."""
    from npa.deploy import publish_public

    plan = publish_public.build_publish_plan(
        target_registry="ghcr.io/example/workbench"
    )
    monkeypatch.setattr(
        publish_public, "_crane_manifest_readable", _run2_readability(plan)
    )
    monkeypatch.setattr(publish_public, "_crane_copy", lambda item: None)

    verified: list[str] = []

    def anon(ref: str, **_: object) -> tuple[bool, str]:
        verified.append(ref)
        return True, "HTTP 200"

    monkeypatch.setattr(publish_public, "anonymous_pull_ok", anon)

    rc = publish_public.main(
        ["--target", "ghcr.io/example/workbench", "--skip-missing"]
    )

    assert rc == 0
    assert len(verified) == len(plan) - 5
    assert not any("npa-cosmos3:" in ref for ref in verified)
