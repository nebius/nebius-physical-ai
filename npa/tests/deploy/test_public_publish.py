"""License-guarded public-registry publishing.

Nebius CR has no anonymous/public mode, so public exposure means mirroring the
OSS-redistributable image subset to a public registry. These tests lock the license
boundary: whatever is classified non-redistributable must never be selected for a public
registry, and the selector must stay in sync with the packaging contract's
``redistribution:`` fields.

``OMNIVERSE_RESTRICTED_TOOLS`` is currently EMPTY — the four Isaac images were
re-architected to fetch Isaac Sim / Isaac Lab at first run under the operator's own EULA
acceptance instead of baking it, so every workbench tool is now publishable. That makes
the boundary tests the delicate ones: asserting "nothing is restricted" would pass just
as well against a guard that had been deleted. So the tests that exercise the refusal
monkeypatch a synthetic restricted tool in, proving the mechanism still bites while its
membership is empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.deploy import images
from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    DEFAULT_PUBLIC_CONTAINER_REGISTRY,
    OMNIVERSE_RESTRICTED_DERIVED_IMAGES,
    OMNIVERSE_RESTRICTED_TOOLS,
    container_image_for_tool,
    is_public_registry,
    is_publicly_redistributable,
    omniverse_restricted_image_names,
    public_container_registry,
    publicly_publishable_tools,
)
from npa.deploy.publish_public import build_publish_plan

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml"


def test_isaac_images_are_no_longer_restricted() -> None:
    """Removing baked Omniverse Kit made the Isaac images publishable.

    They now fetch Isaac Sim / Isaac Lab at first run from pypi.nvidia.com under the
    operator's own EULA acceptance and ship no NVIDIA Isaac bytes, verified against the
    built image by npa/scripts/scan_image_omniverse_payload.py (isaac-lab: 83,043 entries
    scanned; sonic: 125,655 entries; both VERDICT clean).
    """
    for tool in ("isaac-lab", "sonic", "groot"):
        assert is_publicly_redistributable(tool), tool


def test_no_tool_is_currently_restricted() -> None:
    """Nothing is excluded any more, which took THREE separate fixes for sonic.

    Omniverse Kit was only the first: sonic also baked gated model weights (git-LFS
    smudging) and NVIDIA Omniverse 3D assets (the RoboCasa asset library under
    decoupled_wbc/dexmg). Both were found by scanning the built image, and neither was
    visible in the Dockerfile. The scan that clears it:
    npa-sonic:0.1.2-rtfetch-rc5, 125,655 entries, 16 allowlisted paths, VERDICT clean.
    """
    assert OMNIVERSE_RESTRICTED_TOOLS == frozenset()
    assert OMNIVERSE_RESTRICTED_DERIVED_IMAGES == frozenset()
    for tool in ("isaac-lab", "sonic", "groot"):
        assert is_publicly_redistributable(tool), tool


def test_public_set_excludes_every_restricted_tool(monkeypatch) -> None:
    """The exclusion still works. Monkeypatched, because the real set is empty and an
    all-inclusive selector would satisfy an assertion over an empty set trivially."""
    monkeypatch.setattr(images, "OMNIVERSE_RESTRICTED_TOOLS", frozenset({"genesis", "cosmos"}))
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
    ):
        assert tool in public, tool
    assert public == set(CONTAINER_IMAGE_NAMES) - OMNIVERSE_RESTRICTED_TOOLS


def test_publish_plan_now_includes_the_isaac_images() -> None:
    """The point of the re-architecture: these are publishable at last."""
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    names = {item.source_ref.rsplit("/", 1)[-1].split(":", 1)[0] for item in plan}
    for image in ("npa-isaac-lab", "npa-sonic", "npa-groot"):
        assert image in names, image
    # sonic-mujoco is a sonic variant, so it ships through sonic's image manifest rather
    # than as its own tool key.
    assert "npa-sonic-mujoco" not in names
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
    monkeypatch.setattr(images, "OMNIVERSE_RESTRICTED_TOOLS", frozenset({"genesis"}))
    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    names = {item.source_ref.rsplit("/", 1)[-1].split(":", 1)[0] for item in plan}
    assert "npa-genesis" not in names
    # sonic is publishable under this monkeypatched set, so the plan must contain it -
    # proving the refusal followed the patched set instead of a captured one.
    assert "npa-sonic" in names


def test_publish_plan_requires_a_target() -> None:
    with pytest.raises(ValueError):
        build_publish_plan(target_registry="")


def test_publish_plan_copies_the_pinned_tag_unchanged() -> None:
    """A mirror must serve the same ``name:tag`` the primary registry serves, or
    every pin in the repo (and every customer's) breaks against the mirror."""
    plan = build_publish_plan(
        target_registry="ghcr.io/example/workbench",
        source_registry="cr.eu-north1.nebius.cloud/example",
    )
    assert plan
    for item in plan:
        source_image = item.source_ref.rsplit("/", 1)[-1]
        target_image = item.target_ref.rsplit("/", 1)[-1]
        assert source_image == target_image, item


def test_public_registry_defaults_to_ghcr(monkeypatch) -> None:
    monkeypatch.delenv("NPA_PUBLIC_REGISTRY", raising=False)
    assert public_container_registry() == DEFAULT_PUBLIC_CONTAINER_REGISTRY
    assert DEFAULT_PUBLIC_CONTAINER_REGISTRY.startswith("ghcr.io/")


def test_public_registry_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "docker.io/nebius/workbench")
    assert public_container_registry() == "docker.io/nebius/workbench"


def test_publish_plan_targets_public_registry_by_default() -> None:
    plan = build_publish_plan(target_registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY)
    # Derived from the contract rather than a magic number, so adding a freely
    # redistributable image does not silently drift this gate. (main's form, kept over an
    # earlier hardcoded 19 from this branch -- which main's 20th tool, foxglove-embed,
    # would have broken immediately.)
    assert len(plan) == len(publicly_publishable_tools())
    # And, since the Isaac re-architecture emptied the restricted set: every image the repo
    # builds is now publishable. This is the assertion that would catch a tool silently
    # dropping out of the plan, which the derived equality above cannot.
    assert len(plan) == len(CONTAINER_IMAGE_NAMES) - len(OMNIVERSE_RESTRICTED_TOOLS)
    for item in plan:
        assert item.target_ref.startswith(DEFAULT_PUBLIC_CONTAINER_REGISTRY + "/npa-")
    # npa-foxglove-embed carries only MIT (@foxglove/embed) + Apache-2.0 (Caddy)
    # content plus our own assets, so it belongs in the public set.
    assert "foxglove-embed" in {item.tool for item in plan}


def test_restricted_image_names_cover_every_contract_restricted_image() -> None:
    """The operator-facing excluded list must name every restricted image, derived
    variants included, without any caller hardcoding them.

    Both sides are currently empty, which is the property being locked: the code and the
    packaging contract must agree about what may not be published.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_restricted = {
        name
        for name, entry in contract["images"].items()
        if entry.get("redistribution") == "restricted"
    }
    names = omniverse_restricted_image_names()
    assert names == sorted(names), "names must be stable/sorted for operator output"
    assert contract_restricted <= set(names), sorted(contract_restricted - set(names))
    assert set(OMNIVERSE_RESTRICTED_DERIVED_IMAGES).isdisjoint(CONTAINER_IMAGE_NAMES)
    assert set(OMNIVERSE_RESTRICTED_DERIVED_IMAGES).isdisjoint(publicly_publishable_tools())


def test_contract_marks_the_isaac_images_public_and_runtime_fetch() -> None:
    """The contract must record BOTH facts: publishable, and what earns it.

    `redistribution: public` on its own would look like someone relabelled four restricted
    images; `isaac_runtime_fetch: true` is the claim that earns it, and
    npa/tests/docker/test_packaging_contract.py checks the Dockerfiles implement it.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    for name in ("isaac-lab", "sonic", "sonic-mujoco", "groot"):
        entry = contract["images"][name]
        assert entry["redistribution"] == "public", name
        assert entry.get("isaac_runtime_fetch") is True, name


def test_the_restriction_mechanism_still_exists() -> None:
    """Deliberately kept with an empty membership, not deleted.

    The next runtime we cannot ship needs exactly this machinery, and a mechanism that
    gets deleted when unused has to be rebuilt and re-reviewed under time pressure.
    """
    assert hasattr(images, "OMNIVERSE_RESTRICTED_TOOLS")
    assert hasattr(images, "OMNIVERSE_RESTRICTED_DERIVED_IMAGES")
    assert omniverse_restricted_image_names() == []
    for symbol in (
        "is_publicly_redistributable",
        "omniverse_restricted_image_names",
        "publicly_publishable_tools",
        "is_public_registry",
    ):
        assert callable(getattr(images, symbol)), symbol
    assert "restricted" in yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))[
        "redistribution"
    ]["classes"], "the restricted class must survive having no members"


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
        # sonic-mujoco is a sonic variant (covered by the "sonic" restriction)
        tool = "sonic" if image_name == "sonic-mujoco" else image_name
        if tool in CONTAINER_IMAGE_NAMES:
            assert not is_publicly_redistributable(tool), image_name
        else:
            # non-canonical restricted image (e.g. sonic-mujoco) must map to a
            # restricted canonical tool
            assert tool in OMNIVERSE_RESTRICTED_TOOLS, image_name


# --- Resolution guard: a restricted tool must never resolve from a public registry ----
#
# The docs tell external consumers to point NPA_REGISTRY at the public mirror. Asking
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
    monkeypatch.setattr(images, "OMNIVERSE_RESTRICTED_TOOLS", frozenset({"genesis"}))
    with pytest.raises(ValueError, match="not publicly redistributable"):
        container_image_for_tool("genesis", registry=registry)


def test_restricted_tools_still_resolve_from_an_operators_own_registry(monkeypatch) -> None:
    """Build-your-own into a private registry is the licensed path; do not block it."""
    monkeypatch.setattr(images, "OMNIVERSE_RESTRICTED_TOOLS", frozenset({"genesis"}))
    ref = container_image_for_tool("genesis", registry="cr.eu-north1.nebius.cloud/example")
    assert ref.startswith("cr.eu-north1.nebius.cloud/example/npa-genesis:")


def test_public_registry_detection() -> None:
    assert is_public_registry("ghcr.io/nebius/nebius-physical-ai")
    assert is_public_registry("GHCR.IO/Nebius/Workbench")
    assert not is_public_registry("cr.eu-north1.nebius.cloud/e00example")
    assert not is_public_registry("")


def test_public_mirror_override_is_treated_as_public(monkeypatch) -> None:
    """Whatever is configured as the mirror is public, even on a private-looking host."""
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "mirror.example.com/workbench")
    assert is_public_registry("mirror.example.com/workbench")


def test_oss_tools_resolve_from_the_public_mirror_normally() -> None:
    """The guard must not get in the way of the images that ARE publishable."""
    ref = container_image_for_tool("lerobot", registry=DEFAULT_PUBLIC_CONTAINER_REGISTRY)
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

    assert _registry_host("ghcr.io/nebius/nebius-physical-ai/npa-lerobot:0.5.1") == "ghcr.io"
    assert _registry_host("cr.eu-north1.nebius.cloud/abc/npa-lerobot:0.5.1") == (
        "cr.eu-north1.nebius.cloud"
    )


def test_verify_public_reports_every_private_image(monkeypatch) -> None:
    from npa.deploy import publish_public

    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    private = {plan[0].target_ref, plan[1].target_ref}

    def fake_check(ref: str, **_: object) -> tuple[bool, str]:
        return (False, "HTTP 403 (package is private)") if ref in private else (True, "HTTP 200")

    monkeypatch.setattr(publish_public, "anonymous_pull_ok", fake_check)
    failures = publish_public.verify_public(plan)

    assert {item.target_ref for item, _ in failures} == private
    assert all("403" in detail for _, detail in failures)


def test_verify_public_exits_non_zero_when_anything_is_private(monkeypatch, capsys) -> None:
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
    # The message has to tell an operator exactly what to do, because there is no API for it.
    assert "Change visibility" in captured.err
    assert "irreversible" in captured.err


def test_verify_public_exits_zero_when_everything_is_public(monkeypatch) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "HTTP 200"))
    assert publish_public.main(["--target", "ghcr.io/example/workbench", "--verify-public"]) == 0


def test_verify_public_does_not_copy_anything(monkeypatch) -> None:
    """--verify-public must never be a publish path in disguise."""
    from npa.deploy import publish_public

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"--verify-public must not copy {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "ok"))
    assert publish_public.main(["--target", "ghcr.io/example/workbench", "--verify-public"]) == 0


def test_anonymous_check_sends_no_credentials_for_a_private_registry(monkeypatch) -> None:
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
        "cr.eu-north1.nebius.cloud/abc/npa-lerobot:0.5.1"
    )

    assert ok, detail
    assert seen["url"].startswith("https://cr.eu-north1.nebius.cloud/v2/abc/npa-lerobot/manifests/")
    assert seen.get("auth") is None, "no Authorization header may be sent for a non-GHCR host"


def test_a_token_endpoint_refusal_is_reported_as_a_verdict_not_a_glitch(monkeypatch) -> None:
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
    ok, detail = publish_public.anonymous_pull_ok("ghcr.io/example/workbench/npa-lerobot:1.0")

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


def test_the_copy_path_writes_nothing_when_a_source_is_unreadable(monkeypatch, capsys) -> None:
    from npa.deploy import publish_public

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"nothing may be copied after a failed preflight: {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(
        publish_public,
        "_crane_manifest_readable",
        lambda ref, **_: (False, "UNAUTHORIZED: authentication required"),
    )

    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])

    assert rc == 1
    assert "nothing was copied" in capsys.readouterr().err


def test_preflight_reports_the_registrys_own_reason(monkeypatch) -> None:
    """UNAUTHORIZED (dead token) and MANIFEST_UNKNOWN (absent tag) need different fixes."""
    from npa.deploy import publish_public

    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "Error: fetching manifest\nMANIFEST_UNKNOWN: manifest unknown"

    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/usr/bin/crane")
    monkeypatch.setattr(publish_public.subprocess, "run", lambda *a, **k: FakeCompleted())

    ok, detail = publish_public._crane_manifest_readable("cr.example/abc/npa-lerobot:1.0")

    assert not ok
    assert detail == "MANIFEST_UNKNOWN: manifest unknown"


def test_the_preflight_flag_never_copies(monkeypatch) -> None:
    from npa.deploy import publish_public

    def explode(item) -> None:  # pragma: no cover - must not run
        raise AssertionError(f"--preflight must not copy {item.target_ref}")

    monkeypatch.setattr(publish_public, "_crane_copy", explode)
    monkeypatch.setattr(publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok"))

    assert publish_public.main(["--target", "ghcr.io/example/workbench", "--preflight"]) == 0


def test_a_successful_copy_still_fails_while_the_packages_are_private(monkeypatch, capsys) -> None:
    """Copying every image and exiting 0 would be the silent false success we guard against."""
    from npa.deploy import publish_public

    copied: list[str] = []
    monkeypatch.setattr(publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok"))
    monkeypatch.setattr(publish_public, "_crane_copy", lambda item: copied.append(item.target_ref))
    monkeypatch.setattr(publish_public, "anonymous_pull_ok", lambda ref, **_: (False, "HTTP 403"))

    rc = publish_public.main(["--target", "ghcr.io/example/workbench"])
    captured = capsys.readouterr()

    assert rc == 1
    assert copied, "the copy itself must still have happened"
    assert "The copy succeeded" in captured.err, "must not read as a failed copy"
    # The click-through list is the whole point: no hunting for 20-odd packages by hand.
    assert "/packages/container/" in captured.out


def test_a_copy_exits_zero_only_once_the_packages_are_public(monkeypatch) -> None:
    from npa.deploy import publish_public

    monkeypatch.setattr(publish_public, "_crane_manifest_readable", lambda ref, **_: (True, "ok"))
    monkeypatch.setattr(publish_public, "_crane_copy", lambda item: None)
    monkeypatch.setattr(publish_public, "anonymous_pull_ok", lambda ref, **_: (True, "HTTP 200"))

    assert publish_public.main(["--target", "ghcr.io/example/workbench"]) == 0


def test_settings_url_encodes_the_repository_nested_package_name() -> None:
    from npa.deploy.publish_public import package_settings_url

    url = package_settings_url("ghcr.io/nebius/nebius-physical-ai/npa-lerobot:0.5.1")

    # GHCR package name is "<repo>/<image>"; the slash is percent-encoded in the path, and
    # a raw slash here silently 404s.
    assert url == (
        "https://github.com/orgs/nebius/packages/container/"
        "nebius-physical-ai%2Fnpa-lerobot/settings"
    )


def test_settings_url_is_none_for_a_registry_with_a_different_visibility_model() -> None:
    from npa.deploy.publish_public import package_settings_url

    assert package_settings_url("cr.eu-north1.nebius.cloud/abc/npa-lerobot:0.5.1") is None


def test_the_checklist_covers_exactly_the_packages_still_private() -> None:
    from npa.deploy import publish_public

    plan = build_publish_plan(target_registry="ghcr.io/example/workbench")
    failures = [(plan[0], "HTTP 403"), (plan[2], "HTTP 403")]

    checklist = publish_public.visibility_checklist(failures)

    assert checklist.count("- [ ] ") == 2
    assert publish_public.ghcr_owner_and_package(plan[1].target_ref)[1] not in checklist


def test_the_checklist_labels_a_package_the_way_its_settings_page_does() -> None:
    """The label must be the package name, not the whole reference, or the list does not
    match the page it links to."""
    from npa.deploy import publish_public
    from npa.deploy.publish_public import PublishItem

    item = PublishItem(
        tool="lerobot",
        source_ref="cr.eu-north1.nebius.cloud/abc/npa-lerobot:0.5.1",
        target_ref="ghcr.io/nebius/nebius-physical-ai/npa-lerobot:0.5.1",
    )

    assert publish_public.visibility_checklist([(item, "HTTP 403")]) == (
        "- [ ] [nebius-physical-ai/npa-lerobot]"
        "(https://github.com/orgs/nebius/packages/container/"
        "nebius-physical-ai%2Fnpa-lerobot/settings)"
    )
