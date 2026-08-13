"""The PAIDF deploy guide must build the tags submit actually pulls.

A first run against a fresh project has to build all three Cosmos images. The
guide's build/verify commands drifted from `npa/src/npa/deploy/images.py`
(`:0.1.0` and a `golden-eval-smoke` transfer tag vs the `:0.1.2` /
`skypilot-ready` pins), so following the guide to the letter still produced a
registry submit could not pull.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    build_and_push_command,
    supported_tool_version,
    tool_for_image_name,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_GUIDE = (
    REPO_ROOT / "docs" / "workbench" / "guides" / "physical-ai-data-factory-deploy.md"
)
PAIDF_TOOLS = ("cosmos2-transfer", "cosmos-evaluator", "cosmos-curate")


@pytest.mark.parametrize("tool", PAIDF_TOOLS)
def test_the_guide_never_names_a_tag_the_code_does_not_pin(tool: str) -> None:
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    image_name = CONTAINER_IMAGE_NAMES[tool]
    pinned = supported_tool_version(tool)

    mentioned = {
        line.split(f"{image_name}:", 1)[1].split()[0].strip("\"'`,;)")
        for line in guide.splitlines()
        if f"{image_name}:" in line
    }

    assert mentioned, f"the guide never mentions {image_name}"
    assert mentioned == {pinned}, (
        f"{image_name} tags in the deploy guide {sorted(mentioned)} do not match "
        f"images.py ({pinned})"
    )


def test_the_guide_does_not_claim_the_transfer_image_is_already_published() -> None:
    # A fresh project's registry returns NAME_UNKNOWN for all three.
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")

    assert "(already published)" not in guide


def test_the_guide_points_at_the_image_preflight() -> None:
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")

    assert "preflight-images" in guide


@pytest.mark.parametrize("tool", PAIDF_TOOLS)
def test_a_missing_image_yields_a_build_command_for_the_pinned_tag(tool: str) -> None:
    image_name = CONTAINER_IMAGE_NAMES[tool]
    reference = f"cr.eu-north1.nebius.cloud/e000/{image_name}:whatever-was-requested"

    command = build_and_push_command(reference)

    assert f"npa/docker/workbench/{tool}/Dockerfile" in command
    # The remedy names the tag the code pins, not the one that was missing.
    assert command.endswith(
        f"-t cr.eu-north1.nebius.cloud/e000/{image_name}:{supported_tool_version(tool)} npa"
    )


def test_a_third_party_image_gets_no_build_command() -> None:
    assert build_and_push_command("nvcr.io/nvidia/pytorch:24.01") == ""
    assert build_and_push_command("") == ""


def test_image_names_round_trip_to_tools() -> None:
    for tool, name in CONTAINER_IMAGE_NAMES.items():
        assert tool_for_image_name(name) == tool
    assert tool_for_image_name("not-an-npa-image") == ""


def test_the_quick_start_checks_images_before_submitting() -> None:
    """The copy-paste path went straight to submit and failed on missing images."""

    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    quick_start = guide.split("## 5. Submit", 1)[0]

    assert "preflight-images" in quick_start


def test_the_quick_start_forwards_the_hugging_face_token() -> None:
    # The curator fetches weights with it; the early block omitted it while a
    # later section included it, so a first submit silently lacked HF.
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    quick_start = guide.split("## 5. Submit", 1)[0]

    assert "--secret-env HF_TOKEN" in quick_start


def test_shell_examples_do_not_let_a_pipe_swallow_a_failed_submit() -> None:
    # `npa ... | tee run.log` reports 0 for a submit that printed `Error:`.
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")

    assert "set -o pipefail" in guide


def test_the_deploy_guide_path_names_the_image_step() -> None:
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    whole_path = (
        guide.split("## Quick start (copy-paste)", 1)[1]
        .split("```bash", 1)[1]
        .split("```", 1)[0]
    )

    assert "preflight-images" in whole_path
    # The copy-paste path must make its requested GPU count explicit.
    assert "--gpu-nodes" in whole_path


def test_no_build_command_when_the_dockerfile_is_not_where_we_would_say() -> None:
    """Not every tool builds from npa/docker/workbench/<tool>/Dockerfile.

    Printing a `-f` path that does not exist sends an operator to a command that
    cannot work; no command is better than a wrong one.
    """

    for tool in ("envgen", "reference-policy"):
        image = CONTAINER_IMAGE_NAMES[tool]
        assert not (
            REPO_ROOT / "npa" / "docker" / "workbench" / tool / "Dockerfile"
        ).is_file()
        assert build_and_push_command(f"cr.example/p/{image}:t") == ""


def test_a_missing_image_offers_a_server_side_copy_before_a_rebuild() -> None:
    # These images run to tens of GB; a 25 GB push through the local machine was
    # killed, while a registry-side retag succeeded.
    from npa.orchestration.skypilot.registry_preflight import (
        _missing_image_remedy,
        parse_image_reference,
    )

    remedy = _missing_image_remedy(
        parse_image_reference(
            "cr.us-central1.nebius.cloud/u00proj/npa-cosmos-curate:0.1.2"
        )
    )

    assert "crane copy" in remedy
    assert "server-side" in remedy
    # The copy is offered ahead of the local build.
    assert remedy.index("crane copy") < remedy.index("docker buildx build")


def test_the_copy_hint_never_suggests_copying_a_ref_onto_itself() -> None:
    from npa.deploy.images import primary_container_registry
    from npa.orchestration.skypilot.registry_preflight import (
        _missing_image_remedy,
        parse_image_reference,
    )

    same = f"{primary_container_registry()}/npa-cosmos-curate:0.1.2"
    remedy = _missing_image_remedy(parse_image_reference(same))

    assert f"crane copy {same} {same}" not in remedy


def test_the_deploy_guide_selects_the_public_mirror_before_preflight() -> None:
    """Building three multi-GB images is avoidable: the mirror already has them."""

    from npa.deploy.images import DEFAULT_PUBLIC_CONTAINER_REGISTRY

    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    whole_path = (
        guide.split("## Quick start (copy-paste)", 1)[1]
        .split("```bash", 1)[1]
        .split("```", 1)[0]
    )

    assert DEFAULT_PUBLIC_CONTAINER_REGISTRY in whole_path
    assert whole_path.index(DEFAULT_PUBLIC_CONTAINER_REGISTRY) < whole_path.index(
        "preflight-images"
    )
