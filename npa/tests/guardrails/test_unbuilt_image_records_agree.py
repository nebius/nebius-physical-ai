"""Four separate files claim whether an image has been built. They must agree.

"This image has not been built yet" is recorded in four places, each of which a
different guard reads:

* ``images.UNVALIDATED_PUBLICATION_TOOLS`` — what ``publish_public`` refuses;
* ``SUPPORTED_TOOL_VERSIONS`` — a tag ending ``-unbuilt``, so a tag that has
  never been produced cannot be mistaken for one that has;
* ``blackwell-dc-images.json`` — ``validation: pending-build``;
* ``golden_evals.yaml`` — a golden eval that is not ``ready``.

Build day removes one of them. The other three then go stale silently, and every
one of them is a claim about what has been *proven* — the exact class of claim
this repository is careful about everywhere else. So the equivalence is asserted
rather than left to whoever does the build remembering all four.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import yaml

from npa.deploy.images import (
    CONTAINER_IMAGE_NAMES,
    PUBLICATION_QUARANTINE_TOOLS,
    STALE_PUBLICATION_TOOLS,
    SUPPORTED_TOOL_VERSIONS,
    UNVALIDATED_PUBLICATION_TOOLS,
    VALIDATION_CANDIDATE_TOOLS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BLACKWELL = REPO_ROOT / "npa" / "docker" / "workbench" / "blackwell-dc-images.json"
GOLDEN_EVALS = REPO_ROOT / "npa" / "src" / "npa" / "smoke" / "golden_evals.yaml"

UNBUILT_TAG_SUFFIX = "-unbuilt"
PENDING_BUILD = "pending-build"
#: Built, bytes checked, but never run on a GPU. Still unvalidated for
#: publication - the byte evidence is only half of what publication claims.
PENDING_GPU = "pending-gpu"
UNPROVEN_STATES = frozenset({PENDING_BUILD, PENDING_GPU})


def _blackwell_images() -> dict[str, dict[str, object]]:
    payload = json.loads(BLACKWELL.read_text(encoding="utf-8"))
    return {str(image["name"]): image for image in payload["images"]}


def _golden_eval_containers() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(GOLDEN_EVALS.read_text(encoding="utf-8"))
    return payload["containers"]


def _image_name(tool: str) -> str:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES

    return str(CONTAINER_IMAGE_NAMES.get(tool, f"npa-{tool}"))


def test_every_unbuilt_tool_says_so_in_all_four_records() -> None:
    blackwell = _blackwell_images()
    containers = _golden_eval_containers()

    for tool in sorted(UNVALIDATED_PUBLICATION_TOOLS):
        version = str(SUPPORTED_TOOL_VERSIONS.get(tool, ""))
        assert version.endswith(UNBUILT_TAG_SUFFIX), (
            f"{tool} is unvalidated for publication but its tag {version!r} does "
            f"not end in {UNBUILT_TAG_SUFFIX}; a tag that reads as a release is "
            "how an unbuilt image gets pulled by mistake"
        )

        entry = blackwell.get(_image_name(tool))
        if entry is not None and entry.get("verdict") == "not-applicable":
            # CPU ingestion images can be publication-unvalidated without a GPU
            # architecture assertion. Do not invent a pending GPU build for them.
            assert entry.get("validation") == "not-required", tool
            assert containers[tool]["golden_eval"]["gpu"] == "none", tool
        elif entry is not None:
            assert entry.get("validation") in UNPROVEN_STATES, (
                f"{tool} is unvalidated for publication but "
                f"blackwell-dc-images.json records "
                f"validation={entry.get('validation')!r}; publication needs both "
                "byte evidence and a GPU result"
            )

        container = containers.get(tool)
        if container is not None:
            status = (container.get("golden_eval") or {}).get("status")
            assert status != "ready", (
                f"{tool} is unvalidated but its golden eval claims status={status!r}; "
                "a ready eval asserts there is an image to run it against"
            )


def test_no_built_tool_is_left_carrying_an_unbuilt_tag() -> None:
    """The other direction: build day must not leave the tag behind."""

    stale = sorted(
        tool
        for tool, version in SUPPORTED_TOOL_VERSIONS.items()
        if str(version).endswith(UNBUILT_TAG_SUFFIX)
        and tool not in UNVALIDATED_PUBLICATION_TOOLS
    )

    assert stale == [], (
        f"{stale} still carry an {UNBUILT_TAG_SUFFIX} tag but are no longer listed "
        "as unvalidated for publication"
    )


def test_fixed_tag_candidates_remain_in_the_publication_quarantine() -> None:
    assert PUBLICATION_QUARANTINE_TOOLS == (
        UNVALIDATED_PUBLICATION_TOOLS
        | VALIDATION_CANDIDATE_TOOLS
        | STALE_PUBLICATION_TOOLS
    )
    for tool in VALIDATION_CANDIDATE_TOOLS:
        version = str(SUPPORTED_TOOL_VERSIONS[tool])
        assert version and not version.endswith(UNBUILT_TAG_SUFFIX), tool
        build = REPO_ROOT / "npa" / "docker" / "workbench" / tool / "build.sh"
        assert build.is_file(), tool
        assert version in build.read_text(encoding="utf-8"), tool


def test_stale_publications_are_built_releases_awaiting_requalification() -> None:
    """Stale bytes are not confused with images that have never been built."""

    assert STALE_PUBLICATION_TOOLS == frozenset(
        {
            "cosmos-curate",
            "cosmos-evaluator",
            "cosmos3",
            "cosmos3-ray-serve",
            "isaac-lab",
            "isaac-arena",
        }
    )
    assert STALE_PUBLICATION_TOOLS.isdisjoint(UNVALIDATED_PUBLICATION_TOOLS)
    assert STALE_PUBLICATION_TOOLS.isdisjoint(VALIDATION_CANDIDATE_TOOLS)
    for tool in STALE_PUBLICATION_TOOLS:
        assert not SUPPORTED_TOOL_VERSIONS[tool].endswith(UNBUILT_TAG_SUFFIX), tool


def test_stale_publication_quarantine_propagates_to_derived_images() -> None:
    """A child cannot be accepted while retaining every layer of a stale parent."""

    image_to_tool = {
        image: tool for tool, image in CONTAINER_IMAGE_NAMES.items()
    }
    parent_pattern = re.compile(r"\bnpa-[a-z0-9-]+(?=[:@])")
    for dockerfile in sorted((REPO_ROOT / "npa/docker/workbench").glob("*/Dockerfile")):
        child = dockerfile.parent.name
        if child not in CONTAINER_IMAGE_NAMES:
            continue
        parents = {
            image_to_tool[image]
            for image in parent_pattern.findall(
                dockerfile.read_text(encoding="utf-8")
            )
            if image in image_to_tool
        }
        stale_parents = parents & STALE_PUBLICATION_TOOLS
        if stale_parents:
            assert child in STALE_PUBLICATION_TOOLS, (
                f"{child} inherits stale image layer(s) from "
                f"{sorted(stale_parents)} but remains publication-eligible"
            )


def test_pending_build_never_carries_a_confident_verdict() -> None:
    """A verdict is a statement about an artifact; pending-build has none.

    ``pending-build`` says in its own definition: "never upgrade this from a
    reading of the Dockerfile". A ``ready`` verdict beside it does exactly that,
    which is how ``npa-ltx2`` came to claim both at once. ``pending-gpu`` is the
    same: bytes checked is not an architecture result.
    """

    payload = json.loads(BLACKWELL.read_text(encoding="utf-8"))
    confident = {"ready", "port"}

    offenders = sorted(
        str(image["name"])
        for image in payload["images"]
        if image.get("validation") in UNPROVEN_STATES
        and str(image.get("verdict")) in confident
    )

    assert offenders == [], (
        f"{offenders} record validation={PENDING_BUILD!r} beside a confident "
        "verdict. Nothing has been built, so the verdict can only have come from "
        "reading the Dockerfile."
    )


def test_every_verdict_and_validation_state_is_one_of_the_declared_values() -> None:
    payload = json.loads(BLACKWELL.read_text(encoding="utf-8"))
    verdicts = set(payload["verdicts"])
    states = set(payload["validation_states"])

    for image in payload["images"]:
        assert image.get("verdict") in verdicts, image["name"]
        assert image.get("validation") in states, image["name"]
