"""Keep supported-candidate quarantine distinct from qualified development bytes.

"This image has not been built yet" is recorded in four places, each of which a
different guard reads:

* ``images.UNVALIDATED_PUBLICATION_TOOLS`` — what ``publish_public`` refuses;
* ``SUPPORTED_TOOL_VERSIONS`` — a tag ending ``-unbuilt``, so a tag that has
  never been produced cannot be mistaken for one that has;
* ``blackwell-dc-images.json`` — ``validation: pending-build``;
* ``golden_evals.yaml`` — a golden eval that is not ``ready``.

An immutable public development image can pass a narrower real workload before
the supported candidate completes its larger qualification. That evidence must
identify the actual artifact and scope; it cannot silently promote the default
tag, make its golden evaluation ready, or enter the supported public selection.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from npa.deploy.images import (
    PUBLICATION_QUARANTINE_TOOLS,
    SUPPORTED_TOOL_VERSIONS,
    UNVALIDATED_PUBLICATION_TOOLS,
    VALIDATION_CANDIDATE_TOOLS,
    development_image_for_tool,
    publicly_publishable_tools,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BLACKWELL = REPO_ROOT / "npa" / "docker" / "workbench" / "blackwell-dc-images.json"
GOLDEN_EVALS = REPO_ROOT / "npa" / "src" / "npa" / "smoke" / "golden_evals.yaml"

UNBUILT_TAG_SUFFIX = "-unbuilt"
PENDING_BUILD = "pending-build"
#: Built and bytes checked, but without a real GPU capability result.
#: Development publication alone does not qualify the supported candidate.
PENDING_GPU = "pending-gpu"
UNPROVEN_STATES = frozenset({PENDING_BUILD, PENDING_GPU})
PENDING_QUALIFICATION = "pending-qualification"


def _assert_runtime_qualification(runtime: dict[str, object]) -> None:
    assert runtime["completed"] is True
    assert runtime["successful"] is True
    for key in ("scope", "gpu"):
        assert isinstance(runtime[key], str) and runtime[key].strip()
    capability = runtime["compute_capability"]
    assert isinstance(capability, str) and re.fullmatch(r"[1-9][0-9]*\.[0-9]+", capability)
    assert re.fullmatch(r"[0-9a-f]{64}", str(runtime["evidence_sha256"]))
    artifacts = runtime["artifacts"]
    assert isinstance(artifacts, list) and artifacts
    for artifact in artifacts:
        assert isinstance(artifact["name"], str) and artifact["name"].strip()
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        assert type(artifact["size_bytes"]) is int and artifact["size_bytes"] > 0


def _assert_development_qualification(tool: str, entry: dict[str, object]) -> None:
    qualification = entry["development_qualification"]
    assert qualification["channel"] == "development"
    source = qualification["source_commit_sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", source)
    assert qualification["reference"] == development_image_for_tool(tool, git_sha=source)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", qualification["digest"])
    gates = qualification["prepublication_validation"]
    assert gates["passed"] is True and gates["anonymous_verified"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", gates["evidence_sha256"])
    assert re.fullmatch(
        r"https://github.com/nebius/nebius-physical-ai/actions/runs/[1-9][0-9]*",
        gates["workflow_url"],
    )
    runtime = qualification["runtime"]
    assert runtime["digest"] == qualification["digest"]
    _assert_runtime_qualification(runtime)
    assert qualification["supported_qualification"] == "pending"
    remaining = qualification["remaining_qualification"]
    assert isinstance(remaining, list) and remaining
    assert all(isinstance(scope, str) and scope.strip() for scope in remaining)
    assert tool not in publicly_publishable_tools()


def _blackwell_images() -> dict[str, dict[str, object]]:
    payload = json.loads(BLACKWELL.read_text(encoding="utf-8"))
    return {str(image["name"]): image for image in payload["images"]}


def _golden_eval_containers() -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(GOLDEN_EVALS.read_text(encoding="utf-8"))
    return payload["containers"]


def _image_name(tool: str) -> str:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES

    return str(CONTAINER_IMAGE_NAMES.get(tool, f"npa-{tool}"))


def test_every_unvalidated_supported_candidate_keeps_quarantine_records() -> None:
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
        if entry is not None:
            state = entry.get("validation")
            if state == PENDING_QUALIFICATION:
                _assert_development_qualification(tool, entry)
            assert state in UNPROVEN_STATES | {PENDING_QUALIFICATION}, (
                f"{tool} is unvalidated for publication but "
                f"blackwell-dc-images.json records "
                f"validation={entry.get('validation')!r}; publication needs both "
                "artifact evidence and the full supported qualification"
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
        UNVALIDATED_PUBLICATION_TOOLS | VALIDATION_CANDIDATE_TOOLS
    )
    for tool in VALIDATION_CANDIDATE_TOOLS:
        version = str(SUPPORTED_TOOL_VERSIONS[tool])
        assert version and not version.endswith(UNBUILT_TAG_SUFFIX), tool
        # Fixed historical pins may remain while replacement bytes use the
        # trusted full-SHA development builder. No build script should have to
        # retag unqualified bytes with that historical supported-version string.
        image = development_image_for_tool(tool, git_sha="b" * 40)
        assert image == f"ghcr.io/nebius/nebius-physical-ai/npa-{tool}:dev-{'b' * 40}"
        assert tool not in publicly_publishable_tools()
        dockerfile = REPO_ROOT / "npa" / "docker" / "workbench" / tool / "Dockerfile"
        assert dockerfile.is_file(), tool


def test_development_qualification_never_substitutes_for_supported_release() -> None:
    for name, entry in _blackwell_images().items():
        if entry["validation"] == PENDING_QUALIFICATION:
            _assert_development_qualification(name.removeprefix("npa-"), entry)


@pytest.fixture
def development_qualification() -> dict[str, object]:
    """Synthetic metadata exercises the guard; it is not runtime evidence."""
    source, digest = "a" * 40, "sha256:" + "b" * 64
    return {"development_qualification": {
        "channel": "development", "source_commit_sha": source,
        "reference": development_image_for_tool("curobo", git_sha=source),
        "digest": digest, "supported_qualification": "pending",
        "remaining_qualification": ["Complete multi-dataset benchmark"],
        "prepublication_validation": {
            "passed": True, "anonymous_verified": True, "evidence_sha256": "c" * 64,
            "workflow_url": "https://github.com/nebius/nebius-physical-ai/actions/runs/1",
        },
        "runtime": {
            "digest": digest, "completed": True, "successful": True,
            "scope": "Three synthetic Franka poses", "gpu": "NVIDIA B200",
            "compute_capability": "10.0", "evidence_sha256": "d" * 64,
            "artifacts": [{"name": "plans.json", "sha256": "e" * 64, "size_bytes": 1}],
        },
    }}


@pytest.mark.parametrize("path,value", [
    (("source_commit_sha",), "f" * 40),
    (("source_commit_sha",), "a" * 7),
    (("reference",), "ghcr.io/nebius/nebius-physical-ai/npa-openpi:dev-" + "a" * 40),
    (("reference",), "ghcr.io/nebius/nebius-physical-ai/npa-curobo:latest"),
    (("digest",), "sha256:short"),
    (("prepublication_validation", "passed"), False),
    (("prepublication_validation", "anonymous_verified"), False),
    (("prepublication_validation", "evidence_sha256"), ""),
    (("runtime", "digest"), "sha256:" + "f" * 64),
    (("runtime", "completed"), False),
    (("runtime", "successful"), False),
    (("runtime", "scope"), ""),
    (("runtime", "scope"), True),
    (("runtime", "scope"), "  "),
    (("runtime", "gpu"), ""),
    (("runtime", "gpu"), True),
    (("runtime", "gpu"), "  "),
    (("runtime", "compute_capability"), ""),
    (("runtime", "compute_capability"), True),
    (("runtime", "compute_capability"), "Blackwell"),
    (("runtime", "evidence_sha256"), ""),
    (("runtime", "artifacts"), []),
    (("runtime", "artifacts"), [{"name": "plans.json", "sha256": "", "size_bytes": 1}]),
    (("runtime", "artifacts"), [{"name": "plans.json", "sha256": "e" * 64, "size_bytes": True}]),
    (("runtime", "artifacts"), [{"name": "  ", "sha256": "e" * 64, "size_bytes": 1}]),
    (("supported_qualification",), "complete"),
    (("remaining_qualification",), []),
    (("remaining_qualification",), True),
    (("remaining_qualification",), [True]),
    (("remaining_qualification",), ["  "]),
])
def test_incomplete_development_evidence_cannot_relax_quarantine(
    development_qualification: dict[str, object], path: tuple[str, ...], value: object,
) -> None:
    _assert_development_qualification("curobo", development_qualification)
    broken = deepcopy(development_qualification)
    target = broken["development_qualification"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        _assert_development_qualification("curobo", broken)


def test_absent_runtime_cannot_relax_quarantine(development_qualification: dict) -> None:
    del development_qualification["development_qualification"]["runtime"]
    with pytest.raises(KeyError):
        _assert_development_qualification("curobo", development_qualification)


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
