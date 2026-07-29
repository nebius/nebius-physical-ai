"""Guardrail: the documented path from README to a real submit stays runnable.

A README-path walkthrough dead-ended twice on documentation rather than code:
the Physical AI Data Factory "Quick start (copy-paste)" omitted the npa-source
staging, the SkyPilot bootstrap and the cluster, so the very first submit failed;
and the install instructions disagreed about the virtualenv path (`.venv` in the
README, `npa/.venv` in the deploy runbook), which silently ran `npa` from the
wrong interpreter.

These checks pin the copy-paste blocks so they cannot drift back.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"
INSTALL = REPO_ROOT / "docs" / "install.md"
QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
GUIDES_README = REPO_ROOT / "docs" / "workbench" / "guides" / "README.md"
PAIDF_GUIDE = (
    REPO_ROOT / "docs" / "workbench" / "guides" / "physical-ai-data-factory.md"
)
PAIDF_DEPLOY = (
    REPO_ROOT / "docs" / "workbench" / "guides" / "physical-ai-data-factory-deploy.md"
)

#: Docs a first-time user follows to install. They must agree on the venv path.
USER_FACING_INSTALL_DOCS = (README, INSTALL, QUICKSTART, GUIDES_README, PAIDF_DEPLOY)

_FENCE_RE = re.compile(r"```(?:bash|sh|shell|console)\n(.*?)```", re.DOTALL)


def _shell_blocks(path: Path) -> list[str]:
    return _FENCE_RE.findall(path.read_text(encoding="utf-8"))


def test_user_facing_docs_agree_on_the_venv_path() -> None:
    """No user-facing install block may create or activate `npa/.venv`."""
    offenders: list[str] = []
    for path in USER_FACING_INSTALL_DOCS:
        for block in _shell_blocks(path):
            for line in block.splitlines():
                stripped = line.strip()
                if "npa/.venv" in stripped and (
                    stripped.startswith(("python3 -m venv", "source", "uv venv"))
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped}")
    assert not offenders, (
        "User-facing install instructions must use repo-root `.venv` (contributor "
        "tooling uses npa/.venv; see docs/install.md). Offending lines:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", [README, INSTALL, QUICKSTART])
def test_install_docs_create_the_repo_root_venv(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "python3 -m venv .venv" in text, f"{path.relative_to(REPO_ROOT)}"


def test_install_docs_explain_the_contributor_venv_convention() -> None:
    """The split is fine, but it has to be stated somewhere users will look."""
    text = INSTALL.read_text(encoding="utf-8")
    assert "npa/.venv" in text
    assert ".venv" in text


def _submit_blocks(path: Path) -> list[str]:
    return [
        block
        for block in _shell_blocks(path)
        if "npa workbench workflow submit" in block
    ]


@pytest.mark.parametrize("path", [PAIDF_GUIDE, PAIDF_DEPLOY])
def test_documented_submits_can_install_npa_on_the_worker(path: Path) -> None:
    """Every copy-paste submit must supply npa to image-less steps.

    Without `--stage-src`, `NPA_SRC_S3_URI` or `--image`, the render aborts with
    "planned step ... has no workbench image and NPA_SRC_S3_URI is unset".
    """
    blocks = _submit_blocks(path)
    assert blocks, f"no submit example found in {path.relative_to(REPO_ROOT)}"
    for block in blocks:
        assert any(
            marker in block
            for marker in ("--stage-src", "NPA_SRC_S3_URI", "--image", "--plan-only")
        ), (
            f"{path.relative_to(REPO_ROOT)} has a `workflow submit` example with no "
            "npa source for image-less steps:\n" + block
        )


@pytest.mark.parametrize("path", [PAIDF_GUIDE, PAIDF_DEPLOY])
def test_documented_submits_pass_a_real_bucket(path: Path) -> None:
    for block in _submit_blocks(path):
        assert "--var bucket=" in block, (
            f"{path.relative_to(REPO_ROOT)} has a `workflow submit` example without "
            "--var bucket=, so it would run against the spec's example-bucket "
            "placeholder:\n" + block
        )


def test_paidf_quickstart_is_a_complete_ordered_path() -> None:
    """The quickstart must not skip the steps a fresh account needs."""
    text = PAIDF_DEPLOY.read_text(encoding="utf-8")
    quickstart = text.split("## Quick start (copy-paste)", 1)[1].split("\n## ", 1)[0]
    for required in (
        "npa skypilot bootstrap",
        "npa provision-if-absent",
        "stage-src",
        "--infra k8s/",
        "--secret-env NEBIUS_TOKEN_FACTORY_KEY",
    ):
        assert required in quickstart, (
            f"the Physical AI Data Factory quickstart no longer mentions {required!r}; "
            "a first submit will fail on it"
        )


def test_paidf_quickstart_documents_failure_recovery() -> None:
    text = PAIDF_DEPLOY.read_text(encoding="utf-8")
    assert "If submit fails" in text
    for symptom in ("NPA_SRC_S3_URI is unset", "sky-jobs-controller", "example-bucket"):
        assert symptom in text, f"missing recovery guidance for {symptom!r}"


def test_readme_documents_the_ordered_green_path() -> None:
    text = README.read_text(encoding="utf-8")
    ordered = [
        "npa configure",
        "npa workbench health preflight",
        "npa provision-if-absent",
        "npa skypilot bootstrap",
        "npa workbench workflow stage-src",
        "npa workbench workflow submit",
    ]
    positions = []
    for command in ordered:
        index = text.find(command)
        assert index != -1, f"README no longer mentions `{command}`"
        positions.append(index)
    assert positions == sorted(positions), (
        "the README green-path commands are no longer in runnable order: " + str(ordered)
    )


def test_paidf_credentials_sample_uses_the_canonical_schema() -> None:
    """`npa configure` writes `tokens:`; the guide must not teach a dead shape."""
    text = PAIDF_DEPLOY.read_text(encoding="utf-8")
    assert "NEBIUS_TOKEN_FACTORY_KEY: <your-token-factory-key>" in text
    assert "HF_TOKEN: <your-hf-token>" in text
