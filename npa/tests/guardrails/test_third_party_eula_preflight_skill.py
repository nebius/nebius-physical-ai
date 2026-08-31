"""Discovery and consent guardrails for the reusable third-party EULA preflight."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.workbench.model_access import HF, WORKBENCH_ASSETS, usable_hf_payload_probe


REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL = REPO_ROOT / "skills/atomic/third-party-eula-preflight/SKILL.md"
INDEX = REPO_ROOT / "skills/index.yaml"
LINKED_SKILLS = (
    REPO_ROOT / "skills/workflows/sim2real-operate/SKILL.md",
    REPO_ROOT / "skills/tools/isaac-lab/SKILL.md",
    REPO_ROOT / "skills/tools/sonic/SKILL.md",
    REPO_ROOT / "skills/tools/groot/SKILL.md",
    REPO_ROOT / "skills/atomic/solution-licensing/SKILL.md",
)
EULA_VALUE = "ACCEPT_EULA=Y"


def test_eula_preflight_is_canonical_and_discoverable() -> None:
    index = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    entry = next(
        item for item in index["skills"] if item["name"] == "third-party-eula-preflight"
    )

    assert REPO_ROOT / entry["path"] == SKILL
    assert entry["category"] == "atomic"
    assert "before provisioning" in entry["when_to_use"].lower()


def test_eula_preflight_documents_scoped_default_and_explicit_opt_out() -> None:
    text = SKILL.read_text(encoding="utf-8")
    normalized = " ".join(text.split()).lower()
    required = (
        "official terms links",
        "defaults vendor acceptance on",
        "explicit opt-out",
        "fail before provisioning",
        "Do not reuse this",
        "Never default optional",
        "PRIVACY_CONSENT",
        "npa-isaac-lab",
        "Isaac-backed SONIC modes",
        "GR00T Isaac simulation",
        "absent variable succeeds",
        "--no-accept-eula",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "do not add duplicate",
        "telemetry off by default",
        "built layers contain no proprietary Isaac or Kit bytes",
        "Do not store",
        "secret values or unnecessary personal data",
        "token and its actual upstream permissions are the only local gate",
        "probe every required repository before provisioning",
        "do not provide `--skip-model-check`",
        "do not add an NPA EULA/terms boolean",
    )
    for phrase in required:
        assert phrase.lower() in normalized, phrase
    assert EULA_VALUE in text
    assert "${ACCEPT_EULA-Y}" in text
    assert "${ACCEPT_EULA:-Y}" in text


def test_operational_skills_link_the_preflight() -> None:
    for path in LINKED_SKILLS:
        assert "skills/atomic/third-party-eula-preflight/SKILL.md" in path.read_text(
            encoding="utf-8"
        ), path


def test_isaac_tool_skills_preserve_default_opt_out_and_internal_plumbing() -> None:
    for tool in ("isaac-lab", "sonic", "groot"):
        path = REPO_ROOT / f"skills/tools/{tool}/SKILL.md"
        text = " ".join(path.read_text(encoding="utf-8").split())
        for phrase in (
            "defaults `ACCEPT_EULA=Y`",
            "`N`, `NO`, `0`, `FALSE`",
            "`Y`, `YES`, `1`, and `TRUE`",
            "other values are invalid",
            "before download",
            "derives `OMNI_KIT_ACCEPT_EULA=YES` internally",
            "Keep `PRIVACY_CONSENT` and telemetry off",
        ):
            assert phrase in text, f"{path}: {phrase}"


def test_retired_manual_gate_surfaces_do_not_return() -> None:
    roots = (
        REPO_ROOT / "npa/src",
        REPO_ROOT / "npa/scripts",
        REPO_ROOT / "npa/workflows",
    )
    retired = (
        "--accept-nvidia-eula",
        "--skip-model-check",
        "omni_kit_accept_eula",
        "isaacsim_accept_eula",
    )
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in retired:
                assert marker not in text, f"retired manual gate {marker!r} in {path}"


def test_openpi_product_policy_keeps_its_scoped_runtime_gate() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "## OpenPI pi0.5 Reference" in text
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES" in text
    assert "before any accepted checkpoint" in text
    assert "Forward it only as a runtime secret" in text

    workflow = REPO_ROOT / "npa/workflows/workbench/npa-workflows/byof-openpi.yaml"
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" in workflow.read_text(encoding="utf-8")


def test_every_gated_hf_catalog_asset_has_a_pinned_payload_byte_probe() -> None:
    gated_hf = [
        asset for asset in WORKBENCH_ASSETS if asset.provider == HF and asset.gated
    ]

    assert gated_hf
    for asset in gated_hf:
        assert usable_hf_payload_probe(asset), (
            f"{asset.repo} must pin a revision and a payload probe_path; README, "
            "model-card, license, tokenizer, and config files are not entitlement proof"
        )
