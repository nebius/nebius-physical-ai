"""Keep Gitleaks exceptions narrow while scanner primitives are bootstrapped."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[3] / ".gitleaks.toml"
WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"


def _config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_test_fixtures_do_not_disable_default_secret_rules() -> None:
    """Scope test-tree paths to custom infrastructure rules only."""
    config = _config()
    global_paths = [
        pattern
        for allowlist in config.get("allowlists", [])
        for pattern in allowlist.get("paths", [])
    ]
    assert config["extend"]["useDefault"] is True
    assert global_paths == []

    secret_probes = (
        "api_" + 'key = "NpaGeneratedControl9Qx7Lm4Zr8Vt2Ks6"',
        "NpaGeneratedControl9Qx7Lm4Zr8Vt2Ks6",
    )
    assert all(
        not any(re.search(expression, probe) for probe in secret_probes)
        for allowlist in config.get("allowlists", [])
        for expression in allowlist.get("regexes", [])
    )

    rules = {rule["id"]: rule for rule in config["rules"]}
    for rule_id in {
        "lerobot-bucket-pattern",
        "nebius-dated-operational-name",
        "nebius-generic-id",
        "nebius-prefixed-id",
        "nebius-private-object-location",
        "nebius-private-registry-location",
    }:
        scoped_paths = [
            pattern
            for allowlist in rules[rule_id].get("allowlists", [])
            for pattern in allowlist.get("paths", [])
        ]
        assert any(
            re.search(pattern, "npa/tests/test_secret_probe.py")
            for pattern in scoped_paths
        ), rule_id


def test_default_rule_exceptions_are_exact_lines() -> None:
    """Bind every default-rule exception to one exact anchored fixture line."""
    rules = {rule["id"]: rule for rule in _config()["rules"]}
    allowlists = rules["generic-api-key"]["allowlists"]
    assert len(allowlists) == 11
    assert "private-key" not in rules
    expressions = [
        expression for allowlist in allowlists for expression in allowlist["regexes"]
    ]
    assert "sk-[a-z0-9]+" not in "\n".join(expressions)

    key_name = "NEBIUS_TOKEN_FACTORY_" + "KEY"
    unrelated = (
        "api_" + 'key = "NpaGenerated9Qx7Lm4Zr8Vt2Ks6AaBbCcDdEeFf"',
        f'leak_sample = "{key_name}=sk-fedcba9876543210fedcba"',
    )
    assert all(
        not any(re.search(expression, probe) for probe in unrelated)
        for expression in expressions
    )
    for allowlist in allowlists:
        assert allowlist["condition"] == "AND"
        assert allowlist["regexTarget"] == "line"
        assert "paths" not in allowlist
        assert allowlist["regexes"]
        assert all(
            expression.startswith(("^", "(?m)^")) and expression.endswith("$")
            for expression in allowlist["regexes"]
        )


def test_global_allowlist_has_no_path_exceptions() -> None:
    """Prevent any global exception from suppressing a complete source path."""
    assert all("paths" not in item for item in _config()["allowlists"])


def test_bootstrap_does_not_execute_candidate_scanner_helpers() -> None:
    """Keep helper activation in a later PR whose trusted base contains them."""
    for name in (
        "gitleaks.yml",
        "security-regression.yml",
        "publish-public-images.yml",
    ):
        workflow = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "run_gitleaks_tree_scan.py" not in workflow
        assert "materialize_gitleaks_tree.py" not in workflow
