from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_trivy_policy_uses_current_nested_configuration_schema() -> None:
    policy = yaml.safe_load((ROOT / "trivy.yaml").read_text(encoding="utf-8"))

    assert policy["vulnerability"]["ignore-unfixed"] is True
    assert policy["pkg"]["types"] == ["os"]
    assert policy["severity"] == ["CRITICAL"]
    assert "ignore-unfixed" not in policy
    assert "pkg-types" not in policy
