"""Profile authentication must survive a long native Terraform operation."""

from npa.cluster_backends import mk8s_execution as E


def test_explicit_profile_uses_refreshable_auth_for_supported_recipe(tmp_path, monkeypatch):
    (tmp_path / "variables.tf").write_text(
        'variable "nebius_profile" {}\nvariable "nebius_cli" {}\n'
    )
    seen = []
    monkeypatch.setattr(E, "_terraform_env", lambda binary, **kwargs: (
        seen.append((binary, kwargs)),
        {"NEBIUS_IAM_TOKEN": "stale", "NPA_NEBIUS_IAM_TOKEN": "stale", "TF_VAR_iam_token": "fresh"},
    )[1])
    env = E._cluster_tf_env(
        "/tools/nebius", tenant_id="tenant-test", project_id="project-test",
        region="uk-south2", subnet_id="subnet-test", profile="selected",
        recipe_dir=tmp_path,
    )
    assert seen == [("/tools/nebius", {"profile": "selected"})]
    assert "NEBIUS_IAM_TOKEN" not in env and "NPA_NEBIUS_IAM_TOKEN" not in env
    assert env["TF_VAR_nebius_profile"] == "selected"
    assert env["TF_VAR_nebius_cli"] == "/tools/nebius"
    assert env["TF_VAR_iam_token"] == "fresh"
    assert env["TF_VAR_parent_id"] == "project-test"


def test_legacy_recipe_retains_explicit_minted_token(tmp_path, monkeypatch):
    (tmp_path / "variables.tf").write_text('variable "iam_token" {}\n')
    monkeypatch.setattr(E, "_terraform_env", lambda *args, **kwargs: {
        "NEBIUS_IAM_TOKEN": "fresh", "TF_VAR_iam_token": "fresh",
    })
    env = E._cluster_tf_env(
        "nebius", tenant_id="tenant-test", project_id="project-test",
        region="uk-south2", subnet_id="subnet-test", profile="selected",
        recipe_dir=tmp_path,
    )
    assert env["NEBIUS_IAM_TOKEN"] == "fresh"
    assert "TF_VAR_nebius_profile" not in env
