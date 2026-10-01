"""Missing SDK deadlines must not shorten the existing apply operation budget."""

import json
import shutil
import subprocess

import hcl2
import pytest
from hcl2.utils import SerializationOptions

from npa.cluster_backends import provider_rpc as rpc


def decoded(path):
    return hcl2.loads(
        path.read_text(),
        serialization_options=SerializationOptions(
            with_comments=False, strip_string_quotes=True, explicit_blocks=False
        ),
    )


def test_missing_deadlines_preserve_hcl_comments_and_nested_braces(tmp_path):
    original = """# provider "nebius" { timeout = "1s" }
locals {
  example = <<-EOT
provider "nebius" { }
EOT
}
provider "nebius" {
  endpoints = { mk8s = "https://example.invalid/{path}" }
  service_account { private_key_file = var.key }
}
provider "nebius" { alias = "other" }
"""
    path = tmp_path / "provider.tf"
    path.write_text(original)
    before = decoded(path)
    receipt = rpc.configure_provider_rpc_deadlines(tmp_path, 120)
    after = decoded(path)
    defaults = {key: "120m" for key in ("timeout", "per_retry_timeout", "auth_timeout")}
    assert receipt["inserted_defaults"] == defaults
    for key, value in defaults.items():
        assert after["provider"][0]["nebius"].pop(key) == value
    assert before == after
    assert '# provider "nebius" { timeout = "1s" }' in path.read_text()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["provider.tf"]


@pytest.mark.parametrize("expression", ['"3m"', "var.operator_timeout", "null"])
def test_explicit_operator_deadlines_and_retries_untouched(tmp_path, expression):
    path = tmp_path / "provider.tf"
    original = f"""provider "nebius" {{
  timeout = {expression}
  per_retry_timeout = {expression}
  auth_timeout = {expression}
  retries = 7
}}
"""
    path.write_text(original)
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 17)["inserted_defaults"] == {}
    assert path.read_text() == original


def test_hcl_override_and_json_override_settings_are_respected(tmp_path):
    path = tmp_path / "provider.tf"
    path.write_text('provider "nebius" { retries = 9 }\n')
    override = tmp_path / "operator_override.tf"
    override.write_text('provider "nebius" { timeout = var.deadline }\n')
    json_override = tmp_path / "z_override.tf.json"
    original_json = '{"provider":{"nebius":{"auth_timeout":null}}}'
    json_override.write_text(original_json)
    result = rpc.configure_provider_rpc_deadlines(tmp_path, 31)
    assert result["inserted_defaults"] == {"per_retry_timeout": "31m"}
    assert result["preserved_operator_fields"] == ["auth_timeout", "timeout"]
    assert override.read_text() == 'provider "nebius" { timeout = var.deadline }\n'
    assert json_override.read_text() == original_json
    assert decoded(path)["provider"][0]["nebius"]["retries"] == 9


def test_json_provider_and_alias_preserve_semantics(tmp_path):
    path = tmp_path / "provider.tf.json"
    configuration = {
        "provider": {
            "nebius": [
                {"timeout": "${var.deadline}"},
                {"alias": "other", "timeout": "1s"},
            ]
        },
        "locals": {"value": "brace }"},
    }
    path.write_text(json.dumps(configuration))
    rpc.configure_provider_rpc_deadlines(tmp_path, 41)
    after = json.loads(path.read_text())
    assert after["provider"]["nebius"][0].pop("per_retry_timeout") == "41m"
    assert after["provider"]["nebius"][0].pop("auth_timeout") == "41m"
    assert after == configuration


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "120"])
def test_invalid_budget_is_rejected_before_input_changes(tmp_path, budget):
    path = tmp_path / "provider.tf"
    path.write_text('provider "nebius" {}')
    with pytest.raises(ValueError, match="positive"):
        rpc.configure_provider_rpc_deadlines(tmp_path, budget)
    assert path.read_text() == 'provider "nebius" {}'


@pytest.mark.parametrize(
    "source",
    [
        'provider "nebius" {',
        'provider "other" {}',
        'provider "nebius" {}\nprovider "nebius" {}',
    ],
)
def test_ambiguous_or_invalid_configuration_fails_without_writes(tmp_path, source):
    path = tmp_path / "provider.tf"
    path.write_text(source)
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 120)["status"] == "advisory"
    assert path.read_text() == source


def test_symlinked_source_is_never_read_or_modified(tmp_path):
    target = tmp_path / "outside"
    target.write_text('provider "nebius" {}')
    (tmp_path / "provider.tf").symlink_to(target)
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 120)["status"] == "advisory"
    assert target.read_text() == 'provider "nebius" {}'


def test_source_change_during_inspection_does_not_get_overwritten(
    tmp_path, monkeypatch
):
    path = tmp_path / "provider.tf"
    path.write_text('provider "nebius" {}')
    original_inspect = rpc._inspect

    def changed(workdir):
        result = original_inspect(workdir)
        path.write_text('provider "nebius" { timeout = "9m" }')
        return result

    monkeypatch.setattr(rpc, "_inspect", changed)
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 120)["reason_code"] == (
        "configuration_changed_during_inspection"
    )
    assert path.read_text() == 'provider "nebius" { timeout = "9m" }'


def test_existing_materialized_deadlines_survive_new_outer_budget(tmp_path):
    path = tmp_path / "provider.tf"
    path.write_text('provider "nebius" {}')
    rpc.configure_provider_rpc_deadlines(tmp_path, 120)
    materialized = path.read_bytes()
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 10)["inserted_defaults"] == {}
    assert path.read_bytes() == materialized


@pytest.mark.parametrize(
    "content",
    [
        'provider "nebius" {}\n',
        "provider nebius {}\n",
        'provider "neb\\u0069us" {}\n',
        'provider "nebius" {} # closing brace } comment\n',
        'provider "nebius" { domain = "api.example.invalid/{value}" }\n',
        'provider "nebius" { timeout = null }\n',
        'provider "nebius" { timeout = var.operator_deadline }\n',
        'locals {\n sample = <<-EOT\nprovider "nebius" {}\nEOT\n}\nprovider "nebius" {}\n',
        'provider "nebius" { alias = "other" }\nprovider "nebius" {}\n',
        'provider "other" { timeout = "2s" }\nprovider "nebius" {}\n',
        'provider "nebius" /* brace { } */ { timeout = null }\n',
    ],
)
def test_native_terraform_accepts_materialized_syntax(tmp_path, content):
    terraform = shutil.which("terraform")
    if not terraform:
        pytest.skip("native Terraform syntax control requires a local Terraform binary")
    path = tmp_path / "provider.tf"
    path.write_text(content)
    for phase in ("input", "output"):
        result = subprocess.run(
            [terraform, "fmt", "-write=false", "-no-color", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (phase, result.stderr)
        if phase == "input":
            rpc.configure_provider_rpc_deadlines(tmp_path, 73)


def test_unsupported_comment_before_label_refuses_without_writes(tmp_path):
    path = tmp_path / "provider.tf"
    original = 'provider /* brace { } */ "nebius" {}\n'
    path.write_text(original)
    assert rpc.configure_provider_rpc_deadlines(tmp_path, 120)["reason_code"] == (
        "unsupported_recipe_shape"
    )
    assert path.read_text() == original


@pytest.mark.parametrize(
    "suffix,body", [("tf", 'provider "nebius" {'), ("tf.json", '{"provider": []}')]
)
def test_advisory_never_partially_edits_other_valid_inputs(tmp_path, suffix, body):
    valid = tmp_path / "a.tf"
    valid.write_text('provider "nebius" {}')
    (tmp_path / ("z." + suffix)).write_text(body)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    receipt = rpc.configure_provider_rpc_deadlines(tmp_path, 120)
    assert receipt["status"] == "advisory"
    assert receipt["inserted_defaults"] == {}
    assert receipt["configuration_changed"] is False
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_symlinked_recipe_directory_is_not_traversed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    provider = outside / "provider.tf"
    provider.write_text('provider "nebius" {}')
    recipe = tmp_path / "recipe"
    recipe.symlink_to(outside, target_is_directory=True)
    assert (
        rpc.configure_provider_rpc_deadlines(recipe, 12)["reason_code"]
        == "unsafe_recipe_path"
    )
    assert provider.read_text() == 'provider "nebius" {}'


def test_failed_atomic_replacement_is_advisory_and_preserves_original(
    tmp_path, monkeypatch
):
    provider = tmp_path / "provider.tf"
    provider.write_text('provider "nebius" {}')

    def denied(*args):
        raise PermissionError("private path must not escape")

    monkeypatch.setattr(rpc.os, "replace", denied)
    receipt = rpc.configure_provider_rpc_deadlines(tmp_path, 12)
    assert receipt["reason_code"] == "recipe_io_unavailable"
    assert "private" not in json.dumps(receipt)
    assert provider.read_text() == 'provider "nebius" {}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["provider.tf"]
