"""Verify the consumed Helm values path and migration of legacy userns patches."""

import os
import subprocess

import pytest
import yaml

from npa.soperator import lifecycle


TEMPLATE = """resources:
  - apiVersion: v1
    kind: ConfigMap
    metadata:
      name: terraform-fluxcd-values
    data:
      values.yaml: |
        soperator:
          nodeConfigurator:
            enabled: true
            values:
              resources: {}
          anotherRelease:
            enabled: true
"""


def test_userns_values_reach_the_optional_chart_configmap():
    patched, changed = lifecycle._patch_nodeconfigurator_text(TEMPLATE)
    resources = yaml.safe_load(patched)["resources"]
    assert changed and len(resources) == 2
    assert resources[0] == yaml.safe_load(TEMPLATE)["resources"][0]
    configmap = resources[1]
    assert configmap["metadata"]["name"] == "terraform-nodeconfigurator"
    assert configmap["metadata"]["labels"]["reconcile.fluxcd.io/watch"] == "Enabled"
    values = yaml.safe_load(configmap["data"]["values.yaml"])
    assert set(values) == {"initContainers"}
    assert len(values["initContainers"]) == 1
    assert values["initContainers"][0]["image"] == "docker.io/library/busybox:stable"
    command = values["initContainers"][0]["command"][2]
    assert command.startswith("set -eu\n")
    assert '[ "${apparmor_enabled}" = "false" ]' in command
    assert "sysctl -w kernel.apparmor_restrict_unprivileged_userns=0" in command
    assert "sysctl -w net.core.wmem_max=536870912" in command
    assert lifecycle._patch_nodeconfigurator_text(patched) == (patched, False)


def test_existing_legacy_patch_migrates_without_changing_other_chart_values():
    legacy, _ = lifecycle._legacy_nodeconfigurator_text(TEMPLATE)
    migrated, changed = lifecycle._patch_nodeconfigurator_text(legacy)
    fresh, _ = lifecycle._patch_nodeconfigurator_text(TEMPLATE)
    assert changed
    assert migrated == fresh
    assert migrated.count(lifecycle._NODECONFIGURATOR_USERNS_MARKER) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_modified_known_patch_is_not_silently_accepted(legacy):
    patch = (
        lifecycle._legacy_nodeconfigurator_text
        if legacy
        else lifecycle._patch_nodeconfigurator_text
    )
    modified = patch(TEMPLATE)[0].replace("net.core.wmem_max=536870912", "bad-command")
    with pytest.raises(lifecycle.UpstreamContractError, match="modified"):
        lifecycle._patch_nodeconfigurator_text(modified)


def test_sysctl_failure_cannot_be_hidden_by_a_later_success(tmp_path):
    patched, _ = lifecycle._patch_nodeconfigurator_text(TEMPLATE)
    configmap = yaml.safe_load(patched)["resources"][1]
    values = yaml.safe_load(configmap["data"]["values.yaml"])
    command = values["initContainers"][0]["command"][2]
    probe = tmp_path / "sysctl"
    probe.write_text(
        '#!/bin/sh\ncase "$*" in *unprivileged_userns_clone*) exit 17;; esac\nexit 0\n'
    )
    probe.chmod(0o700)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env=dict(os.environ, PATH=str(tmp_path)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 17
