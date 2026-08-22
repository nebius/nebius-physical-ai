"""Which node a detection-training deploy asks for.

Live: a deploy defaulted to the `l40s` selector on a cluster whose GPU nodes are labelled
`gpu-rtx6000`. The pod stayed Unschedulable and the only symptom was `rollout status` timing
out — nothing in the output mentioned node labels, which is why this took a `kubectl get nodes`
to see (EVIDENCE §R46).
"""

from __future__ import annotations

import pytest

from npa.cli.workbench.detection_training import GPU_NODE_SELECTORS


def test_the_workbench_gpu_cluster_is_selectable() -> None:
    """The RTX PRO 6000 nodes this repo's own live cluster runs."""

    assert GPU_NODE_SELECTORS["rtxpro6000"] == "gpu-rtx6000"
    assert GPU_NODE_SELECTORS["rtx6000"] == "gpu-rtx6000"


def test_the_existing_shorthands_are_unchanged() -> None:
    assert GPU_NODE_SELECTORS["h100"] == "gpu-h100-sxm"
    assert GPU_NODE_SELECTORS["l40s"] == "gpu-l40s-d"


def test_every_selector_is_an_instance_type_label_value() -> None:
    """A label value, not a label: the key is always node.kubernetes.io/instance-type."""

    for shorthand, value in GPU_NODE_SELECTORS.items():
        assert "/" not in value, f"{shorthand} looks like a label key, not a value"
        assert value.startswith("gpu-"), shorthand


@pytest.mark.parametrize("unknown", ["b200", "", "GPU"])
def test_an_unknown_shorthand_has_no_selector_so_the_cli_can_refuse(unknown: str) -> None:
    assert GPU_NODE_SELECTORS.get(unknown) is None


def test_the_ambient_kubeconfig_is_the_default(tmp_path, monkeypatch) -> None:
    """`--cluster-name` used to default to a profile, silently targeting another cluster.

    Live, that gave the least helpful failure available: `kubectl apply` reported "deployment
    configured", `rollout status` timed out, and the deployment existed in no namespace of the
    cluster being inspected — because it was on a different one (EVIDENCE §R46).
    """

    from npa.cli.workbench.detection_training import _resolve_kubeconfig

    assert _resolve_kubeconfig(cluster_name="", kubeconfig="") == ""


def test_an_explicit_profile_is_still_honoured(tmp_path, monkeypatch) -> None:
    from npa.cli.workbench import detection_training as dt

    cached = tmp_path / ".npa" / "clusters" / "some-cluster" / "kubeconfig"
    cached.parent.mkdir(parents=True)
    cached.write_text("apiVersion: v1")
    monkeypatch.setattr(dt.Path, "home", staticmethod(lambda: tmp_path))

    assert dt._resolve_kubeconfig(cluster_name="some-cluster", kubeconfig="") == str(cached)


def test_an_explicit_path_wins_over_a_profile() -> None:
    from npa.cli.workbench.detection_training import _resolve_kubeconfig

    assert (
        _resolve_kubeconfig(cluster_name="some-cluster", kubeconfig="/tmp/kc")
        == "/tmp/kc"
    )


def test_every_command_taking_cluster_name_agrees_on_the_default() -> None:
    """Two commands with different default clusters would deploy here and inspect elsewhere."""

    import inspect

    from npa.cli.workbench import detection_training as dt

    defaults = {}
    for name in ("deploy_cmd", "list_cmd", "train_cmd", "status_cmd"):
        command = getattr(dt, name, None)
        if command is None:
            continue
        parameter = inspect.signature(command).parameters.get("cluster_name")
        if parameter is not None:
            defaults[name] = parameter.default.default
    assert defaults, "expected at least one command to take --cluster-name"
    assert set(defaults.values()) == {""}, defaults
