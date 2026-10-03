"""Keep RoboTwin authorization bound to the actual shared renderer."""

from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.robotwin_preflight import (
    _recognize_rendered_contract,
)
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)


@pytest.mark.parametrize("mutation", [None, "remove", "redirect"])
def test_robo_twin_binds_rendered_control_interpreter(monkeypatch, mutation):
    monkeypatch.setenv("NPA_SRC_S3_URI", "${NPA_SRC_S3_URI}")
    root = Path(__file__).resolve().parents[4]
    spec = load_spec(root / "workflows/testing/byof-robotwin.yaml")
    run_id = "robotwin-contract-test"
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id=run_id),
        run_id=run_id,
        options=SkypilotRenderOptions(
            materialize_registry_secrets=False,
            aws_endpoint_url="${AWS_ENDPOINT_URL}",
        ),
    )
    documents = list(yaml.safe_load_all(rendered))
    export = 'export NPA_CONTROL_PYTHON="$npa_python"\n'
    assert documents[1]["run"].count(export) == 1
    if mutation is not None:
        replacement = (
            ""
            if mutation == "remove"
            else "export NPA_CONTROL_PYTHON=/tmp/untrusted-python\n"
        )
        documents[1]["run"] = documents[1]["run"].replace(export, replacement)
    assert _recognize_rendered_contract(documents) is (mutation is None)
