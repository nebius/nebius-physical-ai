"""Keep FiftyOne's advertised golden checks bound to its selected smoke script."""

import ast
from pathlib import Path
import shlex

from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.smoke.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]


def test_fiftyone_golden_selects_the_cpu_brain_and_app_checks() -> None:
    spec = load_manifest()["fiftyone"]
    command = shlex.split(spec.golden_eval.command)
    script = Path(command[1]).relative_to("/opt/npa")
    module = ast.parse((ROOT / "npa" / script).read_text())
    main = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    checks = next(
        node.value
        for node in ast.walk(main)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "checks"
    )
    invoked = {node.id for node in checks.elts if isinstance(node, ast.Name)}

    assert spec.golden_eval.gpu == "none"
    assert {"check_brain_curation", "check_launch_app"} <= invoked
    assert len(invoked) == len(GOLDEN_EVAL_CAPABILITIES["fiftyone"])
