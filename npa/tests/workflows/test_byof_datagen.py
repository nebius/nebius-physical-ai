from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "npa" / "scripts" / "run_byof_datagen.py"
YAML_PATH = ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "byof-datagen-rtxpro-smoke.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_byof_datagen", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_workflow_sets_datagen_envs() -> None:
    module = _load_module()
    docs = module.render_workflow(
        YAML_PATH,
        run_id="byof-datagen-test",
        task="LeIsaac-SO101-PickOrange-v0",
        num_envs=8,
        num_demos=20,
        image="cr.example.nebius.cloud/proj/npa-byof:test",
    )
    envs = next(doc["envs"] for doc in docs[1:] if isinstance(doc.get("envs"), dict))
    assert envs["NPA_BYOF_RUN_ID"] == "byof-datagen-test"
    assert envs["BYOF_TASK"] == "LeIsaac-SO101-PickOrange-v0"
    assert envs["BYOF_NUM_ENVS"] == "8"
    assert envs["BYOF_NUM_DEMOS"] == "20"
    assert envs["BYOF_REPO_ROOT"] == "/opt/byof"
    resources = next(doc["resources"] for doc in docs[1:] if isinstance(doc.get("resources"), dict))
    assert resources["image_id"] == "docker:cr.example.nebius.cloud/proj/npa-byof:test"


def test_main_render_only_writes_yaml(capsys) -> None:
    module = _load_module()
    rc = module.main(
        [
            "--yaml",
            str(YAML_PATH),
            "--run-id",
            "byof-datagen-render",
            "--render-only",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    rendered = Path(payload["rendered_yaml"])
    assert rendered.is_file()
    text = rendered.read_text(encoding="utf-8")
    assert "LeIsaac-SO101-PickOrange-v0" in text
    assert "byof-datagen-render" in text
    assert "scripts/datagen/state_machine/generate.py" in text
