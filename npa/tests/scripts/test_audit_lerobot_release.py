"""The LeRobot release pre-flight must fail on the breakages it claims to catch.

A green checker that cannot go red is worse than no checker, so each check is
driven against a synthetic wheel tree that is deliberately broken in exactly one
way. The fixture mirrors the real wheel layout (``lerobot/`` package plus a
``lerobot-<v>.dist-info/`` directory) closely enough for ``ast``-based
inspection, which is all the script does — it never imports LeRobot.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "npa" / "scripts" / "audit_lerobot_release.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_lerobot_release", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit_mod = _load()


METADATA = """Metadata-Version: 2.4
Name: lerobot
Version: {version}
Requires-Python: >=3.12
Requires-Dist: torch<2.12.0,>=2.7
Requires-Dist: torchvision<0.27.0,>=0.22.0
Requires-Dist: torchcodec<0.12.0,>=0.3.0; extra == "dataset"
Requires-Dist: diffusers<0.40.0,>=0.38.0; extra == "diffusers-dep"
Requires-Dist: transformers<5.6.0,>=5.4.0; extra == "transformers-dep"
Requires-Dist: lerobot[diffusers-dep]; extra == "diffusion"
Requires-Dist: lerobot[transformers-dep]; extra == "smolvla"
Requires-Dist: lerobot[transformers-dep]; extra == "libero"
"""

ENTRY_POINTS = """[console_scripts]
lerobot-train = lerobot.scripts.lerobot_train:main
lerobot-eval = lerobot.scripts.lerobot_eval:main
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    """A synthetic wheel tree that satisfies every check."""

    root = tmp_path / "lerobot-9.9.9"
    dist = root / "lerobot-9.9.9.dist-info"
    _write(dist / "METADATA", METADATA.format(version="9.9.9"))
    _write(dist / "entry_points.txt", ENTRY_POINTS)

    for module, symbol, _ in audit_mod.IMPORT_SURFACE:
        rel = module.replace(".", "/")
        target = root / rel / "__init__.py" if module == "lerobot.datasets" else root / f"{rel}.py"
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if symbol and symbol not in existing:
            kind = "class" if symbol[0].isupper() else "def"
            body = f"{kind} {symbol}: ...\n" if kind == "class" else f"def {symbol}(*args, **kwargs): ...\n"
            _write(target, existing + body)

    # Callables whose parameter names the script asserts on.
    _write(
        root / "lerobot/policies/factory.py",
        "def make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None): ...\n"
        "def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs): ...\n",
    )
    _write(
        root / "lerobot/policies/utils.py",
        "def prepare_observation_for_inference(observation, device, task=None, robot_type=None): ...\n",
    )
    _write(
        root / "lerobot/datasets/lerobot_dataset.py",
        "class LeRobotDataset:\n"
        "    def __init__(self, repo_id, root=None, episodes=None, revision=None): ...\n",
    )
    _write(root / "lerobot/optim/factory.py", "def make_optimizer_and_scheduler(cfg, policy): ...\n")

    _write(root / "lerobot/datasets/dataset_metadata.py", 'CODEBASE_VERSION = "v3.0"\n')
    _write(
        root / "lerobot/configs/train.py",
        "class TrainPipelineConfig:\n    env_eval_freq: int = 20_000\n",
    )
    _write(
        root / "lerobot/configs/policies.py",
        "class PreTrainedConfig:\n    pretrained_path: str | None = None\n",
    )
    _write(root / "lerobot/configs/parser.py", 'PATH_KEY = "path"\n')
    return root


def _run(monkeypatch, wheel: Path, manifest: dict | None = None):
    monkeypatch.setattr(audit_mod, "fetch_wheel", lambda *a, **k: wheel)
    monkeypatch.setattr(audit_mod, "load_manifest_entry", lambda version: manifest)
    return audit_mod.audit("9.9.9", wheel.parent, offline=True)


def _status(report, name: str) -> str:
    return next(c["status"] for c in report.checks if c["check"] == name)


def test_intact_wheel_passes_every_check(monkeypatch, wheel):
    report = _run(monkeypatch, wheel)
    assert report.failed == [], report.failed


def test_renamed_module_is_caught(monkeypatch, wheel):
    (wheel / "lerobot/policies/factory.py").unlink()
    report = _run(monkeypatch, wheel)
    assert _status(report, "import-surface") == "FAIL"


def test_removed_symbol_is_caught(monkeypatch, wheel):
    path = wheel / "lerobot/utils/device_utils.py"
    path.write_text("def something_else(): ...\n", encoding="utf-8")
    report = _run(monkeypatch, wheel)
    assert _status(report, "import-surface") == "FAIL"


def test_dropped_keyword_parameter_is_caught(monkeypatch, wheel):
    # `env_cfg` is passed by keyword from npa/server/app.py.
    (wheel / "lerobot/policies/factory.py").write_text(
        "def make_policy(cfg, ds_meta=None): ...\n"
        "def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs): ...\n",
        encoding="utf-8",
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "callable-parameters") == "FAIL"


def test_kwargs_absorbs_unknown_parameters(monkeypatch, wheel):
    (wheel / "lerobot/optim/factory.py").write_text(
        "def make_optimizer_and_scheduler(**kwargs): ...\n", encoding="utf-8"
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "callable-parameters") == "PASS"


def test_missing_entry_point_is_caught(monkeypatch, wheel):
    dist = wheel / "lerobot-9.9.9.dist-info" / "entry_points.txt"
    dist.write_text("[console_scripts]\nlerobot-train = x:main\n", encoding="utf-8")
    report = _run(monkeypatch, wheel)
    assert _status(report, "console-entry-points") == "FAIL"


def test_dataset_format_bump_is_caught(monkeypatch, wheel):
    (wheel / "lerobot/datasets/dataset_metadata.py").write_text(
        'CODEBASE_VERSION = "v4.0"\n', encoding="utf-8"
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "dataset-codebase-version") == "FAIL"


def test_renamed_train_flag_is_caught(monkeypatch, wheel):
    (wheel / "lerobot/configs/train.py").write_text(
        "class TrainPipelineConfig:\n    validation_cadence: int = 1\n", encoding="utf-8"
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "train-env-eval-flag") == "FAIL"


def test_manifest_flag_disagreeing_with_wheel_is_caught(monkeypatch, wheel):
    report = _run(monkeypatch, wheel, manifest={"train_env_eval_flag": "eval_freq"})
    assert _status(report, "manifest-agrees-with-wheel") == "FAIL"


def test_manifest_flag_agreeing_with_wheel_passes(monkeypatch, wheel):
    report = _run(monkeypatch, wheel, manifest={"train_env_eval_flag": "env_eval_freq"})
    assert _status(report, "manifest-agrees-with-wheel") == "PASS"


def test_forced_torch_pin_outside_declared_bounds_is_caught(monkeypatch, wheel):
    # Upstream declares torch<2.12.0; forcing 2.12.1 afterwards ships a broken image.
    report = _run(monkeypatch, wheel, manifest={"torch_pin": "torch==2.12.1"})
    assert _status(report, "manifest torch pins") == "FAIL"


def test_forced_torch_pin_inside_declared_bounds_passes(monkeypatch, wheel):
    report = _run(monkeypatch, wheel, manifest={"torch_pin": "torch==2.9.0"})
    assert _status(report, "manifest torch pins") == "PASS"


def test_lower_bound_pin_below_declared_ceiling_is_caught(monkeypatch, wheel):
    # `diffusers>=0.30.0` floors below the declared >=0.38.0 floor.
    report = _run(monkeypatch, wheel, manifest={"diffusers_pin": "diffusers>=0.30.0"})
    assert _status(report, "manifest torch pins") == "FAIL"


def _gate_diffusion_on_diffusers(wheel: Path) -> None:
    """Mirror 0.6.x, which enforces the extra inside ``DiffusionPolicy.__init__``."""

    (wheel / "lerobot/policies/diffusion/modeling_diffusion.py").write_text(
        "class DiffusionPolicy:\n"
        "    def __init__(self, config):\n"
        '        require_package("diffusers", extra="diffusion")\n',
        encoding="utf-8",
    )


def test_policy_extra_missing_from_manifest_is_caught(monkeypatch, wheel):
    # The module still imports; only construction fails. Checking the import
    # surface cannot see this, which is exactly how it shipped in 0.6.0.
    _gate_diffusion_on_diffusers(wheel)
    report = _run(monkeypatch, wheel, manifest={"pip_extras": "training,evaluation,pusht,libero"})
    assert _status(report, "import-surface") == "PASS"
    assert _status(report, "policy-extras") == "FAIL"


def test_policy_extra_present_in_manifest_passes(monkeypatch, wheel):
    # `diffusion` only reaches diffusers through `lerobot[diffusers-dep]`, so
    # this also proves the self-referential extra graph is walked.
    _gate_diffusion_on_diffusers(wheel)
    report = _run(monkeypatch, wheel, manifest={"pip_extras": "pusht,diffusion"})
    assert _status(report, "policy-extras") == "PASS"


def test_policy_extras_skipped_when_version_unknown_to_manifest(monkeypatch, wheel):
    _gate_diffusion_on_diffusers(wheel)
    report = _run(monkeypatch, wheel, manifest=None)
    assert _status(report, "policy-extras") == "PASS"


def test_ungated_policy_needs_no_extra(monkeypatch, wheel):
    # ACT declares no require_package gate, so a bare install can construct it.
    report = _run(monkeypatch, wheel, manifest={"pip_extras": "pusht"})
    assert _status(report, "policy-extras") == "PASS"


def test_repo_manifest_versions_are_all_auditable():
    """Every version this repo claims to support must be described well enough to check."""

    manifest = json.loads(
        (REPO_ROOT / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text(encoding="utf-8")
    )
    for version in manifest["supported_versions"]:
        entry = audit_mod.load_manifest_entry(version)
        assert entry is not None, f"{version} listed as supported but has no manifest entry"
        assert entry.get("train_env_eval_flag"), f"{version} declares no train_env_eval_flag"
