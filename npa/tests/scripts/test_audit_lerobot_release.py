"""The LeRobot release pre-flight must fail on the breakages it claims to catch.

A green checker that cannot go red is worse than no checker, so each check is
driven against a synthetic wheel tree that is deliberately broken in exactly one
way. The fixture mirrors the real wheel layout (``lerobot/`` package plus a
``lerobot-<v>.dist-info/`` directory) closely enough for ``ast``-based
inspection, which is all the script does — it never imports LeRobot.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import packaging.markers as packaging_markers
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


def test_symbol_nested_inside_a_function_is_not_a_module_export(monkeypatch, wheel):
    path = wheel / "lerobot/utils/device_utils.py"
    path.write_text(
        "def device_factory():\n"
        "    def get_safe_torch_device(): ...\n"
        "    return get_safe_torch_device\n",
        encoding="utf-8",
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "import-surface") == "FAIL"


@pytest.mark.parametrize(
    "source",
    [
        "try:\n"
        "    from backend import get_safe_torch_device\n"
        "except ModuleNotFoundError:\n"
        "    from fallback import get_safe_torch_device\n",
        "try:\n"
        "    from backend import get_safe_torch_device\n"
        "except Exception:\n"
        "    from fallback import get_safe_torch_device\n",
        "if sys.version_info >= (3, 12):\n"
        "    from backend import get_safe_torch_device\n",
        "try:\n"
        "    from backend import get_safe_torch_device\n"
        "finally:\n"
        "    cleaned_up = True\n",
        pytest.param(
            "try:\n"
            "    from backend import get_safe_torch_device\n"
            "except* Exception:\n"
            "    from fallback import get_safe_torch_device\n",
            id="try-star",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 11),
                reason="except* syntax requires Python 3.11",
            ),
        ),
        "with import_context():\n"
        "    from backend import get_safe_torch_device\n",
        "for backend in backends:\n"
        "    from backend import get_safe_torch_device\n",
        "while select_backend():\n"
        "    from backend import get_safe_torch_device\n",
        "match backend_name:\n"
        "    case _:\n"
        "        from backend import get_safe_torch_device\n",
    ],
    ids=[
        "module-not-found-error",
        "broad-exception",
        "version-if",
        "try-finally",
        "try-star",
        "with",
        "for",
        "while",
        "match",
    ],
)
def test_module_scope_conditional_exports_are_found(monkeypatch, wheel, source):
    (wheel / "lerobot/utils/device_utils.py").write_text(source, encoding="utf-8")
    report = _run(monkeypatch, wheel)
    assert _status(report, "import-surface") == "PASS"


def _make_device_export_lazy(wheel: Path) -> None:
    (wheel / "lerobot/utils/device_utils.py").write_text(
        "def __getattr__(name):\n"
        "    if name == 'get_safe_torch_device':\n"
        "        return load_device_helper()\n"
        "    raise AttributeError(name)\n",
        encoding="utf-8",
    )


def test_lazy_getattr_export_is_an_unverifiable_warning(monkeypatch, wheel):
    _make_device_export_lazy(wheel)
    report = _run(monkeypatch, wheel)
    check = next(c for c in report.checks if c["check"] == "import-surface")

    assert check["status"] == "WARN"
    assert "lazily re-exported" in check["detail"]
    assert "could not verify statically" in check["detail"]
    assert "symbol gone" not in check["detail"]


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


def test_bare_requires_dist_does_not_disable_pin_check(monkeypatch, wheel):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(
            "Requires-Dist: torch<2.12.0,>=2.7",
            "Requires-Dist: torch\nRequires-Dist: torch<2.12.0,>=2.7",
        ),
        encoding="utf-8",
    )
    report = _run(monkeypatch, wheel, manifest={"torch_pin": "torch==99.0.0"})
    assert _status(report, "manifest torch pins") == "FAIL"


def test_marker_gated_requirement_is_not_applied_unconditionally(monkeypatch, wheel):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(
            "Requires-Dist: torch<2.12.0,>=2.7",
            'Requires-Dist: torch==2.5.0; platform_machine == "aarch64"\n'
            "Requires-Dist: torch<2.12.0,>=2.7",
        ),
        encoding="utf-8",
    )
    report = _run(monkeypatch, wheel)
    assert _status(report, "b300-image torch stack") == "PASS"


def test_marker_evaluation_uses_x86_64_linux_image_not_aarch64_host(
    monkeypatch, wheel
):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(
            "Requires-Dist: torch<2.12.0,>=2.7",
            'Requires-Dist: torch==2.5.0; platform_machine == "aarch64"\n'
            "Requires-Dist: torch<2.12.0,>=2.7",
        ),
        encoding="utf-8",
    )
    aarch64_host = packaging_markers.default_environment()
    aarch64_host.update(
        {
            "os_name": "posix",
            "platform_machine": "aarch64",
            "platform_system": "Linux",
            "sys_platform": "linux",
        }
    )
    monkeypatch.setattr(
        packaging_markers,
        "default_environment",
        lambda: aarch64_host.copy(),
    )

    report = _run(monkeypatch, wheel)
    assert _status(report, "b300-image torch stack") == "PASS"


def test_uninstalled_extra_does_not_constrain_image_pins(monkeypatch, wheel):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8")
        + 'Requires-Dist: torch==2.7.1; extra == "jetson"\n',
        encoding="utf-8",
    )

    report = _run(monkeypatch, wheel, manifest={"pip_extras": "training,evaluation"})
    assert _status(report, "b300-image torch stack") == "PASS"


def test_installed_extra_still_constrains_image_pins(monkeypatch, wheel):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8")
        + 'Requires-Dist: torch==2.7.1; extra == "jetson"\n',
        encoding="utf-8",
    )

    report = _run(monkeypatch, wheel, manifest={"pip_extras": "jetson"})
    assert _status(report, "b300-image torch stack") == "FAIL"


def test_transitively_installed_extra_constrains_image_pins(monkeypatch, wheel):
    metadata = wheel / "lerobot-9.9.9.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8")
        + 'Requires-Dist: lerobot[jetson]; extra == "hardware"\n'
        + 'Requires-Dist: torch==2.7.1; extra == "jetson"\n',
        encoding="utf-8",
    )

    report = _run(monkeypatch, wheel, manifest={"pip_extras": "hardware"})
    assert _status(report, "b300-image torch stack") == "FAIL"


@pytest.mark.parametrize(
    "pin",
    ["torch<2.11.0", "torch~=2.9.0", "torch!=2.10.0", "torch>=2.7,<2.11"],
)
def test_pin_floor_handles_every_specifier_form(monkeypatch, wheel, pin):
    report = _run(monkeypatch, wheel, manifest={"torch_pin": pin})
    assert _status(report, "manifest torch pins") == "FAIL"


def test_lower_bound_pin_below_declared_ceiling_is_caught(monkeypatch, wheel):
    # `diffusers>=0.30.0` floors below the declared >=0.38.0 floor.
    report = _run(
        monkeypatch,
        wheel,
        manifest={"diffusers_pin": "diffusers>=0.30.0", "pip_extras": "diffusion"},
    )
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


def test_missing_dist_info_is_a_check_failure_not_a_traceback(monkeypatch, wheel):
    shutil.rmtree(wheel / "lerobot-9.9.9.dist-info")
    report = _run(monkeypatch, wheel)
    assert [(check["check"], check["status"]) for check in report.checks] == [
        ("wheel-metadata", "FAIL")
    ]


def test_json_mode_stdout_is_pure_json(monkeypatch, wheel, capsys):
    monkeypatch.setattr(audit_mod, "fetch_wheel", lambda *a, **k: wheel)
    monkeypatch.setattr(audit_mod, "load_manifest_entry", lambda version: None)

    assert audit_mod.main(["9.9.9", "--offline", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["version"] == "9.9.9"


def test_lazy_warning_human_summary_is_honest_and_non_blocking(
    monkeypatch, wheel, capsys
):
    _make_device_export_lazy(wheel)
    monkeypatch.setattr(audit_mod, "fetch_wheel", lambda *a, **k: wheel)
    monkeypatch.setattr(
        audit_mod,
        "load_manifest_entry",
        lambda version: {"pip_extras": "", "train_env_eval_flag": "env_eval_freq"},
    )

    assert audit_mod.main(["9.9.9", "--offline"]) == 0
    captured = capsys.readouterr()
    assert "[WARN] import-surface" in captured.out
    assert "WARN: 1 non-blocking unverifiable finding(s)" in captured.out
    assert "PASS: lerobot 9.9.9 satisfies" not in captured.out
    assert captured.err == ""


def test_lazy_warning_json_stdout_is_pure_and_summary_is_on_stderr(
    monkeypatch, wheel, capsys
):
    _make_device_export_lazy(wheel)
    monkeypatch.setattr(audit_mod, "fetch_wheel", lambda *a, **k: wheel)
    monkeypatch.setattr(
        audit_mod,
        "load_manifest_entry",
        lambda version: {"pip_extras": "", "train_env_eval_flag": "env_eval_freq"},
    )

    assert audit_mod.main(["9.9.9", "--offline", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    import_check = next(c for c in payload[0]["checks"] if c["check"] == "import-surface")
    assert import_check["status"] == "WARN"
    assert "WARN: 1 non-blocking unverifiable finding(s)" in captured.err
    assert "PASS: lerobot 9.9.9 satisfies" not in captured.err


def test_import_surface_covers_every_static_lerobot_import():
    covered: dict[tuple[str, str | None], set[str]] = {}
    for module, symbol, provenance in audit_mod.IMPORT_SURFACE:
        assert isinstance(provenance, tuple), (
            f"{module}:{symbol} provenance must be a tuple of repository-relative paths"
        )
        covered.setdefault((module, symbol), set()).update(provenance)
    imports: dict[tuple[str, str | None], set[str]] = {}

    for root in (REPO_ROOT / "npa/src", REPO_ROOT / "npa/demo", REPO_ROOT / "research"):
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "lerobot" or node.module.startswith("lerobot.")
                ):
                    for alias in node.names:
                        imports.setdefault((node.module, alias.name), set()).add(relative)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "lerobot" or alias.name.startswith("lerobot."):
                            imports.setdefault((alias.name, None), set()).add(relative)

    missing_bindings = {
        binding: paths for binding, paths in imports.items() if binding not in covered
    }
    missing_provenance = {
        binding: paths - covered.get(binding, set())
        for binding, paths in imports.items()
        if paths - covered.get(binding, set())
    }
    assert missing_bindings == {}
    assert missing_provenance == {}


def test_b300_image_pins_match_the_dockerfile():
    dockerfile = (REPO_ROOT / "npa/docker/workbench/lerobot/Dockerfile.b300").read_text(
        encoding="utf-8"
    )
    found = {}
    for package in audit_mod.B300_IMAGE_PINS:
        matches = re.findall(rf"(?<![\w-]){re.escape(package)}==([\w.+-]+)", dockerfile)
        assert len(matches) == 1, f"expected one {package} pin, found {matches}"
        found[package] = matches[0]

    assert found == audit_mod.B300_IMAGE_PINS


def test_repo_manifest_versions_are_all_auditable():
    """Every version this repo claims to support must be described well enough to check."""

    manifest = json.loads(
        (REPO_ROOT / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text(encoding="utf-8")
    )
    for version in manifest["supported_versions"]:
        entry = audit_mod.load_manifest_entry(version)
        assert entry is not None, f"{version} listed as supported but has no manifest entry"
        assert entry.get("train_env_eval_flag"), f"{version} declares no train_env_eval_flag"
