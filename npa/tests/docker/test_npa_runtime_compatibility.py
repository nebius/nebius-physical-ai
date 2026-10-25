"""Keep installed NPA image closures compatible with the supported application."""

from pathlib import Path
import re
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
import pytest


ROOT = Path(__file__).resolve().parents[3]
PROJECT = tomllib.loads((ROOT / "npa/pyproject.toml").read_text())["project"]


def test_supported_python_floor_matches_the_image_decoder():
    supported = SpecifierSet(PROJECT["requires-python"])
    assert "3.10.99" not in supported
    assert "3.11.0" in supported
    pillow = next(
        Requirement(d) for d in PROJECT["dependencies"] if d.startswith("pillow")
    )
    assert "12.3.0" not in pillow.specifier
    assert "13.0.0" in pillow.specifier


@pytest.mark.parametrize(
    ("tool", "filename"),
    [
        ("curobo", "requirements.lock"),
        ("mjlab", "requirements.lock"),
        ("lyra2", "requirements.lock"),
        ("seedvr2", "service-requirements.lock"),
    ],
)
def test_installed_npa_image_lock_satisfies_current_application(tool, filename):
    lock = (ROOT / "npa/docker/workbench" / tool / filename).read_text()
    pins = {
        canonicalize_name(name): version
        for name, version in re.findall(r"^([\w.-]+)==([^\s;\\]+)", lock, re.MULTILINE)
    }
    environment = default_environment() | {
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "extra": "",
    }
    for value in PROJECT["dependencies"]:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        assert name in pins, f"{tool} omits {requirement}"
        assert pins[name] in requirement.specifier, (
            f"{tool}: {name}=={pins[name]} violates {requirement}"
        )
