"""Keep literal Docker build dependency pins visible to vulnerability scans."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def dependencies(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("security_dependencies")


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.training"])
def test_all_stages_retain_conflicting_package_versions(tmp_path, dependencies, name):
    source = tmp_path / "source"
    source.mkdir()
    (source / name).write_text(
        'FROM scratch AS build\nRUN pip install "requests==2.19.1"\n'
        "FROM scratch\nRUN python -m pip install \\\n"
        '  "requests==2.32.5" "accelerate==1.14.0"\n'
    )
    output = tmp_path / "report"
    inventory = dependencies._inventory(source, output, tmp_path / "cache")
    assert set(inventory.values()) == {name}
    assert len(inventory) == 3
    packages = set()
    for target in inventory:
        expected = dependencies._expected_packages(output / "inputs" / target)
        assert len(expected) == 1
        packages.update(expected)
    assert packages == {
        ("requests", "2.19.1"),
        ("requests", "2.32.5"),
        ("accelerate", "1.14.0"),
    }
    with pytest.raises(ValueError, match="omitted declared packages"):
        dependencies._validate_coverage({"Results": []}, inventory, output)


def test_shell_comments_and_labels_do_not_invent_dependencies(tmp_path, dependencies):
    source = tmp_path / "Dockerfile"
    source.write_text(
        '# RUN pip install requests==2.19.1\nLABEL example="requests==2.19.1"\n'
        'RUN pip install "urllib3==2.8.0" # requests==2.19.1\n'
    )
    inventory = dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)
    assert len(inventory) == 1
    target = next(iter(inventory))
    assert dependencies._expected_packages(tmp_path / "inputs" / target) == {
        ("urllib3", "2.8.0")
    }


def test_quoted_markers_and_adjacent_shell_operators_preserve_pins(
    tmp_path, dependencies
):
    source = tmp_path / "Dockerfile"
    source.write_text(
        "RUN pip install 'requests[socks]==2.19.1; python_version < \"3.14\"'&&pip check\n"
    )
    inventory = dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)
    target = next(iter(inventory))
    assert dependencies._expected_packages(tmp_path / "inputs" / target) == {
        ("requests", "2.19.1")
    }


def test_malformed_shell_cannot_produce_an_empty_success(tmp_path, dependencies):
    source = tmp_path / "Dockerfile"
    source.write_text('RUN pip install "requests==2.19.1\n')
    with pytest.raises(ValueError, match="No closing quotation"):
        dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)


def test_json_exec_form_retains_exact_pins(tmp_path, dependencies):
    source = tmp_path / "Dockerfile"
    source.write_text('RUN ["python", "-m", "pip", "install", "requests==2.19.1"]\n')
    inventory = dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)
    target = next(iter(inventory))
    assert dependencies._expected_packages(tmp_path / "inputs" / target) == {
        ("requests", "2.19.1")
    }


def test_malformed_exec_arguments_fail_closed(tmp_path, dependencies):
    source = tmp_path / "Dockerfile"
    source.write_text('RUN ["pip", "install", "requests==2.19.1", 3]\n')
    with pytest.raises(ValueError, match="requires string arguments"):
        dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)


def test_shell_without_literal_pins_needs_no_dependency_lexing(tmp_path, dependencies):
    source = tmp_path / "Dockerfile"
    source.write_text(
        'RUN test "$(sed -n \'s/^version = "\\([^"]*\\)"/\\1/p\' file)" = 1\n'
    )
    inventory = dependencies._dockerfile_inventory(source, "Dockerfile", tmp_path)
    assert inventory == {}
