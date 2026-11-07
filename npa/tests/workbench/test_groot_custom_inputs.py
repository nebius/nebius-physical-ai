"""Contracts for explicitly trusted custom LeRobot extension archives."""

import hashlib
from pathlib import Path
import tarfile

import numpy as np
import pytest

from npa.workbench.groot.custom_inputs import (
    ActionContract,
    ExternalCodeSpec,
    InputProvenance,
    extract_content_addressed_archive,
    install_external_code,
    invoke_external_callable,
    verify_provenance_selection,
    verify_minmax_roundtrip,
)


def test_input_provenance_requires_a_stable_source_identifier() -> None:
    with pytest.raises(ValueError, match="source_identifier"):
        InputProvenance.model_validate(
            {
                "source_kind": "archive",
                "source_revision": "revision-1",
                "source_sha256": "a" * 64,
                "license": "operator-provided",
                "privacy_classification": "private",
                "redistributable_media": False,
                "label_provenance": "operator sidecar",
            }
        )


def test_provenance_selection_is_bound_to_the_explicit_workflow_override() -> None:
    provenance = InputProvenance(
        source_kind="archive",
        source_identifier="operator/reference-data",
        source_revision="immutable-revision",
        source_sha256="a" * 64,
        license="operator-provided",
        privacy_classification="private",
        redistributable_media=False,
        label_provenance="operator sidecar",
    )

    verify_provenance_selection(
        provenance, "operator/reference-data", "immutable-revision"
    )
    with pytest.raises(ValueError, match="differs"):
        verify_provenance_selection(provenance, "other/data", "immutable-revision")


def _archive_extension(tmp_path: Path) -> Path:
    project = tmp_path / "reference-extension"
    package = project / "src/reference_extension"
    package.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "reference-extension"
version = "0.0.1"
[tool.setuptools.packages.find]
where = ["src"]
"""
    )
    (package / "__init__.py").write_text("")
    (package / "adapter.py").write_text(
        """def adapt(request):
    return {"dataset_root": request["output_dataset_root"], "request_schema": request["schema_version"]}
def feedback(request):
    return {"feedback_path": request["output_feedback_path"], "has_adapter_result": "adapter_result" in request}
"""
    )
    archive = tmp_path / "reference-extension.tar.gz"
    with tarfile.open(archive, "w:gz") as opened:
        opened.add(project, arcname=project.name)
    return archive


def test_installs_and_invokes_only_declared_hash_bound_extension(tmp_path: Path):
    archive = _archive_extension(tmp_path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    spec = ExternalCodeSpec(
        archive_sha256=digest,
        package="reference_extension",
        adapter_callable="reference_extension.adapter:adapt",
        feedback_callable="reference_extension.adapter:feedback",
    )
    installed = install_external_code(archive, spec, tmp_path / "installed")
    request = {
        "schema_version": "npa.groot-custom-adapter-request/v1",
        "output_dataset_root": "/run/dataset",
        "output_feedback_path": "/run/feedback.json",
    }
    adapted = invoke_external_callable(installed, spec.adapter_callable, request)
    feedback = invoke_external_callable(
        installed, spec.feedback_callable, dict(request, adapter_result=adapted)
    )
    assert adapted["dataset_root"] == "/run/dataset"
    assert feedback == {
        "feedback_path": "/run/feedback.json",
        "has_adapter_result": True,
    }


def test_rejects_digest_mismatch_and_archive_path_escape(tmp_path: Path):
    archive = _archive_extension(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        extract_content_addressed_archive(archive, "0" * 64, tmp_path / "wrong")
    escaped = tmp_path / "escaped.tar"
    with tarfile.open(escaped, "w") as opened:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        import io

        opened.addfile(entry, io.BytesIO(b"x"))
    digest = hashlib.sha256(escaped.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe"):
        extract_content_addressed_archive(escaped, digest, tmp_path / "unsafe")


def test_cartesian_contract_fixture_roundtrips_declared_units_and_terminal_mask():
    """A numerical fixture, not a claim about a recorded robot trajectory."""

    contract = ActionContract(
        kind="cartesian_relative",
        target_keys=["action.tool_delta", "action.gripper"],
        frame="tool",
        translation_unit="millimeters",
        rotation_representation="axis_angle",
        rotation_unit="radians",
        gripper_fields=["opening_fraction"],
        transform_identity="fixture-affine-min-max-v1",
    )
    evidence = verify_minmax_roundtrip(
        np.asarray([[12.5, -4.0, 2.0, 0.1, -0.2, 0.3, 0.75]] * 4),
        np.asarray([-20.0, -10.0, -5.0, -1.0, -1.0, -1.0, 0.0]),
        np.asarray([20.0, 10.0, 5.0, 1.0, 1.0, 1.0, 1.0]),
        np.asarray([True, False, False, False]),
    )
    assert contract.frame == "tool"
    assert contract.translation_unit == "millimeters"
    assert contract.rotation_representation == "axis_angle"
    assert evidence["valid_action_horizon_steps"] == 1
    assert evidence["masked_action_horizon_steps"] == 3
    assert evidence["maximum_absolute_roundtrip_error"] < 1e-5


def test_roundtrip_rejects_nonfinite_or_misaligned_action_contract_inputs():
    with pytest.raises(ValueError, match="dimensions"):
        verify_minmax_roundtrip(
            np.zeros((2, 2)), np.zeros(3), np.ones(3), np.asarray([True, False])
        )
    with pytest.raises(ValueError, match="finite"):
        verify_minmax_roundtrip(
            np.asarray([[float("nan")]]), np.zeros(1), np.ones(1), np.asarray([True])
        )
