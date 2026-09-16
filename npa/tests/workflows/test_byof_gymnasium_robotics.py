from __future__ import annotations

import ast
import hashlib
import json
import os
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/byof-gymnasium-robotics.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
PROFILE = (
    ROOT
    / "npa/src/npa/workflows/byof/profiles/byof-solution-smoke-gymnasium-robotics-rtxpro-gpu.yaml"
)
IMAGE_ROOT = ROOT / "npa/docker/workbench/gymnasium-robotics"
SMOKE = IMAGE_ROOT / "capability_smoke.py"
ASSETS = IMAGE_ROOT / "asset-lock.json"
SOURCE_COMMIT = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
ENV_ID = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _coordinator_source() -> str:
    profile = PROFILE.read_text(encoding="utf-8")
    return textwrap.dedent(profile.rsplit("<<'PY'\n", 1)[1].split("\n  PY", 1)[0])


def _coordinator_helpers(tmp_path: Path) -> dict[str, object]:
    syntax = ast.parse(_coordinator_source())
    helper_names = {
        "_close_bound_output_chain",
        "_descriptor_identity",
        "_directory_identity",
        "_open_bound_output_root",
        "_open_bound_log",
        "_require_bound_directory",
        "_require_bound_output_chain",
        "_require_owned_regular",
        "_require_bound_log_stat",
        "_read_bound_json",
        "_verify_bound_log",
        "_write_bound_summary",
        "_output_uploads",
    }
    body = [
        node
        for node in syntax.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MAX_OUTPUT_BYTES"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in helper_names)
    ]
    helper_path = tmp_path / "coordinator_helpers.py"
    helper_path.write_text(
        ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n",
        encoding="utf-8",
    )
    return runpy.run_path(str(helper_path))


def _bound_output_root(
    helpers: dict[str, object], root: Path
) -> tuple[int, tuple[int, ...], tuple[tuple[object, ...], ...]]:
    return helpers["_open_bound_output_root"](root)


def _close_output_root(
    helpers: dict[str, object], descriptors: tuple[int, ...]
) -> None:
    helpers["_close_bound_output_chain"](descriptors)


def test_neutral_bootstrap_uses_only_the_unbuilt_prebuilt_candidate() -> None:
    config = _workflow()["config"]
    assert config["repo_ref"] == SOURCE_COMMIT
    assert config["repo_auth"] == "none"
    assert config["base_profile"] == "prebuilt"
    assert (
        config["base_image"]
        == "registry.example.invalid/gymnasium-robotics:neutral-unbuilt"
    )
    assert config["build_command"] == ""
    assert config["smoke_command"].endswith(
        "exec /usr/local/bin/npa-gymnasium-entrypoint run-smoke\n"
    )
    assert config["capability_name"] == ENV_ID
    assert config["smoke_artifact_name"] == "gymnasium-robotics-smoke.json"


def test_neutral_image_lock_gate_accepts_only_the_reviewed_content_closure() -> None:
    completed = subprocess.run(
        [str(IMAGE_ROOT / "build.sh"), "verify-bootstrap-locks"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = (IMAGE_ROOT / "build.sh").read_text(encoding="utf-8")
    assert not any(
        command in text for command in ("curl ", "wget ", "git clone", "pip install")
    )


def test_runtime_and_neutral_baked_closures_are_exact_but_publicly_quarantined() -> (
    None
):
    source = json.loads((IMAGE_ROOT / "source-lock.json").read_text())
    apt = json.loads((IMAGE_ROOT / "apt-runtime.lock.json").read_text())
    corresponding = json.loads(
        (IMAGE_ROOT / "corresponding-source.lock.json").read_text()
    )
    assert {source["status"], apt["status"], corresponding["status"]} == {"complete"}
    assert source["source_commit"] == SOURCE_COMMIT
    assert source["mujoco_version"] == "3.12.0"
    assert source["components"]["shadow_sr_common"]["preferred_form_complete"] is False
    assert len(apt["resolved_binary_packages"]) == 142
    assert len(apt["resolved_source_packages"]) == 102
    assert (
        sum(len(item["artifacts"]) for item in apt["resolved_source_packages"]) == 318
    )
    assert "python3-boto3" in {
        item["package"] for item in apt["requested_runtime_packages"]
    }
    assert corresponding["scope"] == "candidate-image-layers-only"
    assert corresponding["runtime_fetched_material_excluded"]
    assert len(source["requirements_lock_sha256"]) == 64
    assert (
        source["resolved_python_artifact_count"]
        == source["expected_python_distribution_count"]
    )
    assert source["expected_python_distribution_count"] == 19
    runtime_requirements = (IMAGE_ROOT / "requirements.in").read_text()
    assert "boto3" not in runtime_requirements
    assert "botocore" not in runtime_requirements


def test_directly_loaded_shadow_asset_closure_is_exact() -> None:
    lock = json.loads(ASSETS.read_text(encoding="utf-8"))
    assert lock["source_commit"] == SOURCE_COMMIT
    assert len(lock["directly_loaded_xml"]) == 5
    assert len(lock["directly_loaded_mesh_texture"]) == 14
    assert set(Path(name).suffix for name in lock["directly_loaded_mesh_texture"]) == {
        ".stl",
        ".png",
    }
    assert all(
        len(value) == 64
        for group in (lock["directly_loaded_xml"], lock["directly_loaded_mesh_texture"])
        for value in group.values()
    )


def test_capability_script_keeps_the_real_hard_gate() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    compile(source, str(SMOKE), "exec")
    for token in (
        ENV_ID,
        "ROLLOUT_STEPS = 120",
        "raw.data.ncon",
        "raw.data.sensordata[touch_ids]",
        "env.step(action)",
        "env.render()",
        "libmujoco.so*",
        '"synthetic_only_fixture": False',
        '"pod_observed_image_digest"',
        '"physics_substeps"',
        '"max_reading"',
        '"distinct_rgb_frame_sha256"',
        '"exit_status": 0',
    ):
        assert token in source
    assert (
        "RTX PRO 6000" in source
        and "Blackwell" in source
        and 'capability != "12.0"' in source
    )


def test_workflow_and_profile_never_route_to_b200() -> None:
    workflow = _workflow()
    assert (
        workflow["resources"]["gpu"]["accelerators"]
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    profile = PROFILE.read_text(encoding="utf-8")
    assert "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in profile
    assert 'NVIDIA_DRIVER_CAPABILITIES: "graphics,utility"' in profile
    assert "export NVIDIA_DRIVER_CAPABILITIES=graphics,utility" in WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert "B200" not in WORKFLOW.read_text(encoding="utf-8")
    assert "B200" not in profile
    assert "/opt/npa/gymnasium-robotics/verify_image.py" in profile
    assert "/opt/npa/gymnasium-robotics/runtime-bootstrap.py" in profile
    assert "NPA_GYMNASIUM_RUNTIME_CACHE" in profile
    assert "current/runtime/bin/python" not in profile
    assert "${NPA_GYMNASIUM_RUNTIME_CACHE}/current" not in profile
    assert "RUNTIME_PYTHON" not in profile
    assert "/usr/bin/python3 -I -B" in profile
    assert "/usr/local/bin/npa-gymnasium-entrypoint prepare-runtime" in profile
    assert '/bin/bash -lc "${BYOF_SMOKE_COMMAND}"' not in profile
    assert '["/usr/local/bin/npa-gymnasium-entrypoint", "run-smoke"]' in profile
    assert "-u AWS_SECRET_ACCESS_KEY" not in profile
    assert "env=runtime_environment" in profile
    assert "npa_pod_image_receipt.json" in profile
    assert 'IfNoneMatch="*"' in profile
    assert (
        "root_fd, root_descriptors, root_bindings = _open_bound_output_root(root)"
        in profile
    )
    assert "artifact_payload = _read_bound_json(" in profile
    assert "summary_payload = _write_bound_summary(" in profile
    assert "upload_payloads = _output_uploads(" in profile
    assert ".read_bytes()" not in profile
    assert ".write_text(" not in profile
    assert 'ContentType="application/json"' in profile
    assert 'Metadata={"sha256": digest}' in profile
    assert 'get_paginator("list_objects_v2")' in profile
    assert "observed_keys != expected_keys" in profile
    assert "root.rglob" not in profile


def test_profile_runtime_environment_is_the_exact_non_secret_allowlist() -> None:
    coordinator = _coordinator_source()
    syntax = ast.parse(coordinator)
    assignments = {
        target.id: node.value
        for node in syntax.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    allowlist = ast.literal_eval(assignments["runtime_environment_allowlist"])
    bootstrap_syntax = ast.parse(
        (IMAGE_ROOT / "runtime-bootstrap.py").read_text(encoding="utf-8")
    )
    bootstrap_assignments = {
        target.id: node.value
        for node in bootstrap_syntax.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    bootstrap_allowlist = ast.literal_eval(
        bootstrap_assignments["RUNTIME_ENVIRONMENT_ALLOWLIST"].args[0]
    )
    assert allowlist == {
        "BYOF_IMAGE",
        "MUJOCO_GL",
        "NPA_BYOF_POD_IMAGE_ID",
        "NPA_GYMNASIUM_RUNTIME_CACHE",
        "NPA_SMOKE_OUTPUT_DIR",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PYOPENGL_PLATFORM",
    }
    assert bootstrap_allowlist == allowlist
    authority = {
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "NEBIUS_IAM_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NGC_API_KEY",
        "DOCKER_AUTH_CONFIG",
        "KUBERNETES_SERVICE_HOST",
        "SSH_AUTH_SOCK",
        "HTTPS_PROXY",
        "NPA_AGENT_DATASET_URI",
    }
    assert not authority.intersection(allowlist)
    assert 'runtime_environment["PATH"]' in coordinator


def test_coordinator_binds_output_root_nofollow_and_close_on_exec(
    tmp_path: Path,
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    root_link = tmp_path / "output-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(
        SystemExit, match="cannot bind Gymnasium output directory safely"
    ):
        helpers["_open_bound_output_root"](root_link)

    root_fd, descriptors, bindings = _bound_output_root(helpers, root)
    try:
        assert not os.get_inheritable(root_fd)
        helpers["_require_bound_output_chain"](bindings)
    finally:
        _close_output_root(helpers, descriptors)


def test_coordinator_creates_final_output_below_bound_parent(tmp_path: Path) -> None:
    helpers = _coordinator_helpers(tmp_path)
    parent = tmp_path / "byof-runs"
    parent.mkdir()
    root = parent / "fresh-run"

    root_fd, descriptors, bindings = _bound_output_root(helpers, root)
    try:
        assert root.is_dir()
        assert os.fstat(root_fd).st_ino == root.stat().st_ino
        helpers["_require_bound_output_chain"](bindings)
    finally:
        _close_output_root(helpers, descriptors)


@pytest.mark.parametrize("attack", ["symlink", "regular"])
def test_coordinator_refuses_unsafe_intermediate_output_component(
    tmp_path: Path, attack: str
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    component = parent / "byof-runs"
    if attack == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        component.symlink_to(target, target_is_directory=True)
    else:
        component.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(
        SystemExit, match="cannot bind Gymnasium output directory safely"
    ):
        helpers["_open_bound_output_root"](component / "run")


def test_coordinator_refuses_same_uid_parent_replacement(tmp_path: Path) -> None:
    helpers = _coordinator_helpers(tmp_path)
    parent = tmp_path / "parent"
    root = parent / "byof-runs" / "run"
    root.mkdir(parents=True)
    _root_fd, descriptors, bindings = _bound_output_root(helpers, root)
    bound_parent = tmp_path / "bound-parent"
    parent.rename(bound_parent)
    root.mkdir(parents=True)
    try:
        with pytest.raises(SystemExit, match="directory chain changed"):
            helpers["_require_bound_output_chain"](bindings)
    finally:
        _close_output_root(helpers, descriptors)


def test_coordinator_refuses_final_output_substitution(tmp_path: Path) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    _root_fd, descriptors, bindings = _bound_output_root(helpers, root)
    bound_root = tmp_path / "bound-output"
    root.rename(bound_root)
    root.mkdir()
    try:
        with pytest.raises(SystemExit, match="directory chain changed"):
            helpers["_require_bound_output_chain"](bindings)
    finally:
        _close_output_root(helpers, descriptors)


def test_coordinator_keeps_child_logs_on_verified_descriptors(tmp_path: Path) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    stdout_fd, stdout_identity = helpers["_open_bound_log"](
        root_fd,
        "solution_smoke_stdout.log",
        os.geteuid(),
        "Gymnasium smoke stdout",
    )
    stderr_fd, stderr_identity = helpers["_open_bound_log"](
        root_fd,
        "solution_smoke_stderr.log",
        os.geteuid(),
        "Gymnasium smoke stderr",
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print('stdout-bound'); print('stderr-bound', file=sys.stderr)",
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout_fd,
            stderr=stderr_fd,
            check=False,
        )
        assert completed.returncode == 0
        helpers["_verify_bound_log"](
            root_fd,
            "solution_smoke_stdout.log",
            stdout_fd,
            os.geteuid(),
            stdout_identity,
            "Gymnasium smoke stdout",
        )
        helpers["_verify_bound_log"](
            root_fd,
            "solution_smoke_stderr.log",
            stderr_fd,
            os.geteuid(),
            stderr_identity,
            "Gymnasium smoke stderr",
        )
    finally:
        os.close(stderr_fd)
        os.close(stdout_fd)
        _close_output_root(helpers, descriptors)

    assert (root / "solution_smoke_stdout.log").read_text() == "stdout-bound\n"
    assert (root / "solution_smoke_stderr.log").read_text() == "stderr-bound\n"


@pytest.mark.parametrize("attack", ["regular", "symlink", "directory"])
def test_coordinator_refuses_preexisting_smoke_log(
    tmp_path: Path, attack: str
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    target = tmp_path / "target.log"
    target.write_text("unchanged\n", encoding="utf-8")
    log = root / "solution_smoke_stdout.log"
    if attack == "regular":
        log.write_text("preexisting\n", encoding="utf-8")
    elif attack == "symlink":
        log.symlink_to(target)
    else:
        log.mkdir()

    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    try:
        with pytest.raises(
            SystemExit, match="cannot create Gymnasium smoke stdout exclusively"
        ):
            helpers["_open_bound_log"](
                root_fd,
                log.name,
                os.geteuid(),
                "Gymnasium smoke stdout",
            )
    finally:
        _close_output_root(helpers, descriptors)

    assert target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize("replacement", ["regular", "symlink"])
def test_coordinator_refuses_same_uid_smoke_log_path_substitution(
    tmp_path: Path, replacement: str
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    name = "solution_smoke_stdout.log"
    log_fd, identity = helpers["_open_bound_log"](
        root_fd, name, os.geteuid(), "Gymnasium smoke stdout"
    )
    bound_log = tmp_path / "bound.log"
    (root / name).rename(bound_log)
    target = tmp_path / "target.log"
    target.write_text("unchanged\n", encoding="utf-8")
    if replacement == "regular":
        (root / name).write_text("replacement\n", encoding="utf-8")
        expected = "path identity changed"
    else:
        (root / name).symlink_to(target)
        expected = "is not a regular file"

    try:
        with pytest.raises(SystemExit, match=expected):
            helpers["_verify_bound_log"](
                root_fd,
                name,
                log_fd,
                os.geteuid(),
                identity,
                "Gymnasium smoke stdout",
            )
    finally:
        os.close(log_fd)
        _close_output_root(helpers, descriptors)

    assert target.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.parametrize("mutation", ["mode", "hardlink"])
def test_coordinator_refuses_smoke_log_metadata_drift(
    tmp_path: Path, mutation: str
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    name = "solution_smoke_stdout.log"
    log_fd, identity = helpers["_open_bound_log"](
        root_fd, name, os.geteuid(), "Gymnasium smoke stdout"
    )
    if mutation == "mode":
        os.chmod(root / name, 0o640)
        expected = "unexpected mode"
    else:
        os.link(root / name, root / "extra-link.log")
        expected = "must have exactly one link"

    try:
        with pytest.raises(SystemExit, match=expected):
            helpers["_verify_bound_log"](
                root_fd,
                name,
                log_fd,
                os.geteuid(),
                identity,
                "Gymnasium smoke stdout",
            )
    finally:
        os.close(log_fd)
        _close_output_root(helpers, descriptors)


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_coordinator_refuses_linked_runtime_artifact(
    tmp_path: Path, attack: str
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    artifact = root / "gymnasium-robotics-smoke.json"
    if attack == "symlink":
        artifact.symlink_to(target)
        expected = "cannot open Gymnasium qualification artifact safely"
    else:
        os.link(target, artifact)
        expected = "must have exactly one link"

    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    try:
        with pytest.raises(SystemExit, match=expected):
            helpers["_read_bound_json"](
                root_fd,
                artifact.name,
                os.geteuid(),
                "Gymnasium qualification artifact",
            )
    finally:
        _close_output_root(helpers, descriptors)


@pytest.mark.parametrize("attack", ["symlink", "regular"])
def test_coordinator_refuses_precreated_summary(tmp_path: Path, attack: str) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    summary = root / "npa_byof_summary.json"
    if attack == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        summary.symlink_to(target)
    else:
        summary.write_text("{}", encoding="utf-8")

    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    try:
        with pytest.raises(
            SystemExit, match="cannot create Gymnasium summary exclusively"
        ):
            helpers["_write_bound_summary"](
                root_fd, summary.name, b"{}\n", os.geteuid()
            )
    finally:
        _close_output_root(helpers, descriptors)


def test_coordinator_refuses_upload_after_output_path_substitution(
    tmp_path: Path,
) -> None:
    helpers = _coordinator_helpers(tmp_path)
    root = tmp_path / "output"
    root.mkdir()
    root_fd, descriptors, _bindings = _bound_output_root(helpers, root)
    bound_root = tmp_path / "bound-output"
    root.rename(bound_root)
    root.mkdir()
    try:
        with pytest.raises(SystemExit, match="directory chain changed"):
            helpers["_require_bound_output_chain"](_bindings)
    finally:
        _close_output_root(helpers, descriptors)
    assert bound_root.is_dir()
    assert root.is_dir()


def test_readiness_is_bound_and_all_execution_evidence_is_blocked() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "unverified"
    assert readiness["prerequisites"]["source_image"]["status"] == "blocked"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"
    source_reason = readiness["prerequisites"]["source_image"]["reason"].lower()
    assert "reference build" in source_reason
    assert "complete product scan" in source_reason
    assert "no retained, pullable, accepted" in source_reason
    assert "historical" in source_reason


def test_no_gated_payload_or_consent_proxy_is_part_of_neutral_design() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            WORKFLOW,
            SMOKE,
            IMAGE_ROOT / "Dockerfile",
            IMAGE_ROOT / "runtime-bootstrap.py",
        )
    )
    lowered = text.lower()
    for forbidden in (
        "hf_token",
        "snapshot_download",
        "accept_terms",
        "privacy_consent",
    ):
        assert forbidden not in lowered
    assert "nvcr.io" not in lowered
    assert "nvidia/cuda" not in lowered


def test_live_gate_still_requires_both_explicit_environment_gates() -> None:
    source = (ROOT / "npa/tests/e2e/test_byof_onboarding_live_e2e.py").read_text(
        encoding="utf-8"
    )
    assert 'NPA_INTEGRATION_E2E") != "1"' in source
    assert 'NPA_BYOF_GYMNASIUM_ROBOTICS_LIVE_GPU") != "1"' in source
    assert "NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE" in source


def test_documentation_keeps_neutral_and_historical_evidence_separate() -> None:
    text = (ROOT / "docs/workbench/byof-gymnasium-robotics.md").read_text(
        encoding="utf-8"
    )
    for token in (
        "neutral bootstrap",
        "pre-registration quarantine",
        "historical",
        "corresponding-source",
        "No model, external dataset, gated artifact, or terms-acceptance flag",
        "RTX PRO 6000 Blackwell",
    ):
        assert token in text
