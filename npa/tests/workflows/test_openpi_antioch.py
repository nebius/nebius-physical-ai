from __future__ import annotations

import json
import inspect
from pathlib import Path
import subprocess

import pytest

from npa.workflows.byof import openpi_antioch as antioch
from npa.workflows.byof.openpi import OPENPI_TERMS_ENV
from npa.workflows.byof.openpi_pipeline import SOURCE_REF


def _completed(
    argv: list[str], stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode, stdout, "")


def _passed_payload() -> dict[str, object]:
    return {
        "outcome": "passed",
        "results": {
            "chunks_run": 3,
            "mean_inference_ms": 120.5,
            "max_joint_travel_rad": 0.8,
            "jaw_travel_mm": 84.0,
            "server": "private-host:8001",
            "checks": [
                {"criterion": criterion, "passed": True, "detail": "measured"}
                for criterion in sorted(antioch.REQUIRED_CHECKS)
            ],
        },
    }


def test_harness_has_no_firewall_or_managed_machine_mutation() -> None:
    source = inspect.getsource(antioch)

    assert "iptables" not in source
    assert "machine ssh" not in source
    assert "scp" not in source
    assert "rsync" not in source


def test_validate_scenario_evidence_is_sanitized() -> None:
    evidence = antioch.validate_scenario_evidence(_passed_payload())

    assert evidence["status"] == "passed"
    assert evidence["action_chunk_shape"] == [15, 8]
    assert evidence["containerized_policy"] is True
    assert "server" not in evidence
    assert "scenario_run_id" not in evidence
    assert "private-host" not in json.dumps(evidence)


def test_json_object_accepts_cli_progress_before_json() -> None:
    assert antioch._json_object(
        'Staging a real run\n{"scenario_run_id":"run-id"}\n', label="queue"
    ) == {"scenario_run_id": "run-id"}


def test_validate_scenario_evidence_requires_every_real_gate() -> None:
    payload = _passed_payload()
    results = payload["results"]
    assert isinstance(results, dict)
    checks = results["checks"]
    assert isinstance(checks, list)
    checks.pop()

    with pytest.raises(antioch.OpenPIAntiochError, match="missing passing checks"):
        antioch.validate_scenario_evidence(payload)


def test_validate_scenario_evidence_rejects_nonfinite_metrics() -> None:
    payload = _passed_payload()
    results = payload["results"]
    assert isinstance(results, dict)
    results["mean_inference_ms"] = float("nan")

    with pytest.raises(antioch.OpenPIAntiochError, match="mean_inference_ms"):
        antioch.validate_scenario_evidence(payload)


def test_validate_direct_run_output_requires_measured_chunks() -> None:
    output = """\
jaw travel: 84.7 mm
chunk 0: 120.0 ms  shape=(15, 8)  arm|a|max=0.1
chunk 1: 130.0 ms  shape=(15, 8)  arm|a|max=0.1
chunk 2: 140.0 ms  shape=(15, 8)  arm|a|max=0.1
mean inference latency: 130.0 ms over 3 chunks
max joint travel: 1.7667 rad
ALL GATES PASSED — stock Isaac on Antioch, policy server off-box
"""

    evidence = antioch.validate_run_output(output, expected_chunks=3)

    assert evidence["execution"] == "antioch_run"
    assert evidence["chunks_run"] == 3
    assert evidence["jaw_travel_mm"] == 84.7


def test_validate_direct_run_output_rejects_connectivity_only() -> None:
    with pytest.raises(antioch.OpenPIAntiochError, match="all-gates verdict"):
        antioch.validate_run_output("connected to policy server\n", expected_chunks=3)


def test_build_refuses_unpinned_source_before_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(list(argv), "not-the-pinned-ref\n")

    monkeypatch.setattr(antioch, "_run", fake_run)
    with pytest.raises(antioch.OpenPIAntiochError, match="must be pinned"):
        antioch.build_local_image(openpi_dir=tmp_path, image="local/openpi:test")

    assert calls == [["git", "rev-parse", "HEAD"]]


def test_build_refuses_dirty_source_before_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return _completed(list(argv), SOURCE_REF + "\n")
        return _completed(list(argv), " M scripts/serve_policy.py\n")

    monkeypatch.setattr(antioch, "_run", fake_run)

    with pytest.raises(antioch.OpenPIAntiochError, match="must be clean"):
        antioch.build_local_image(openpi_dir=tmp_path, image="local/openpi:test")

    assert calls == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    ]


def test_build_uses_stdin_dockerfile_without_acceptance_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return _completed(list(argv), SOURCE_REF + "\n")
        if argv[:3] == ["docker", "image", "inspect"]:
            return _completed(list(argv), "sha256:" + "a" * 64 + "\n")
        return _completed(list(argv))

    monkeypatch.setattr(antioch, "_run", fake_run)
    antioch.build_local_image(openpi_dir=tmp_path, image="local/openpi:test")

    build_argv, build_kwargs = calls[2]
    assert build_argv[-2:] == ["-", str(tmp_path)]
    assert f"ENV {OPENPI_TERMS_ENV}=YES" not in str(build_kwargs["input_text"])
    assert "--build-arg" not in build_argv
    assert "NPA_OPENPI_TERMS_REFUSED" in str(build_kwargs["input_text"])
    assert "pi05_droid" not in str(build_kwargs["input_text"])


def test_negative_probe_strips_parent_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    observed_env: dict[str, str] = {}

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        observed_env.update(kwargs["env"])
        return subprocess.CompletedProcess(
            list(argv), 64, "", "NPA_OPENPI_TERMS_REFUSED\n"
        )

    monkeypatch.setattr(antioch, "_run", fake_run)
    antioch._negative_terms_probe(
        antioch.LiveLoopConfig(
            project_dir=tmp_path,
            cache_dir=tmp_path,
            image="local/openpi:test",
            policy_host="policy-host.example",
        )
    )

    assert OPENPI_TERMS_ENV not in observed_env
    assert "--entrypoint" not in calls[0]


def test_policy_wait_times_out_when_container_stays_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        policy_ready_timeout_s=1,
    )
    monotonic_values = iter([0.0, 1.1])

    def unavailable_socket(*_args, **_kwargs):
        raise OSError("not ready")

    monkeypatch.setattr(antioch.socket, "create_connection", unavailable_socket)
    monkeypatch.setattr(antioch.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(antioch.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        antioch,
        "_run",
        lambda argv, **_kwargs: _completed(list(argv), "true 0\n"),
    )

    with pytest.raises(antioch.OpenPIAntiochError, match="did not become ready"):
        antioch._wait_for_policy(config, container_name="test-policy")


def test_scenario_wait_times_out_when_result_never_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        scenario_timeout_s=1,
    )
    monotonic_values = iter([0.0, 1.1])
    monkeypatch.setattr(antioch.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(antioch.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        antioch,
        "_run",
        lambda argv, **_kwargs: _completed(list(argv), '{"phase":"running"}\n'),
    )

    with pytest.raises(antioch.OpenPIAntiochError, match="did not complete"):
        antioch._wait_for_scenario(config, "scenario-run")


def test_cleanup_failure_does_not_mask_primary_live_loop_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        cleanup_container=True,
        cleanup_scenario=True,
        resource_owner="task-owner",
    )

    monkeypatch.setattr(antioch, "_negative_terms_probe", lambda _config: None)
    monkeypatch.setattr(
        antioch, "_image_identity", lambda _config: "sha256:" + "a" * 64
    )
    monkeypatch.setattr(
        antioch,
        "_ensure_policy_container",
        lambda _config: antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="task-owner",
            running=True,
        ),
    )
    monkeypatch.setattr(antioch, "_wait_for_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(antioch, "_submit_scenario", lambda _config: "submitted-run")
    monkeypatch.setattr(
        antioch,
        "_wait_for_scenario",
        lambda *_args: (_ for _ in ()).throw(
            antioch.OpenPIAntiochError("primary scenario failure")
        ),
    )
    monkeypatch.setattr(
        antioch,
        "_cleanup_live_resources",
        lambda *_args: [antioch.OpenPIAntiochError("cleanup failure")],
    )
    monkeypatch.setattr(antioch, "_run", lambda argv, **_kwargs: _completed(list(argv)))

    with pytest.raises(antioch.OpenPIAntiochError, match="primary scenario failure"):
        antioch.run_live_loop(config)


def test_live_loop_parser_has_bounded_wait_defaults_and_overrides(
    tmp_path: Path,
) -> None:
    parser = antioch.build_parser()
    required = [
        "live-loop",
        "--project-dir",
        str(tmp_path / "project"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--image",
        "local/openpi:test",
        "--policy-host",
        "policy-host.example",
    ]

    defaults = parser.parse_args(required)
    overrides = parser.parse_args(
        required
        + [
            "--policy-ready-timeout-s",
            "17.5",
            "--scenario-timeout-s",
            "180.5",
        ]
    )

    assert defaults.policy_ready_timeout_s == 300.0
    assert defaults.scenario_timeout_s == 1800.0
    assert overrides.policy_ready_timeout_s == 17.5
    assert overrides.scenario_timeout_s == 180.5


def test_main_emits_sanitized_structured_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        antioch,
        "build_local_image",
        lambda **_kwargs: (_ for _ in ()).throw(
            antioch.OpenPIAntiochError("private runtime detail")
        ),
    )

    status = antioch.main(
        [
            "build-image",
            "--openpi-dir",
            str(tmp_path / "openpi"),
            "--image",
            "local/openpi:test",
        ]
    )
    result = json.loads(capsys.readouterr().err)

    assert status == 1
    assert result == {
        "error_type": "OpenPIAntiochError",
        "message": "OpenPI/Antioch operation failed; inspect private local logs.",
        "status": "failed",
    }


def test_accepted_container_forwards_env_by_name_only(tmp_path: Path) -> None:
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
    )

    argv = antioch._accepted_container_argv(config, container_name="exact-container")

    env_index = argv.index("--env")
    assert argv[env_index + 1] == OPENPI_TERMS_ENV
    assert f"{OPENPI_TERMS_ENV}=YES" not in argv
    assert "exact-container" in argv
    assert "--gpus" in argv
    assert argv[argv.index("--restart") + 1] == "unless-stopped"
    assert antioch.MANAGED_CONTAINER_LABEL in argv
    assert "readonly" in argv[argv.index("--mount") + 1]
    assert "--health-cmd" in argv
    assert argv[-4:] == ["--env", "DROID", "--port", "8000"]


def test_reuses_only_matching_managed_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        running = "true" if argv[1:3] == ["inspect", "container-sha"] else "false"
        return _completed(
            list(argv), f"true\t\tlocal/openpi:test\tcontainer-sha\t{running}\n"
        )

    monkeypatch.setattr(antioch, "_run", fake_run)
    created = antioch._ensure_policy_container(
        antioch.LiveLoopConfig(
            project_dir=tmp_path,
            cache_dir=tmp_path,
            image="local/openpi:test",
            policy_host="policy-host.example",
        )
    )

    assert created.container_id == "container-sha"
    assert created.running is True
    assert ["docker", "start", antioch.DEFAULT_CONTAINER_NAME] in calls


def test_cleanup_refuses_changed_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(
            list(argv), "true\tother-task\tlocal/openpi:test\tcontainer-sha\ttrue\n"
        )

    monkeypatch.setattr(
        antioch,
        "_run",
        fake_run,
    )
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        resource_owner="task-owner",
    )
    with pytest.raises(antioch.OpenPIAntiochError, match="changed ownership"):
        antioch._remove_policy_container(
            config,
            antioch._ContainerIdentity(
                name="exact-container",
                container_id="container-sha",
                image="local/openpi:test",
                managed=True,
                owner="task-owner",
                running=True,
            ),
        )
    assert all("rm" not in call for call in calls)


def test_container_cleanup_removes_exact_id_then_verifies_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    inspect_count = 0

    def fake_run(argv, **_kwargs):
        nonlocal inspect_count
        command = list(argv)
        calls.append(command)
        if command[1] == "inspect":
            inspect_count += 1
            if inspect_count == 2:
                return _completed(command, "", returncode=1)
            return _completed(
                command,
                "true\ttask-owner\tlocal/openpi:test\tcontainer-sha\ttrue\n",
            )
        return _completed(command)

    monkeypatch.setattr(antioch, "_run", fake_run)
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        resource_owner="task-owner",
    )

    antioch._remove_policy_container(
        config,
        antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="task-owner",
            running=True,
        ),
    )

    assert calls == [
        [
            "docker",
            "inspect",
            "container-sha",
            "--format",
            (
                '{{index .Config.Labels "npa.openpi-antioch.managed"}}\t'
                '{{index .Config.Labels "npa.openpi-antioch.owner"}}\t'
                "{{.Config.Image}}\t{{.Id}}\t{{.State.Running}}"
            ),
        ],
        ["docker", "rm", "--force", "container-sha"],
        [
            "docker",
            "inspect",
            "container-sha",
            "--format",
            (
                '{{index .Config.Labels "npa.openpi-antioch.managed"}}\t'
                '{{index .Config.Labels "npa.openpi-antioch.owner"}}\t'
                "{{.Config.Image}}\t{{.Id}}\t{{.State.Running}}"
            ),
        ],
    ]


def test_live_loop_gates_before_any_external_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(OPENPI_TERMS_ENV, raising=False)
    monkeypatch.setattr(
        antioch,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("external command ran before terms gate"),
    )

    with pytest.raises(ValueError, match="scoped operator acceptance"):
        antioch.run_live_loop(
            antioch.LiveLoopConfig(
                project_dir=tmp_path,
                cache_dir=tmp_path,
                image="local/openpi:test",
                policy_host="policy-host.example",
            )
        )


@pytest.mark.parametrize("chunks", [0, -1])
def test_live_loop_rejects_nonpositive_chunks_before_external_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunks: int
) -> None:
    monkeypatch.setattr(
        antioch,
        "_run",
        lambda *_args, **_kwargs: pytest.fail("external command ran before chunk gate"),
    )

    with pytest.raises(antioch.OpenPIAntiochError, match="positive integer"):
        antioch.run_live_loop(
            antioch.LiveLoopConfig(
                project_dir=tmp_path,
                cache_dir=tmp_path,
                image="local/openpi:test",
                policy_host="policy-host.example",
                script="direct-loop.py",
                chunks=chunks,
            )
        )


@pytest.mark.parametrize("chunks", [0, -1])
def test_validate_direct_run_output_rejects_nonpositive_chunks(chunks: int) -> None:
    with pytest.raises(antioch.OpenPIAntiochError, match="positive integer"):
        antioch.validate_run_output("ALL GATES PASSED\n", expected_chunks=chunks)


def test_scenario_submission_binds_the_authoritative_response_not_a_newer_list_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    calls: list[list[str]] = []
    waited_for: list[str] = []
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
    )

    def fake_run(argv, **_kwargs):
        command = list(argv)
        calls.append(command)
        if command[:3] == ["antioch", "scenario", "run"]:
            return _completed(command, '[{"scenario_run_id":"submitted-run"}]\n')
        if command[:3] == ["antioch", "scenario", "list"]:
            pytest.fail("a concurrent newer scenario must never be selected")
        return _completed(command)

    monkeypatch.setattr(antioch, "_run", fake_run)
    monkeypatch.setattr(antioch, "_negative_terms_probe", lambda _config: None)
    monkeypatch.setattr(
        antioch, "_image_identity", lambda _config: "sha256:" + "a" * 64
    )
    monkeypatch.setattr(
        antioch,
        "_ensure_policy_container",
        lambda _config: antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="",
            running=True,
        ),
    )
    monkeypatch.setattr(antioch, "_wait_for_policy", lambda *_args, **_kwargs: None)

    def exact_wait(_config, run_id: str):
        waited_for.append(run_id)
        return _passed_payload()

    monkeypatch.setattr(antioch, "_wait_for_scenario", exact_wait)

    result = antioch.run_live_loop(config)

    submission = next(
        call for call in calls if call[:3] == ["antioch", "scenario", "run"]
    )
    assert submission[-2:] == ["--detach", "--json"]
    assert waited_for == ["submitted-run"]
    assert result["status"] == "passed"
    assert not any(call[:3] == ["antioch", "scenario", "list"] for call in calls)


def test_cancel_active_scenario_cancels_the_exact_id_and_verifies_terminal_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    show_count = 0
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
    )

    def fake_run(argv, **_kwargs):
        nonlocal show_count
        command = list(argv)
        calls.append(command)
        if command[:3] == ["antioch", "scenario", "show"]:
            show_count += 1
            phase = "running" if show_count == 1 else "cancelled"
            return _completed(command, json.dumps({"phase": phase}) + "\n")
        return _completed(command, "{}\n")

    monkeypatch.setattr(antioch, "_run", fake_run)
    state, payload = antioch._cancel_active_scenario(config, "submitted-run")

    assert state == "terminal_verified"
    assert payload["phase"] == "cancelled"
    assert calls == [
        ["antioch", "scenario", "show", "submitted-run", "--json"],
        ["antioch", "scenario", "cancel", "submitted-run", "--json"],
        ["antioch", "scenario", "show", "submitted-run", "--json"],
    ]


def test_task_cleanup_cancels_then_removes_only_the_exact_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operations: list[tuple[str, str]] = []
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        cleanup_scenario=True,
        cleanup_container=True,
        resource_owner="task-owner",
    )
    resources = antioch._LiveResources(
        scenario_run_id="submitted-run",
        container=antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="task-owner",
            running=True,
        ),
    )
    monkeypatch.setattr(
        antioch,
        "_cancel_active_scenario",
        lambda _config, run_id: (
            operations.append(("cancel", run_id)) or "terminal_verified",
            {"phase": "cancelled", "outcome": "cancelled"},
        ),
    )
    monkeypatch.setattr(
        antioch,
        "_remove_policy_container",
        lambda _config, container: operations.append(
            ("remove", container.container_id)
        ),
    )

    assert antioch._cleanup_live_resources(config, resources) == []
    assert operations == [("cancel", "submitted-run"), ("remove", "container-sha")]
    assert resources.scenario_cleanup == "terminal_verified"
    assert resources.container_cleanup == "absent_verified"
    assert resources.container_absent is True


def test_cleanup_error_fails_an_otherwise_successful_live_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        cleanup_scenario=True,
        resource_owner="task-owner",
    )
    monkeypatch.setattr(antioch, "_negative_terms_probe", lambda _config: None)
    monkeypatch.setattr(
        antioch, "_image_identity", lambda _config: "sha256:" + "a" * 64
    )
    monkeypatch.setattr(
        antioch,
        "_ensure_policy_container",
        lambda _config: antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="task-owner",
            running=True,
        ),
    )
    monkeypatch.setattr(antioch, "_wait_for_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(antioch, "_submit_scenario", lambda _config: "submitted-run")
    monkeypatch.setattr(antioch, "_wait_for_scenario", lambda *_args: _passed_payload())
    monkeypatch.setattr(
        antioch,
        "_cleanup_live_resources",
        lambda *_args: [antioch.OpenPIAntiochError("cleanup failure")],
    )
    monkeypatch.setattr(antioch, "_run", lambda argv, **_kwargs: _completed(list(argv)))

    with pytest.raises(antioch.OpenPIAntiochError, match="cleanup failure"):
        antioch.run_live_loop(config)


def test_private_receipt_binds_exact_resources_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = tmp_path / "private-receipt.json"
    config = antioch.LiveLoopConfig(
        project_dir=tmp_path,
        cache_dir=tmp_path,
        image="local/openpi:test",
        policy_host="policy-host.example",
        resource_owner="task-owner",
        private_receipt_path=receipt,
    )
    resources = antioch._LiveResources(
        scenario_run_id="submitted-run",
        image_id="sha256:" + "a" * 64,
        evidence=antioch.validate_scenario_evidence(_passed_payload()),
        scenario_cleanup="terminal_verified",
        scenario_phase="cancelled",
        scenario_outcome="cancelled",
        container_cleanup="absent_verified",
        container_absent=True,
        container=antioch._ContainerIdentity(
            name="exact-container",
            container_id="container-sha",
            image="local/openpi:test",
            managed=True,
            owner="task-owner",
            running=True,
        ),
    )
    monkeypatch.setattr(antioch, "_harness_source_sha", lambda: "b" * 40)

    antioch._write_private_receipt(config, resources, status="passed")

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["harness_source_sha"] == "b" * 40
    assert payload["image_id"] == "sha256:" + "a" * 64
    assert payload["scenario_run_id"] == "submitted-run"
    assert payload["container"]["id"] == "container-sha"
    assert payload["acceptance"] == {
        "action_chunk_shape": [15, 8],
        "chunks_run": 3,
        "jaw_travel_mm": 84.0,
        "max_joint_travel_rad": 0.8,
        "mean_inference_ms": 120.5,
        "passing_checks": sorted(antioch.REQUIRED_CHECKS),
    }
    assert payload["scenario_cleanup"]["phase"] == "cancelled"
    assert payload["container_cleanup"]["absent_verified"] is True
    assert receipt.stat().st_mode & 0o777 == 0o600
