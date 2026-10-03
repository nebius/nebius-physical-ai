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

    build_argv, build_kwargs = calls[1]
    assert build_argv[-2:] == ["-", str(tmp_path)]
    assert OPENPI_TERMS_ENV not in str(build_kwargs["input_text"])
    assert "pi05_droid" not in str(build_kwargs["input_text"])


def test_negative_probe_strips_parent_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENPI_TERMS_ENV, "YES")
    observed_env: dict[str, str] = {}

    def fake_run(argv, **kwargs):
        observed_env.update(kwargs["env"])
        return _completed(list(argv), "NPA_OPENPI_TERMS_REFUSED\n", 64)

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
        return _completed(list(argv), "true local/openpi:test false\n")

    monkeypatch.setattr(antioch, "_run", fake_run)
    created = antioch._ensure_policy_container(
        antioch.LiveLoopConfig(
            project_dir=tmp_path,
            cache_dir=tmp_path,
            image="local/openpi:test",
            policy_host="policy-host.example",
        )
    )

    assert created is False
    assert calls[-1] == ["docker", "start", antioch.DEFAULT_CONTAINER_NAME]


def test_cleanup_refuses_unlabelled_name_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        antioch,
        "_run",
        lambda argv, **_kwargs: _completed(list(argv), "\n"),
    )
    with pytest.raises(antioch.OpenPIAntiochError, match="unlabelled"):
        antioch._remove_policy_container(
            antioch.LiveLoopConfig(
                project_dir=tmp_path,
                cache_dir=tmp_path,
                image="local/openpi:test",
                policy_host="policy-host.example",
            )
        )


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
