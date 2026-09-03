from __future__ import annotations

import base64
import io
import json
import os
import socket
import stat
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from npa.workbench.antioch import cluster_deploy, cluster_runtime
from npa.workbench.antioch.health import StateHealthServer
from npa.workbench.antioch.vendor_cli import AntiochCliError


def _config(tmp_path: Path, **updates: object) -> cluster_deploy.ClusterLiveConfig:
    config_dir = tmp_path / "antioch-config"
    config_dir.mkdir(mode=0o700)
    (config_dir / "config.json").write_text("{}", encoding="utf-8")
    os.chmod(config_dir / "config.json", 0o600)
    project = tmp_path / "project-id"
    project.write_text("private-project-id\n", encoding="utf-8")
    os.chmod(project, 0o600)
    values: dict[str, object] = {
        "workflow_run": "workflow-a",
        "state_id": "antioch-live-a",
        "kubeconfig": str(tmp_path / "kubeconfig"),
        "namespace": "workbench",
        "adapter_image": "registry.invalid/npa-antioch@sha256:" + "a" * 64,
        "policy_selector": {"app": "openpi-policy"},
        "policy_network_policy_name": "openpi-policy",
        "policy_auth_secret_name": "openpi-auth",
        "policy_tls_secret_name": "openpi-tls",
        "policy_cache_pvc_name": "openpi-cache",
        "antioch_config_dir": str(config_dir),
        "antioch_project_id_file": str(project),
        "kubelet_source_cidrs": ["192.0.2.10/32"],
    }
    values.update(updates)
    return cluster_deploy.ClusterLiveConfig.model_validate(values)


def test_private_config_requires_mode_0600_and_per_state_identity(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path)
    second = first.model_copy(update={"state_id": "antioch-live-b"})
    assert first.identity != second.identity
    path = tmp_path / "runtime.json"
    path.write_text(first.model_dump_json(), encoding="utf-8")
    os.chmod(path, 0o644)
    with pytest.raises(cluster_deploy.ClusterLiveError, match="0600"):
        cluster_deploy.load_private_config(path)
    os.chmod(path, 0o600)
    assert cluster_deploy.load_private_config(path) == first


def test_cluster_live_requires_digest_pinned_adapter(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="sha256"):
        _config(tmp_path, adapter_image="registry.invalid/npa-antioch:latest")


def _accepted_live_metrics() -> dict[str, int | float]:
    return {
        "elapsed_seconds": 930.0,
        "frames": 120,
        "requests": 100,
        "round_trips": 100,
        "applied": 500,
        "action_horizon": 15,
        "action_dimension": 8,
        "action_finite": 1,
        "rejected_wrong_shape": 0,
        "rejected_non_finite": 0,
        "rejected_joint_limit": 0,
        "rejected_gripper_range": 0,
        "rejected_joint_step": 0,
        "camera_quality_schema": 3,
        "camera_validated_requests": 100,
        "camera_pair_id": 100,
        "request_camera_pair_id": 100,
        "round_trip_camera_pair_id": 100,
        "camera_render_sequence": 1200,
        "request_render_sequence": 1199,
        "round_trip_render_sequence": 1199,
        "camera_pair_difference_current": 42.0,
        "camera_exterior_red_cube_pixels_current": 300,
        "camera_exterior_cube_in_frame_current": 1,
        "camera_wrist_cube_in_frame_current": 1,
        "camera_exterior_luminance_mean_current": 40.0,
        "camera_exterior_luminance_variance_current": 100.0,
        "camera_wrist_luminance_mean_current": 35.0,
        "camera_wrist_luminance_variance_current": 90.0,
        "luminance_mean_min": 30.0,
        "luminance_variance_min": 80.0,
        "joint_limit_projections": 0,
        "joint_step_projections": 0,
        "end_effector_cube_approach_m": 0.2,
        "end_effector_cube_distance_m": 0.04,
        "gripper_contact_samples": 20,
        "gripper_contact_force_max_n": 1.5,
        "cube_lift_max_m": 0.051,
        "pickup_hold_seconds": 1.0,
        "pickup_success": 1,
        "latency_p95_ms": 100.0,
        "latency_p99_ms": 120.0,
        "latency_max_ms": 130.0,
        "reconnects": 0,
    }


def test_live_acceptance_requires_current_pair_identity_and_physical_pickup() -> None:
    accepted = cluster_deploy.qualify_live_metrics(_accepted_live_metrics())
    assert accepted["accepted"] is True
    assert accepted["failures"] == []

    for changed, expected_failure in (
        ({"camera_validated_requests": 99}, "camera_pair_identity"),
        ({"round_trip_camera_pair_id": 99}, "camera_pair_identity"),
        ({"round_trip_render_sequence": 1198}, "camera_pair_identity"),
        ({"camera_wrist_luminance_variance_current": 0}, "current_camera_quality"),
        ({"luminance_mean_min": 0}, "accepted_camera_quality"),
        ({"camera_pair_difference_current": 7.9}, "accepted_camera_quality"),
        ({"camera_exterior_red_cube_pixels_current": 19}, "accepted_camera_quality"),
        ({"camera_wrist_cube_in_frame_current": 0}, "accepted_camera_quality"),
        ({"gripper_contact_samples": 0}, "physical_gripper_contact"),
        ({"cube_lift_max_m": 0.049}, "sustained_pickup"),
        ({"pickup_hold_seconds": 0.999}, "sustained_pickup"),
        ({"pickup_success": 0}, "sustained_pickup"),
    ):
        metrics = _accepted_live_metrics()
        metrics.update(changed)
        rejected = cluster_deploy.qualify_live_metrics(metrics)
        assert rejected["accepted"] is False
        assert expected_failure in rejected["failures"]


def test_cluster_live_terms_acceptance_is_process_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.delenv("NPA_ANTIOCH_ACCEPT_TERMS", raising=False)
    with pytest.raises(cluster_deploy.ClusterLiveError, match="exact required value"):
        cluster_deploy._terms_acceptance()
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "yes")
    with pytest.raises(cluster_deploy.ClusterLiveError, match="exact required value"):
        cluster_deploy._terms_acceptance()
    monkeypatch.setenv("NPA_ANTIOCH_ACCEPT_TERMS", "YES")
    assert cluster_deploy._terms_acceptance() == b"YES"
    assert "antioch_terms_file" not in type(config).model_fields


def test_config_archive_preserves_owner_only_nested_assignment_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    root = Path(config.antioch_config_dir)
    ssh = root / "ssh"
    ssh.mkdir(mode=0o700)
    private_key = ssh / "assigned-machine"
    private_key.write_bytes(b"private-runtime-state")
    os.chmod(private_key, 0o600)
    lock = root / "session.lock"
    lock.touch(mode=0o600)
    os.chmod(lock, 0o600)

    first = cluster_deploy._config_archive(root)["config.tar"]
    second = cluster_deploy._config_archive(root)["config.tar"]
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r") as archive:
        members = {member.name: member for member in archive.getmembers()}
        assert set(members) == {
            "config.json",
            "session.lock",
            "ssh",
            "ssh/assigned-machine",
        }
        assert members["ssh"].isdir()
        assert members["ssh"].mode == 0o700
        assert members["ssh/assigned-machine"].isfile()
        assert members["ssh/assigned-machine"].mode == 0o600
        extracted = archive.extractfile(members["ssh/assigned-machine"])
        assert extracted is not None
        assert extracted.read() == b"private-runtime-state"


def test_config_archive_rejects_non_owner_only_nested_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    nested = Path(config.antioch_config_dir) / "ssh"
    nested.mkdir(mode=0o755)
    os.chmod(nested, 0o755)
    with pytest.raises(cluster_deploy.ClusterLiveError, match="owner-only"):
        cluster_deploy._config_archive(Path(config.antioch_config_dir))


def test_public_manifests_keep_vm_out_and_policy_cluster_local(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifests = cluster_deploy.build_public_manifests(config)
    service = manifests["policy_service"]
    assert service["spec"]["type"] == "ClusterIP"
    assert "externalIPs" not in service["spec"]
    pod = manifests["adapter_deployment"]["spec"]["template"]["spec"]
    assert {item["name"] for item in pod["containers"]} == {
        "antioch-controller",
        "policy-relay",
    }
    assert pod["automountServiceAccountToken"] is False
    assert pod["terminationGracePeriodSeconds"] >= 1_100
    init = pod["initContainers"][0]
    init_command = init["command"][-1]
    volume_names = [volume["name"] for volume in pod["volumes"]]
    assert len(volume_names) == len(set(volume_names))
    init_mount_names = [mount["name"] for mount in init["volumeMounts"]]
    assert len(init_mount_names) == len(set(init_mount_names))
    assert "tar --extract --file /sources/config/config.tar" in init_command
    assert "cp -L /sources/bundle/*" in init_command
    assert init["securityContext"]["capabilities"] == {
        "drop": ["ALL"],
        "add": ["CHOWN"],
    }
    init_mounts = {mount["name"]: mount["mountPath"] for mount in init["volumeMounts"]}
    assert init_mounts["state"] == "/state"
    assert init_mounts["runtime"] == "/runtime"
    assert init_mounts["runtime-cache"] == "/runtime-cache"
    assert (
        "chown -R 10001:10001 /private /state /runtime /runtime-cache" in init_command
    )
    assert "cp -a" not in init_command
    controller, relay = pod["containers"]
    controller_mounts = {mount["name"]: mount for mount in controller["volumeMounts"]}
    relay_mounts = {mount["name"]: mount for mount in relay["volumeMounts"]}
    assert controller_mounts["private"]["readOnly"] is False
    assert relay_mounts["private"]["readOnly"] is True
    assert "cluster_runtime" in " ".join(controller["command"])
    assert "14400" in controller["command"]
    assert "openpi_franka_mk8s_live_v2" in controller["command"]
    assert "antioch.relay" in " ".join(relay["command"])
    assert "18444" in relay["command"]
    assert controller["readinessProbe"]["httpGet"] == {
        "path": "/ready",
        "port": "ctrl-health",
    }
    assert controller["livenessProbe"]["httpGet"] == {
        "path": "/live",
        "port": "ctrl-health",
    }
    assert relay["readinessProbe"]["httpGet"] == {
        "path": "/ready",
        "port": "relay-health",
    }
    assert relay["livenessProbe"]["httpGet"] == {
        "path": "/live",
        "port": "relay-health",
    }
    assert controller["ports"] == [{"name": "ctrl-health", "containerPort": 18080}]
    for container in (controller, relay):
        assert container["readinessProbe"]["exec"] is None
        assert container["livenessProbe"]["exec"] is None
    assert "--resume-after-stop" in relay["command"]
    assert controller["readinessProbe"]["failureThreshold"] == 3
    assert relay["readinessProbe"]["failureThreshold"] == 3
    assert "--owner-identity" in controller["command"]
    assert "--owner-identity" in relay["command"]
    rendered = json.dumps(manifests, sort_keys=True)
    assert "LoadBalancer" not in rendered
    assert "hostNetwork" not in rendered
    assert "private-project-id" not in rendered
    assert "api-key" not in rendered
    assert "CERT_NONE" not in rendered

    ingress = manifests["policy_network_policy"]["spec"]["ingress"]
    assert ingress[0]["from"] == [
        {"podSelector": {"matchLabels": cluster_deploy._labels(config)}}
    ]
    assert ingress[0]["ports"] == [{"protocol": "TCP", "port": 8443}]
    adapter_policy = manifests["adapter_network_policy"]["spec"]
    assert adapter_policy["ingress"] == [
        {
            "from": [{"ipBlock": {"cidr": "192.0.2.10/32"}}],
            "ports": [
                {"protocol": "TCP", "port": 18080},
                {"protocol": "TCP", "port": 18081},
            ],
        }
    ]
    assert adapter_policy["policyTypes"] == ["Ingress", "Egress"]
    assert adapter_policy["egress"][-1] == {
        "to": [
            {
                "ipBlock": {
                    "cidr": cluster_deploy.UNRESTRICTED_VENDOR_EGRESS_CIDR
                }
            }
        ],
        "ports": [
            {"protocol": "TCP", "port": 22},
            {"protocol": "TCP", "port": 443},
            {"protocol": "TCP", "port": 8443},
        ],
    }


def test_cluster_runtime_probe_is_fail_closed(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    kwargs = {
        "component": "controller",
        "expected_owner_identity": "owner",
        "max_age_seconds": 30.0,
    }
    assert cluster_runtime.probe(state, **kwargs) == 1
    state.write_text(json.dumps({"status": "starting"}), encoding="utf-8")
    assert cluster_runtime.probe(state, **kwargs) == 1
    healthy = {
        "schema_version": 3,
        "owner_identity": "owner",
        "session_id": "session",
        "scenario_run_id": "run",
        "status": "running",
        "daemon_status": "owned",
        "heartbeat_unix": time.time(),
        "vendor_process_status": "running",
        "daemon_guest_state": "healthy",
        "controller_pid": 1,
        "vendor_pid": 2,
        "vendor_parent_pid": 1,
        "vendor_process_group_isolated": True,
        "daemon_observed_at": time.time(),
        "rome_guest_observed_at": time.time(),
        "scenario_session_leases": 1,
        "process_leases": 1,
        "stream_leases": 1,
    }
    state.write_text(json.dumps(healthy), encoding="utf-8")
    assert cluster_runtime.probe(state, **kwargs) == 0
    healthy["heartbeat_unix"] = time.time() - 31
    state.write_text(json.dumps(healthy), encoding="utf-8")
    assert cluster_runtime.probe(state, **kwargs) == 1
    healthy["heartbeat_unix"] = time.time()
    healthy["owner_identity"] = "different"
    state.write_text(json.dumps(healthy), encoding="utf-8")
    assert cluster_runtime.probe(state, **kwargs) == 1

    relay = {
        "schema_version": 2,
        "owner_identity": "owner",
        "status": "connected",
        "heartbeat_unix": time.time() - 241,
    }
    state.write_text(json.dumps(relay), encoding="utf-8")
    assert (
        cluster_runtime.probe(
            state,
            component="relay-liveness",
            expected_owner_identity="owner",
            max_age_seconds=240,
        )
        == 1
    )

    relay["status"] = "stopped"
    relay["heartbeat_unix"] = time.time()
    assert (
        cluster_runtime.probe(
            state,
            component="relay-liveness",
            expected_owner_identity="owner",
            max_age_seconds=240,
        )
        == 1
    )
    state.write_text(json.dumps(relay), encoding="utf-8")
    assert (
        cluster_runtime.probe(
            state,
            component="relay-liveness",
            expected_owner_identity="owner",
            max_age_seconds=240,
        )
        == 0
    )


def test_state_health_server_preserves_fail_closed_probe_semantics() -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = int(reservation.getsockname()[1])
    server = StateHealthServer(
        port=port,
        checks={"/ready": lambda: False, "/live": lambda: True},
    )
    server.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/live", timeout=2
        ) as reply:
            assert reply.status == 200
            assert reply.read() == b"ok\n"
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2)
        assert exc_info.value.code == 503
    finally:
        server.close()


def test_state_health_server_can_close_before_start() -> None:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = int(reservation.getsockname()[1])
    server = StateHealthServer(
        port=port,
        checks={"/ready": lambda: False, "/live": lambda: False},
    )
    server.close()


def test_atomic_state_read_recovers_one_transient_partial_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(("{", json.dumps({"schema_version": 2, "status": "running"})))

    class FlakyPath:
        def read_text(self, **_kwargs):  # noqa: ANN003, ANN202
            return next(replies)

    monkeypatch.setattr(cluster_runtime.time, "sleep", lambda _seconds: None)
    assert cluster_runtime._read_state(FlakyPath()) == {  # type: ignore[arg-type]
        "schema_version": 2,
        "status": "running",
    }


def test_recovery_heartbeat_keeps_liveness_fresh_but_readiness_revoked(
    tmp_path: Path,
) -> None:
    state = tmp_path / "controller.json"
    recovery = {
        "status": "recovering",
        "daemon_status": "replacing_supervisor",
        "owner_identity": "owner",
        "session_id": "session",
        "scenario": "scenario",
        "scenario_run_id": "failed-run",
        "heartbeat_unix": time.time() - 60,
        "recovery_reason": "controller_child_exit",
        "recoveries": 1,
        "vendor_exit_class": "nonzero",
        "vendor_exit_code": 1,
    }
    cluster_runtime._write_state(state, **recovery)
    first_publication = cluster_runtime._read_state(state)["published_unix"]

    with cluster_runtime._recovery_heartbeat(
        state, interval_seconds=0.01, **recovery
    ):
        time.sleep(0.04)
        refreshed = cluster_runtime._read_state(state)
        assert refreshed["published_unix"] > first_publication
        assert cluster_runtime._state_ready(
            refreshed,
            component="controller-liveness",
            expected_owner_identity="owner",
            max_age_seconds=0.02,
        )
        assert not cluster_runtime._state_ready(
            refreshed,
            component="controller",
            expected_owner_identity="owner",
            max_age_seconds=30,
        )


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"child_dead": True}, "controller_child_exit"),
        (
            {
                "last_owned_heartbeat": 1.0,
                "consecutive_absence": 3,
                "age_seconds": 31.0,
            },
            "daemon_owner_absent",
        ),
        (
            {"consecutive_errors": 3, "age_seconds": 31.0},
            "daemon_state_unreadable",
        ),
        ({"consecutive_errors": 1, "age_seconds": 31.0}, ""),
    ],
)
def test_supervisor_recovery_requires_converged_loss(
    values: dict[str, object], expected: str
) -> None:
    defaults: dict[str, object] = {
        "child_dead": False,
        "last_owned_heartbeat": 0.0,
        "consecutive_absence": 0,
        "consecutive_errors": 0,
        "age_seconds": 0.0,
        "startup_age_seconds": 0.0,
        "max_age_seconds": 30.0,
    }
    assert cluster_runtime._supervisor_recovery_reason(**(defaults | values)) == expected


def test_vendor_stream_process_observes_real_child_exit_and_drains_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = tmp_path / "vendor-client"
    executable.write_text(
        "#!/bin/sh\n"
        "printf 'secret-shaped-vendor-output\\n'\n"
        "printf 'NPA_OPENPI_METRICS frames=2 round_trips=1\\n'\n"
        "exit 7\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    vendor = cluster_runtime.VendorStreamProcess.start(
        executable=executable,
        runtime=tmp_path,
        scenario="scenario-for-test",
        timeout_seconds=60,
    )
    assert vendor.process.wait(timeout=5) == 7
    assert vendor.exit_snapshot() == ("nonzero", 7)
    assert vendor._drain is not None
    vendor._drain.join(timeout=5)
    output_bytes, output_age = vendor.output_snapshot()
    assert output_bytes > 0
    assert output_age >= 0
    rendered = capsys.readouterr().out
    assert rendered == "NPA_OPENPI_METRICS frames=2 round_trips=1\n"
    assert "secret-shaped" not in rendered


def test_vendor_stream_process_emits_sanitized_camera_rejection_before_child_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    executable = tmp_path / "vendor-client"
    executable.write_text(
        "#!/bin/sh\n"
        "printf 'private vendor prefix NPA_OPENPI_CAMERA_REJECT "
        "view=wrist reason=cube_out_of_frame render_sequence=9 "
        "exterior_red_cube_pixels=42 pair_difference=18.250\\n'\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    vendor = cluster_runtime.VendorStreamProcess.start(
        executable=executable,
        runtime=tmp_path,
        scenario="scenario-for-test",
        timeout_seconds=60,
    )
    deadline = time.monotonic() + 5.0
    while vendor.output_snapshot()[0] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    rendered = capsys.readouterr().out
    vendor.terminate()
    assert rendered == (
        "NPA_OPENPI_CAMERA_REJECT view=wrist reason=cube_out_of_frame "
        "render_sequence=9 exterior_red_cube_pixels=42 pair_difference=18.250\n"
    )
    assert "private vendor prefix" not in rendered
    assert cluster_runtime._sanitized_metric_line(
        b"NPA_OPENPI_CAMERA_REJECT view=wrist reason=secret "
        b"render_sequence=9 exterior_red_cube_pixels=42 pair_difference=18.250"
    ) == ""


def test_vendor_stream_process_outlives_an_unrelated_operator_process(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "vendor-client"
    executable.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    vendor = cluster_runtime.VendorStreamProcess.start(
        executable=executable,
        runtime=tmp_path,
        scenario="scenario-for-test",
        timeout_seconds=60,
    )
    unrelated = __import__("subprocess").run(["/bin/true"], check=False)
    assert unrelated.returncode == 0
    assert vendor.process.poll() is None
    vendor.terminate()


def test_successor_launch_proves_absence_and_dispatches_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    expected = object()

    monkeypatch.setattr(
        cluster_runtime,
        "_cancel_remote_live_runs",
        lambda *_args, **_kwargs: calls.append("reconcile-absence"),
    )
    monkeypatch.setattr(
        cluster_runtime.VendorStreamProcess,
        "start",
        lambda **_kwargs: calls.append("dispatch") or expected,
    )
    result = cluster_runtime._launch_vendor_successor(
        object(),  # type: ignore[arg-type]
        executable=tmp_path / "antioch",
        runtime=tmp_path,
        project_id="project-for-test",
        scenario="scenario-for-test",
        timeout_seconds=60,
    )
    assert result is expected
    assert calls == ["reconcile-absence", "dispatch"]


def test_bound_provider_assignment_is_released_after_exact_run_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cli:
        build_attempts = 0

        def services_build(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("build")
            self.build_attempts += 1
            if self.build_attempts == 1:
                raise AntiochCliError(
                    "assignment SSH is already bound to another local client"
                )
            return {}

        def services_up(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("up")
            return {}

        def machine_release(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("release")
            return {}

    def cancel(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append("cancel")

    monkeypatch.setattr(cluster_runtime, "_cancel_remote_live_runs", cancel)
    recovered = cluster_runtime._start_cluster_service(
        Cli(),  # type: ignore[arg-type]
        runtime=tmp_path,
        project_id="assigned-project-for-test",
        scenario="openpi_franka_mk8s_live",
    )
    assert recovered is True
    assert calls == ["build", "cancel", "release", "build", "up"]


def test_retryable_service_start_failure_recovers_with_capped_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    attempts = 0

    class Cli:
        def services_build(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal attempts
            calls.append("build")
            attempts += 1
            if attempts < 3:
                raise AntiochCliError(
                    "control plane temporarily unavailable",
                    error_type="service_unavailable",
                    retryable=True,
                    http_status=503,
                )
            return {}

        def services_up(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("up")
            return {}

        def machine_release(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("must not be called")

    monkeypatch.setattr(
        cluster_runtime.time,
        "sleep",
        lambda seconds: calls.append(f"sleep:{seconds}"),
    )
    recovered = cluster_runtime._start_cluster_service(
        Cli(),  # type: ignore[arg-type]
        runtime=tmp_path,
        project_id="assigned-project-for-test",
        scenario="openpi_franka_mk8s_live",
    )
    assert recovered is False
    assert calls == ["build", "sleep:2.0", "build", "sleep:5.0", "build", "up"]


def test_fatal_service_start_failure_remains_fatal_without_releasing_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Cli:
        def services_build(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            calls.append("build")
            raise AntiochCliError(
                "authentication denied",
                error_type="authentication",
                retryable=False,
                http_status=401,
            )

        def services_up(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("must not be called")

        def machine_release(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("must not be called")

    monkeypatch.setattr(
        cluster_runtime.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    with pytest.raises(AntiochCliError, match="authentication") as raised:
        cluster_runtime._start_cluster_service(
            Cli(),  # type: ignore[arg-type]
            runtime=tmp_path,
            project_id="assigned-project-for-test",
            scenario="openpi_franka_mk8s_live",
        )
    assert raised.value.retryable is False
    assert calls == ["build"]


def test_remote_state_read_recovers_transient_exec_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(
        (
            "not-base64",
            base64.b64encode(
                b'{"schema_version":2,"status":"connected"}'
            ).decode(),
        )
    )
    commands: list[list[str]] = []

    def execute(*_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        commands.append(kwargs["command"])
        return next(replies)

    monkeypatch.setattr(cluster_deploy.time, "sleep", lambda _seconds: None)
    result = cluster_deploy._read_remote_state(
        execute,
        SimpleNamespace(connect_get_namespaced_pod_exec=object()),
        pod_name="owned-pod",
        namespace="workbench",
        path="/state.json",
    )
    assert result["status"] == "connected"
    assert commands == [
        ["/usr/bin/base64", "-w", "0", "/state.json"],
        ["/usr/bin/base64", "-w", "0", "/state.json"],
    ]


def test_live_metrics_parser_uses_latest_complete_numeric_line() -> None:
    logs = """\
NPA_OPENPI_METRICS frames=10 round_trips=9 latency_p95_ms=123.5
NPA_OPENPI_METRICS frames=broken round_trips=10
NPA_OPENPI_METRICS frames=12 round_trips=11 latency_p95_ms=125.25
"""
    assert cluster_deploy._parse_live_metrics(logs) == {
        "frames": 12,
        "round_trips": 11,
        "latency_p95_ms": 125.25,
    }


def test_retained_openpi_cleanup_owner_is_a_narrow_adoption_proof() -> None:
    metadata = SimpleNamespace(
        labels={"npa.nebius.ai/cleanup-owner": "owned-retained-run"}
    )
    assert cluster_deploy._owned(
        metadata,
        "new-live-identity",
        allow_openpi=True,
        openpi_cleanup_owner="owned-retained-run",
    )
    assert not cluster_deploy._owned(
        metadata,
        "new-live-identity",
        allow_openpi=True,
        openpi_cleanup_owner="different-run",
    )


def test_apply_refuses_ambiguous_policy_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    from kubernetes import client

    monkeypatch.setattr("kubernetes.config.load_kube_config", lambda **_kwargs: None)
    core = SimpleNamespace(read_namespace=lambda **_kwargs: object())
    apps = SimpleNamespace(
        list_namespaced_deployment=lambda **_kwargs: SimpleNamespace(items=[])
    )
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(client, "AppsV1Api", lambda: apps)
    monkeypatch.setattr(client, "NetworkingV1Api", lambda: object())
    with pytest.raises(cluster_deploy.ClusterLiveError, match="exactly one"):
        cluster_deploy.apply_cluster(config)


@pytest.mark.parametrize(
    "case,match",
    [
        ("unowned_policy", "policy Deployment ownership"),
        ("unbound_pvc", "PVC is not Bound"),
        ("unowned_auth", "authentication Secret ownership"),
    ],
)
def test_apply_cluster_guards_ownership_and_dependencies_beyond_selector(
    case: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    from kubernetes import client

    policy_labels = (
        {}
        if case == "unowned_policy"
        else {"app.kubernetes.io/managed-by": cluster_deploy.LIVE_MANAGED_BY}
    )
    deployment = SimpleNamespace(
        metadata=SimpleNamespace(labels=policy_labels),
        spec=SimpleNamespace(
            selector=SimpleNamespace(match_labels=config.policy_selector)
        ),
    )
    pvc = SimpleNamespace(
        status=SimpleNamespace(phase="Pending" if case == "unbound_pvc" else "Bound")
    )
    auth = SimpleNamespace(
        metadata=SimpleNamespace(
            labels=(
                {}
                if case == "unowned_auth"
                else {"app.kubernetes.io/managed-by": cluster_deploy.LIVE_MANAGED_BY}
            )
        ),
        data={"api-key": base64.b64encode(b"a" * 48).decode()},
    )
    core = SimpleNamespace(
        read_namespace=lambda **_kwargs: object(),
        read_namespaced_persistent_volume_claim=lambda **_kwargs: pvc,
        read_namespaced_secret=lambda **_kwargs: auth,
    )
    apps = SimpleNamespace(
        list_namespaced_deployment=lambda **_kwargs: SimpleNamespace(
            items=[deployment]
        )
    )
    monkeypatch.setattr("kubernetes.config.load_kube_config", lambda **_kwargs: None)
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(client, "AppsV1Api", lambda: apps)
    monkeypatch.setattr(client, "NetworkingV1Api", lambda: object())
    with pytest.raises(cluster_deploy.ClusterLiveError, match=match):
        cluster_deploy.apply_cluster(config)


@pytest.mark.parametrize(
    "ready_replicas,restarts,expected_patches,expected_status",
    [
        (1, 0, 0, "not_needed"),
        (0, 0, 1, "rolled_out"),
        (1, 1, 1, "rolled_out"),
    ],
)
def test_reconcile_rolls_only_an_unready_owned_adapter(
    ready_replicas: int,
    restarts: int,
    expected_patches: int,
    expected_status: str,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    deployment = SimpleNamespace(
        metadata=SimpleNamespace(labels=cluster_deploy._labels(config)),
        status=SimpleNamespace(ready_replicas=ready_replicas),
    )
    patches: list[dict[str, object]] = []
    apps = SimpleNamespace(
        read_namespaced_deployment=lambda **_kwargs: deployment,
        patch_namespaced_deployment=lambda **kwargs: patches.append(kwargs),
    )
    core = SimpleNamespace(
        list_namespaced_pod=lambda *_args, **_kwargs: SimpleNamespace(
            items=[
                SimpleNamespace(
                    status=SimpleNamespace(
                        container_statuses=[
                            SimpleNamespace(
                                name="antioch-controller",
                                restart_count=restarts,
                                ready=True,
                            ),
                            SimpleNamespace(
                                name="policy-relay",
                                restart_count=0,
                                ready=True,
                            ),
                        ]
                    ),
                    spec=SimpleNamespace(
                        containers=[
                            SimpleNamespace(
                                name="antioch-controller", image=config.adapter_image
                            ),
                            SimpleNamespace(
                                name="policy-relay", image=config.adapter_image
                            ),
                        ]
                    ),
                )
            ]
        )
    )

    assert (
        cluster_deploy._recover_unready_adapter(apps, core, config)
        == expected_status
    )
    assert len(patches) == expected_patches
    if patches:
        annotations = patches[0]["body"]["spec"]["template"]["metadata"][
            "annotations"
        ]
        assert set(annotations) == {"npa.nebius.ai/owned-recovery-generation"}


def test_policy_placement_status_reports_sanitized_live_gpu_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    container = SimpleNamespace(
        name="policy",
        image="ghcr.io/example/openpi@sha256:" + "a" * 64,
        resources=SimpleNamespace(requests={"nvidia.com/gpu": "1"}),
    )
    deployment = SimpleNamespace(
        spec=SimpleNamespace(
            replicas=1,
            selector=SimpleNamespace(match_labels=config.policy_selector),
            template=SimpleNamespace(
                spec=SimpleNamespace(containers=[container])
            ),
        )
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="policy-pod"),
        spec=SimpleNamespace(node_name="gpu-node"),
    )
    node = SimpleNamespace(
        metadata=SimpleNamespace(labels={"nvidia.com/gpu.product": "NVIDIA-B200"})
    )
    core = SimpleNamespace(
        list_namespaced_pod=lambda *_args, **_kwargs: SimpleNamespace(items=[pod]),
        read_node=lambda **_kwargs: node,
        connect_get_namespaced_pod_exec=object(),
    )

    result = cluster_deploy._policy_placement_status(
        core, deployment, config, lambda *_args, **_kwargs: "10.0\n"
    )

    assert result == {
        "deployment_replicas": 1,
        "pod_count": 1,
        "scheduled_pod_count": 1,
        "gpu_request_per_pod": 1,
        "gpu_request_total": 1,
        "visible_gpu_count": 1,
        "gpu_products": ["NVIDIA-B200"],
        "cuda_capability": "10.0",
        "cuda_sm": "sm_100",
        "image_digest": "sha256:" + "a" * 64,
    }


def test_cluster_status_reports_sanitized_probe_exception_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    from kubernetes import client

    owned = SimpleNamespace(labels=cluster_deploy._labels(config))
    deployment = SimpleNamespace(
        metadata=owned, status=SimpleNamespace(ready_replicas=1)
    )
    policy = SimpleNamespace(
        metadata=SimpleNamespace(
            labels={"app.kubernetes.io/managed-by": cluster_deploy.LIVE_MANAGED_BY}
        ),
        status=SimpleNamespace(ready_replicas=1),
        spec=SimpleNamespace(
            replicas=1,
            selector=SimpleNamespace(match_labels=config.policy_selector),
            template=SimpleNamespace(
                spec=SimpleNamespace(
                    containers=[
                        SimpleNamespace(
                            name="policy",
                            image="ghcr.io/example/openpi@sha256:" + "a" * 64,
                            resources=SimpleNamespace(
                                requests={"nvidia.com/gpu": "1"}
                            ),
                        )
                    ]
                )
            ),
        ),
    )
    pod = SimpleNamespace(
        metadata=SimpleNamespace(name="adapter-pod"),
        status=SimpleNamespace(container_statuses=[]),
        spec=SimpleNamespace(node_name=""),
    )
    policy_pod = SimpleNamespace(
        metadata=SimpleNamespace(name="policy-pod"),
        status=SimpleNamespace(container_statuses=[]),
        spec=SimpleNamespace(node_name="gpu-node"),
    )
    apps = SimpleNamespace(
        read_namespaced_deployment=lambda *_args, **_kwargs: deployment,
        list_namespaced_deployment=lambda *_args, **_kwargs: SimpleNamespace(
            items=[policy]
        ),
    )
    core = SimpleNamespace(
        list_namespaced_pod=lambda *_args, **kwargs: SimpleNamespace(
            items=[pod]
            if "live-identity" in kwargs.get("label_selector", "")
            else [policy_pod]
        ),
        read_node=lambda **_kwargs: SimpleNamespace(
            metadata=SimpleNamespace(
                labels={"nvidia.com/gpu.product": "NVIDIA-B200"}
            )
        ),
        read_namespaced_persistent_volume_claim=lambda **_kwargs: SimpleNamespace(
            status=SimpleNamespace(phase="Bound")
        ),
        connect_get_namespaced_pod_exec=object(),
        read_namespaced_pod_log=lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr("kubernetes.config.load_kube_config", lambda **_kwargs: None)
    monkeypatch.setattr(client, "AppsV1Api", lambda: apps)
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    calls = 0

    def failing_stream(*_args, **_kwargs):  # noqa: ANN202
        nonlocal calls
        calls += 1
        if calls == 1:
            return "10.0\n"
        if calls == 2:
            return base64.b64encode(
                json.dumps(
                    {
                        "status": "running",
                        "scenario": "openpi_franka_mk8s_live",
                        "transport": "same-pod-antioch-tunnel-double-wss",
                        "dev_vm_in_data_path": False,
                    }
                ).encode()
            ).decode()
        if calls == 3:
            raise ValueError("private relay endpoint must not leak")
        raise RuntimeError("private DNS target must not leak")

    monkeypatch.setattr("kubernetes.stream.stream", failing_stream)
    result = cluster_deploy.cluster_status(config)
    assert result["status"] == "not_ready"
    assert result["daemon_liveness_ready"] is False
    assert result["relay_liveness_ready"] is False
    assert result["probe_diagnostics"] == {
        "relay_state": {"status": "failed", "exception_class": "RuntimeError"},
        "policy_dns": {"status": "failed", "exception_class": "RuntimeError"},
    }
    assert {
        key: result["controller"][key]
        for key in (
            "dev_vm_in_data_path",
            "error_type",
            "scenario",
            "status",
            "transport",
        )
    } == {
        "dev_vm_in_data_path": False,
        "error_type": None,
        "scenario": "openpi_franka_mk8s_live",
        "status": "running",
        "transport": "same-pod-antioch-tunnel-double-wss",
    }
    rendered = json.dumps(result)
    assert "private relay" not in rendered
    assert "private DNS" not in rendered


@pytest.mark.parametrize("cleanup_status,stops", [("stopped", True), ("cleanup_failed", False)])
def test_stop_cluster_requires_supported_remote_cleanup_evidence(
    cleanup_status: str,
    stops: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    from kubernetes import client

    deployment = SimpleNamespace(
        metadata=SimpleNamespace(labels=cluster_deploy._labels(config))
    )
    scales: list[dict[str, object]] = []
    apps = SimpleNamespace(
        read_namespaced_deployment=lambda *_args, **_kwargs: deployment,
        patch_namespaced_deployment_scale=lambda *_args, **kwargs: scales.append(kwargs),
    )
    pod = SimpleNamespace(metadata=SimpleNamespace(name="adapter-pod"))
    core = SimpleNamespace(
        list_namespaced_pod=lambda *_args, **_kwargs: SimpleNamespace(items=[pod]),
        connect_get_namespaced_pod_exec=object(),
    )
    monkeypatch.setattr("kubernetes.config.load_kube_config", lambda **_kwargs: None)
    monkeypatch.setattr(client, "AppsV1Api", lambda: apps)
    monkeypatch.setattr(client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(cluster_deploy.time, "sleep", lambda _seconds: None)
    replies = iter(
        (
            "",
            "not-base64",
            base64.b64encode(b"[]").decode(),
            base64.b64encode(
                json.dumps({"status": cleanup_status}).encode()
            ).decode(),
        )
    )
    monkeypatch.setattr("kubernetes.stream.stream", lambda *_args, **_kwargs: next(replies))
    if stops:
        result = cluster_deploy.stop_cluster(config, timeout_seconds=10)
        assert result["remote_terminal_evidence"] == "supported-controller-cleanup"
        assert len(scales) == 1
    else:
        with pytest.raises(cluster_deploy.ClusterLiveError, match="cleanup failed"):
            cluster_deploy.stop_cluster(config, timeout_seconds=10)
        assert scales == []


def test_policy_lookup_uses_pod_selector_not_deployment_metadata() -> None:
    expected = {"app": "policy", "npa.nebius.ai/cleanup-owner": "owned-run"}
    deployment = SimpleNamespace(
        metadata=SimpleNamespace(labels={"app": "different-metadata"}),
        spec=SimpleNamespace(selector=SimpleNamespace(match_labels=dict(expected))),
    )
    assert cluster_deploy._matching_policy_deployments([deployment], expected) == [
        deployment
    ]


def test_source_uses_only_supported_antioch_live_commands() -> None:
    live = Path(cluster_runtime.__file__).read_text(encoding="utf-8")
    helper = (
        Path(cluster_runtime.__file__).with_name("live.py").read_text(encoding="utf-8")
    )
    assert "Rome" not in live
    assert "requests." not in live
    assert "if cleanup_complete:" in live
    assert "restartPolicy" in live
    for command in (
        "services_build",
        "services_up",
        "services_exec",
        "services_copy",
        "services_down",
    ):
        assert command in helper
    assert '"scenario",\n            "run"' in helper


def test_adapter_build_is_base_pinned_and_records_exact_revision() -> None:
    root = Path(cluster_runtime.__file__).resolve().parents[4]
    dockerfile = (root / "docker/workbench/antioch/Dockerfile").read_text()
    build = (root / "docker/workbench/antioch/build.sh").read_text()
    assert "FROM python:3.12-slim-bookworm@sha256:" in dockerfile
    assert 'org.opencontainers.image.revision="${NPA_REVISION}"' in dockerfile
    assert '--build-arg "NPA_REVISION=${REVISION}"' in build
