from __future__ import annotations

from npa.orchestration.skypilot.diagnostics import (
    diagnose_all,
    diagnose_skypilot_output,
)


# Verbatim from a SkyPilot 0.12.2 venv running kubernetes client 36.0.3.
POD_CONFIG_LINE = (
    "Invalid pod_config. Details: Validation error in metadata.labels: "
    "No module named 'kubernetes.client.models.dict[str, str]'"
)


def test_pod_config_failure_is_named_with_a_bootstrap_remedy() -> None:
    diagnosis = diagnose_skypilot_output(POD_CONFIG_LINE)

    assert diagnosis is not None
    assert diagnosis.code == "kubernetes_client_pod_config"
    assert "npa skypilot bootstrap" in diagnosis.remedy
    assert "<36" in diagnosis.remedy


def test_registry_403_is_named_with_a_preflight_remedy() -> None:
    line = (
        'Failed to pull image "registry-us.example/u000/npa-cosmos2-transfer:2.5.1": '
        "failed to resolve reference: unexpected status from HEAD request: 403 Forbidden"
    )

    diagnosis = diagnose_skypilot_output(line)

    assert diagnosis is not None
    assert diagnosis.code == "registry_pull_forbidden"
    assert "preflight-images" in diagnosis.remedy


def test_image_pull_backoff_is_named() -> None:
    diagnosis = diagnose_skypilot_output("pod status: ImagePullBackOff")

    assert diagnosis is not None
    assert diagnosis.code == "image_pull_backoff"


def test_accelerator_precheck_failure_points_at_the_gpus_command() -> None:
    diagnosis = diagnose_skypilot_output(
        "Job 1 failed with FAILED_PRECHECKS: RTX6000:1 not available"
    )

    assert diagnosis is not None
    assert diagnosis.code == "accelerator_unsatisfiable"
    assert "workflow gpus" in diagnosis.remedy


def test_ordinary_progress_output_is_not_diagnosed() -> None:
    assert (
        diagnose_skypilot_output("Launching a new cluster 'sky-jobs-controller'")
        is None
    )
    assert diagnose_skypilot_output("") is None
    assert diagnose_skypilot_output("   ") is None


def test_pod_config_wins_over_a_generic_accelerator_match() -> None:
    # The controller retry loop prints both; the actionable one must be reported.
    line = f"{POD_CONFIG_LINE} (no resources satisfy the request)"

    diagnosis = diagnose_skypilot_output(line)

    assert diagnosis is not None
    assert diagnosis.code == "kubernetes_client_pod_config"


def test_diagnose_all_deduplicates_a_retry_loop() -> None:
    lines = [POD_CONFIG_LINE] * 5 + ["pod status: ImagePullBackOff"]

    codes = [diagnosis.code for diagnosis in diagnose_all(lines)]

    assert codes == ["kubernetes_client_pod_config", "image_pull_backoff"]
