"""Static contract checks for the Rerun SkyPilot visualization worker."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "rerun-viewer" / "Dockerfile"
ENTRYPOINT = ROOT / "npa" / "docker" / "workbench" / "rerun-viewer" / "entrypoint.sh"


def test_rerun_viewer_satisfies_skypilot_non_root_setup_contract() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "@sha256:" in text
    assert "apt-get upgrade -y --no-install-recommends" in text
    for package in ("openssh-server", "rsync", "sudo"):
        assert package in text
    assert "ubuntu ALL=(ALL) NOPASSWD:ALL" in text
    assert "USER ubuntu" in text
    assert "NPA_LIGHT_WORKBENCH_TOOL=rerun-viewer" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text
    assert 'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in text


def test_rerun_viewer_entrypoint_preserves_orchestrator_argv() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")

    assert (
        'ENTRYPOINT ["/opt/npa/docker/workbench/rerun-viewer/entrypoint.sh"]'
        in dockerfile
    )
    assert 'exec "$@"' in entrypoint
    assert (
        "rerun --serve-web --web-viewer --bind 0.0.0.0 "
        "--web-viewer-port 9090 --port 9876"
    ) in entrypoint


def test_rerun_viewer_installs_exact_sdk_in_owned_venv() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "python -m venv /opt/rerun/venv" in text
    assert '"rerun-sdk==${RERUN_SDK_VERSION}"' in text
    assert "--no-cache-dir" in text


def test_rerun_viewer_gates_every_pi05_cpu_role() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert 'm.version("kubernetes") == "36.0.3"' in text
    assert 'm.version("cryptography") == "50.0.2"' in text
    assert "npa.workflows.sim2real.pi05_stage --help" in text
    assert "npa.workflows.sim2real.pi05_data --help" in text
    assert "npa.workflows.byof.openpi_service_lifecycle --help" in text


def test_rerun_viewer_pins_kubernetes_aiohttp_closure() -> None:
    """The minimal CPU image must include Kubernetes 36's async transport."""

    requirements = (
        ROOT / "npa/docker/workbench/rerun-viewer/kubernetes36-requirements.txt"
    ).read_text(encoding="utf-8")
    expected = {
        "aiohappyeyeballs==2.7.1",
        "aiohttp==3.14.4",
        "aiosignal==1.4.0",
        "cryptography==50.0.2",
        "frozenlist==1.8.0",
        "kubernetes==36.0.3",
        "multidict==6.9.1",
        "propcache==0.5.4",
        "yarl==1.25.1",
    }
    assert expected <= set(requirements.splitlines())
    assert "kubernetes36-requirements.txt" in DOCKERFILE.read_text(encoding="utf-8")


def test_rerun_viewer_runs_native_conversion_readback_smoke() -> None:
    """The image build must exercise pandas, Parquet and encoded video."""

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    viewer_requirements = (
        ROOT / "npa/docker/workbench/common/sim2real-viewer-requirements.txt"
    ).read_text(encoding="utf-8")
    verifier = (
        ROOT / "npa/docker/workbench/rerun-viewer/verify_pi05_conversion.py"
    ).read_text(encoding="utf-8")
    assert "pandas==3.0.6" in viewer_requirements
    assert "python /opt/npa/verify_pi05_conversion.py" in dockerfile
    assert "TemporaryDirectory" in verifier
    for evidence in ("tasks.to_pandas()", "data.num_rows", "episodes", "ffmpeg"):
        assert evidence in verifier
