"""Keep the Jammy Python runtime probe pinned and retry-safe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa/tests/docker/fixtures/python-runtime.Dockerfile"


def test_python_runtime_probe_retries_only_the_pinned_snapshot() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "configure-ubuntu-snapshot 20261018T000000Z" in text
    assert "for attempt in 1 2 3 4 5; do" in text
    assert "Acquire::Retries=0" in text
    assert "APT::Update::Error-Mode=any update" in text
    assert "rm -rf /var/lib/apt/lists/*" in text
    assert 'sleep "$attempt"' in text
    assert 'if [ "$attempt" = 5 ]; then exit 1; fi' in text
    for forbidden in (
        "archive.ubuntu.com",
        "security.ubuntu.com",
        "AllowInsecureRepositories",
        "--allow-unauthenticated",
        "trusted=yes",
    ):
        assert forbidden not in text
