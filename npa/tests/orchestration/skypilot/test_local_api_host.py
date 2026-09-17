"""Reject unsupported isolated API hosts before creating state or processes."""

from pathlib import Path

import pytest

from npa.orchestration.skypilot import local_api as api


@pytest.mark.parametrize("platform,has_proc", [("darwin", True), ("linux", False)])
@pytest.mark.parametrize("operation", ["environment", "start", "inspect", "stop"])
def test_unsupported_host_fails_before_state_or_process_changes(
    tmp_path, monkeypatch, platform, has_proc, operation,
):
    monkeypatch.setattr(api.sys, "platform", platform)
    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir",
        lambda path: has_proc if path == Path("/proc/self") else original_is_dir(path),
    )
    processes = []
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **kw: processes.append(a))
    isolated = tmp_path / "new-runtime"
    operations = {
        "environment": lambda: api.isolated_api_environment(isolated, {}),
        "start": lambda: api.ensure_isolated_api(
            isolated_dir=isolated, sky_executable="/unused/sky", environment={}, cwd=str(tmp_path),
        ),
        "inspect": lambda: api.owned_daemon_environment(isolated),
        "stop": lambda: api.stop_isolated_api(isolated),
    }
    with pytest.raises(api.IsolatedApiError, match="requires a Linux operator host with /proc"):
        operations[operation]()
    assert not isolated.exists()
    assert processes == []
