from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_antioch_payload.py"
)


def _load_scanner():
    spec = importlib.util.spec_from_file_location("scan_image_antioch_payload", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scanner_falls_back_to_python_when_python3_is_absent(
    monkeypatch, capsys
) -> None:
    scanner = _load_scanner()
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[4] == "python3":
            return subprocess.CompletedProcess(argv, 127, "", "not found")
        return subprocess.CompletedProcess(
            argv, 0, '{"packages": [], "forbidden_paths": []}\n', ""
        )

    monkeypatch.setattr(scanner, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "example.invalid/image@sha256:abc"])

    assert scanner.main() == 0
    assert [call[4] for call in calls] == ["python3", "python"]
    assert '"verdict": "clean"' in capsys.readouterr().out


def test_scanner_uses_python3_when_available(monkeypatch) -> None:
    scanner = _load_scanner()
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 1, '{"packages": ["antioch"], "forbidden_paths": []}\n', ""
        )

    monkeypatch.setattr(scanner, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "example.invalid/image@sha256:abc"])

    assert scanner.main() == 1
    assert [call[4] for call in calls] == ["python3"]
