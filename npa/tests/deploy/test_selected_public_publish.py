"""Behavioral coverage for the scoped public publisher wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from npa.deploy.publish_public import PublishItem

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github" / "scripts" / "publish_selected_public_image.py"


def _load_script():  # noqa: ANN202 - imported script module is intentionally dynamic
    spec = importlib.util.spec_from_file_location(
        "publish_selected_public_image", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_uses_the_digest_pinned_item_returned_by_preflight(
    monkeypatch,
) -> None:
    module = _load_script()
    digest = "sha256:" + "a" * 64
    selected = PublishItem(
        tool="wan2-2",
        source_ref="source.example/npa-wan2-2:accepted",
        target_ref="ghcr.io/example/npa-wan2-2:accepted",
    )
    pinned = PublishItem(
        tool=selected.tool,
        source_ref=f"source.example/npa-wan2-2@{digest}",
        target_ref=selected.target_ref,
    )
    copied: list[PublishItem] = []
    verified: list[list[PublishItem]] = []
    copy_phase_marks: list[bool] = []

    monkeypatch.setattr(module, "build_publish_plan", lambda **_: [selected])
    monkeypatch.setattr(module, "_preflight_or_explain", lambda _: [pinned])
    monkeypatch.setattr(module, "_crane_copy", lambda item: copied.append(item) or True)
    monkeypatch.setattr(
        module, "_mark_copy_phase_complete", lambda: copy_phase_marks.append(True)
    )
    monkeypatch.setattr(
        module, "verify_public", lambda plan: verified.append(plan) or []
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--tool",
            "wan2-2",
            "--target",
            "ghcr.io/example",
            "--mode",
            "publish",
        ],
    )

    assert module.main() == 0
    assert copied == [pinned]
    assert verified == [[pinned]]
    assert copy_phase_marks == [True]


def test_publish_preflights_then_copies_multiple_selected_images(
    monkeypatch, capsys
) -> None:
    module = _load_script()
    curator = PublishItem(
        tool="cosmos-curate",
        source_ref="source.example/npa-cosmos-curate:release",
        target_ref="ghcr.io/example/npa-cosmos-curate:release",
    )
    evaluator = PublishItem(
        tool="cosmos-evaluator",
        source_ref="source.example/npa-cosmos-evaluator:release",
        target_ref="ghcr.io/example/npa-cosmos-evaluator:release",
    )
    unrelated = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot:release",
        target_ref="ghcr.io/example/npa-lerobot:release",
    )
    pinned = [
        PublishItem(
            tool=item.tool,
            source_ref=item.source_ref.rsplit(":", 1)[0] + "@sha256:" + char * 64,
            target_ref=item.target_ref,
        )
        for item, char in ((curator, "a"), (evaluator, "b"))
    ]
    preflighted: list[list[PublishItem]] = []
    copied: list[PublishItem] = []
    verified: list[list[PublishItem]] = []
    copy_phase_marks: list[bool] = []

    monkeypatch.setattr(
        module, "build_publish_plan", lambda **_: [curator, evaluator, unrelated]
    )

    def preflight(plan: list[PublishItem]) -> list[PublishItem]:
        preflighted.append(plan)
        return pinned

    monkeypatch.setattr(module, "_preflight_or_explain", preflight)
    monkeypatch.setattr(
        module,
        "_crane_copy",
        lambda item: copied.append(item) or item.tool == "cosmos-curate",
    )
    monkeypatch.setattr(
        module, "_mark_copy_phase_complete", lambda: copy_phase_marks.append(True)
    )
    monkeypatch.setattr(
        module, "verify_public", lambda plan: verified.append(plan) or []
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--tool",
            "cosmos-curate, cosmos-evaluator",
            "--target",
            "ghcr.io/example",
            "--mode",
            "publish",
        ],
    )

    assert module.main() == 0
    assert preflighted == [[curator, evaluator]]
    assert copied == pinned
    assert verified == [pinned]
    assert copy_phase_marks == [True]
    assert "Copied 1 of 2 image(s); 1 already current." in capsys.readouterr().out


def test_tool_selector_accepts_repeated_csv_and_space_separated_values() -> None:
    module = _load_script()

    assert module._parse_tools(
        ["cosmos-curate,cosmos-evaluator", "fiftyone lichtblick"]
    ) == [
        "cosmos-curate",
        "cosmos-evaluator",
        "fiftyone",
        "lichtblick",
    ]


def test_tool_selector_rejects_duplicates() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="duplicate workbench tool.*fiftyone"):
        module._parse_tools(["fiftyone,lichtblick", "fiftyone"])


def test_copy_phase_is_not_marked_when_preflight_stops_publication(
    monkeypatch,
) -> None:
    module = _load_script()
    selected = PublishItem(
        tool="wan2-2",
        source_ref="source.example/npa-wan2-2:accepted",
        target_ref="ghcr.io/example/npa-wan2-2:accepted",
    )

    monkeypatch.setattr(module, "build_publish_plan", lambda **_: [selected])
    monkeypatch.setattr(module, "_preflight_or_explain", lambda _: [])
    monkeypatch.setattr(
        module,
        "_crane_copy",
        lambda _: (_ for _ in ()).throw(AssertionError("copy must not run")),
    )
    monkeypatch.setattr(
        module,
        "_mark_copy_phase_complete",
        lambda: (_ for _ in ()).throw(AssertionError("marker must not be written")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--tool",
            "wan2-2",
            "--target",
            "ghcr.io/example",
            "--mode",
            "publish",
        ],
    )

    assert module.main() == 1


def test_checklist_is_scoped_to_the_selected_images(monkeypatch, capsys) -> None:
    module = _load_script()
    selected = PublishItem(
        tool="wan2-2",
        source_ref="source.example/npa-wan2-2:accepted",
        target_ref="ghcr.io/example/npa-wan2-2:accepted",
    )
    unrelated = PublishItem(
        tool="lerobot",
        source_ref="source.example/npa-lerobot:accepted",
        target_ref="ghcr.io/example/npa-lerobot:accepted",
    )
    second = PublishItem(
        tool="fiftyone",
        source_ref="source.example/npa-fiftyone:accepted",
        target_ref="ghcr.io/example/npa-fiftyone:accepted",
    )
    failures = [(selected, "HTTP 403"), (second, "HTTP 403")]
    verified: list[list[PublishItem]] = []

    monkeypatch.setattr(
        module, "build_publish_plan", lambda **_: [selected, second, unrelated]
    )
    monkeypatch.setattr(
        module, "verify_public", lambda plan: verified.append(plan) or failures
    )
    monkeypatch.setattr(module, "visibility_checklist", lambda _: "- [ ] Wan package")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--tool",
            "wan2-2,fiftyone",
            "--target",
            "ghcr.io/example",
            "--mode",
            "checklist",
        ],
    )

    assert module.main() == 1
    assert verified == [[selected, second]]
    assert "- [ ] Wan package" in capsys.readouterr().out


def test_copy_phase_marker_uses_the_github_output_file(monkeypatch, tmp_path) -> None:
    from npa.deploy import publish_public

    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    publish_public._mark_copy_phase_complete()

    assert github_output.read_text(encoding="utf-8") == "copy_phase_completed=true\n"
