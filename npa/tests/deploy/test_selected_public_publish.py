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


def test_copy_phase_marker_uses_the_github_output_file(monkeypatch, tmp_path) -> None:
    from npa.deploy import publish_public

    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    publish_public._mark_copy_phase_complete()

    assert github_output.read_text(encoding="utf-8") == "copy_phase_completed=true\n"


@pytest.mark.parametrize(
    "tag", ["latest", "dev-deadbeef", "release-latest", "main", "edge", "../release", "-option", "bad:tag", "a/b", "release\nnew", "x" * 129]
)
def test_additive_release_refuses_mutable_or_unsafe_tags(tag) -> None:
    module = _load_script()
    with pytest.raises(ValueError, match="safe, immutable"):
        module._validate_release_override(["detection-training"], tag, "sha256:" + "a" * 64, "b" * 40)


@pytest.mark.parametrize(
    "tools,tag,digest,sha",
    [
        (["detection-training", "genesis"], "release-1", "sha256:" + "a" * 64, "b" * 40),
        (["detection-training"], "release-1", "", "b" * 40),
        (["detection-training"], "", "sha256:" + "a" * 64, "b" * 40),
        (["detection-training"], "release-1", "sha256:" + "a" * 64, None),
        (["detection-training"], "release-1", "sha256:short", "b" * 40),
        (["detection-training"], "release-1", "sha256:" + "a" * 64, "short"),
    ],
)
def test_additive_release_requires_one_exact_accepted_source(tools, tag, digest, sha) -> None:
    with pytest.raises(ValueError):
        _load_script()._validate_release_override(tools, tag, digest, sha)


@pytest.fixture
def additive_registry(monkeypatch):
    """Exercise the real plan/preflight/copy adapter with only registry I/O replaced."""
    from npa.deploy import publish_public

    module = _load_script()
    sha = "b" * 40
    digest = "sha256:" + "a" * 64
    repository = "ghcr.io/nebius/nebius-physical-ai/npa-detection-training"
    source = repository + ":dev-" + sha
    target = repository + ":runtime-recovery-1"
    state = {source: digest, repository + "@" + digest: digest}
    calls = []
    errors = {}
    revision = {"value": sha}

    def lookup(ref, **_):
        if ref in errors:
            return False, errors[ref]
        return (True, state[ref]) if ref in state else (False, "MANIFEST_UNKNOWN")

    def copy(argv, **_):
        assert argv[:2] == ["/fixture/crane", "copy"]
        calls.append(argv)
        state[argv[3]] = state[argv[2]]

    monkeypatch.setattr(publish_public, "_crane_digest", lookup)
    monkeypatch.setattr(module, "_crane_digest", lookup)
    monkeypatch.setattr(publish_public, "_crane_manifest_readable", lambda ref: lookup(ref))
    monkeypatch.setattr(publish_public.shutil, "which", lambda _: "/fixture/crane")
    monkeypatch.setattr(publish_public.subprocess, "run", copy)
    monkeypatch.setattr(publish_public, "anonymous_pull_ok", lookup)
    monkeypatch.setattr(module, "anonymous_digest", lookup)
    monkeypatch.setattr(module, "_mark_copy_phase_complete", lambda: None)
    monkeypatch.setattr(module, "_crane_json", lambda _: {"config": {"Labels": {"org.opencontainers.image.revision": revision["value"]}}})
    argv = [str(SCRIPT), "--tool", "detection-training", "--target", "ghcr.io/nebius/nebius-physical-ai", "--development-sha", sha, "--release-tag", "runtime-recovery-1", "--expected-source-digest", digest, "--mode", "publish"]
    monkeypatch.setattr(sys, "argv", argv)
    return module, state, calls, errors, revision, source, target, digest, argv


def test_additive_release_promotes_only_exact_accepted_bytes_then_is_idempotent(additive_registry) -> None:
    module, state, calls, _, _, source, target, digest, _ = additive_registry
    before = dict(state)
    assert module.main() == 0
    assert calls == [["/fixture/crane", "copy", source.rsplit(":", 1)[0] + "@" + digest, target]]
    assert state == {**before, target: digest}
    assert module.main() == 0
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["plan", "preflight", "verify"])
def test_additive_read_modes_never_copy(additive_registry, mode) -> None:
    module, state, calls, _, _, _, target, digest, argv = additive_registry
    argv[-1] = mode
    if mode == "verify":
        state[target] = digest
    assert module.main() == 0
    assert calls == []


def test_additive_wrong_accepted_digest_stops_before_copy(additive_registry) -> None:
    module, _, calls, _, _, _, _, _, argv = additive_registry
    argv[argv.index("--expected-source-digest") + 1] = "sha256:" + "c" * 64
    with pytest.raises(RuntimeError, match="GPU-accepted expected digest"):
        module.main()
    assert calls == []


def test_additive_wrong_source_revision_stops_before_copy(additive_registry) -> None:
    module, _, calls, _, revision, *_ = additive_registry
    revision["value"] = "c" * 40
    with pytest.raises(RuntimeError, match="source revision"):
        module.main()
    assert calls == []


@pytest.mark.parametrize("mode", ["preflight", "publish"])
def test_additive_existing_different_target_is_never_overwritten(additive_registry, mode) -> None:
    module, state, calls, _, _, _, target, _, argv = additive_registry
    argv[-1] = mode
    state[target] = "sha256:" + "c" * 64
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module.main()
    assert calls == []
    assert state[target] == "sha256:" + "c" * 64


@pytest.mark.parametrize("error", ["UNAUTHORIZED", "FORBIDDEN MANIFEST_UNKNOWN", "upstream unavailable", "timed out"])
def test_additive_unknown_or_denied_target_never_copies(additive_registry, error) -> None:
    module, _, calls, errors, _, _, target, *_ = additive_registry
    errors[target] = error
    with pytest.raises(RuntimeError, match="cannot prove additive target"):
        module.main()
    assert calls == []


def test_additive_target_race_is_rechecked_at_actual_copy_boundary(additive_registry, monkeypatch) -> None:
    module, state, calls, _, _, _, target, *_ = additive_registry
    original = module._require_additive_target

    def concurrent_publish(item, digest):
        original(item, digest)
        state[target] = "sha256:" + "c" * 64

    monkeypatch.setattr(module, "_require_additive_target", concurrent_publish)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module.main()
    assert calls == []


def test_additive_source_gate_failure_is_not_bypassed(additive_registry, monkeypatch) -> None:
    from npa.deploy import publish_public
    module, _, calls, *_ = additive_registry
    monkeypatch.setattr(publish_public, "verify_validated_publication", lambda _: (False, "payload/GPU acceptance missing"))
    assert module.main() == 1
    assert calls == []


def test_additive_anonymous_parity_failure_is_reported(additive_registry, monkeypatch) -> None:
    module, _, calls, _, _, _, target, *_ = additive_registry
    original = module.anonymous_digest
    monkeypatch.setattr(module, "anonymous_digest", lambda ref: (True, "sha256:" + "c" * 64) if ref == target else original(ref))
    with pytest.raises(RuntimeError, match="additive release is not anonymously verified"):
        module.main()
    assert len(calls) == 1
