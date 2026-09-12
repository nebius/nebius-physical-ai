from __future__ import annotations

import io
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess

import pytest

from npa._public_https import PublicDownloadError
from npa.guardrails import confidentiality, robomimic_attribution
from npa.guardrails.confidentiality import (
    compile_builtin_nebius_infra,
    compile_denylist,
    load_denylist_pattern,
    main,
    scan_diff_text,
    scan_paths,
    scan_text,
    should_skip_unconfigured_fork_pull_request,
    tracked_text_files,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOMIMIC_PATH = robomimic_attribution.REPOSITORY_PATH
ROBOMIMIC_LOCK = REPO_ROOT / ROBOMIMIC_PATH
ROBOMIMIC_ATTRIBUTION_TEXT = ROBOMIMIC_LOCK.read_text().splitlines()[248]
ROBOMIMIC_ATTRIBUTION_PATTERN = re.escape(ROBOMIMIC_ATTRIBUTION_TEXT)


def test_confidentiality_matcher_reports_redacted_locations_only() -> None:
    denylist = compile_denylist(r"synthetic-secret|project-codename")

    hits = scan_text(
        "public line\ncontains synthetic-secret here\ncontains project-codename here\n",
        denylist,
        source="fixture.txt",
    )

    assert [(hit.source, hit.line_number) for hit in hits] == [
        ("fixture.txt", 2),
        ("fixture.txt", 3),
    ]
    assert not hasattr(hits[0], "line")


def test_confidentiality_matcher_can_ignore_case() -> None:
    denylist = compile_denylist(r"synthetic-secret", ignore_case=True)

    hits = scan_text("contains SYNTHETIC-SECRET here\n", denylist, source="fixture.txt")

    assert [(hit.source, hit.line_number) for hit in hits] == [("fixture.txt", 1)]


@pytest.mark.parametrize(
    "text",
    [
        "tenant: " + "e00" + "a" * 16,
        "project: project-" + "u00" + "b" * 16,
        "cluster: mk8scluster-" + "u00" + "c" * 16,
        "node group: mk8snodegroup-" + "u00" + "d" * 16,
        "instance: computeinstance-" + "u00" + "e" * 16,
        "capacity block: capacityblockgroup-" + "u00" + "f" * 16,
        "bucket: storagebucket-" + "u00" + "g" * 16,
        "registry: registry-" + "u00" + "h" * 16,
        "network: network-" + "u00" + "j" * 16,
        "object: s3://campaign-bucket-" + "a1b2c3d4" + "/evidence.json",
        (
            "image: registry.example/validation-registry-"
            + "20260815"
            + "/npa-tool@sha256:"
            + "1" * 64
        ),
        "campaign: checkpoint-evaluation-" + "20260815",
    ],
)
def test_builtin_nebius_infra_guard_detects_synthetic_leak_classes(text: str) -> None:
    hits = scan_text(text, compile_builtin_nebius_infra(), source="synthetic.txt")

    assert [(hit.source, hit.line_number) for hit in hits] == [("synthetic.txt", 1)]


def test_builtin_nebius_infra_guard_allows_placeholders_and_hashes() -> None:
    sha40 = "e02" + "a" * 37
    sha64 = "e07" + "b" * 61
    text = "\n".join(
        [
            "Nebius project and tenant identifiers are configured externally.",
            "project: project-test",
            "cluster: mk8scluster-a",
            "image: <your-registry>/npa-tool:tag",
            "object: s3://example-bucket/runs/example/",
            "object: s3://${NPA_ARTIFACT_BUCKET}/runs/example/",
            f"framework revision: {sha40}",
            f"image digest: sha256:{sha64}",
        ]
    )

    assert scan_text(text, compile_builtin_nebius_infra(), source="safe.txt") == []


def test_builtin_nebius_infra_cli_scans_pr_text_without_echoing_match(
    monkeypatch, capsys
) -> None:
    private_value = "project-" + "u00" + "z" * 16
    monkeypatch.setattr("sys.stdin", io.StringIO(f"project: {private_value}\n"))

    assert main(["--built-in-nebius-infra", "--stdin-source", "pr-body"]) == 1
    captured = capsys.readouterr()
    assert "pr-body:1" in captured.err
    assert private_value not in captured.err


def test_builtin_nebius_infra_diff_scan_ignores_removed_values() -> None:
    private_value = "project-" + "u00" + "q" * 16
    diff = "\n".join(
        [
            "diff --git a/report.md b/report.md",
            "--- a/report.md",
            "+++ b/report.md",
            f"-{private_value}",
            "+exact project identifier retained externally",
        ]
    )

    assert (
        scan_diff_text(
            diff,
            compile_builtin_nebius_infra(),
            source="staged-diff",
        )
        == []
    )

    added = diff + f"\n+{private_value}\n"
    hits = scan_diff_text(
        added,
        compile_builtin_nebius_infra(),
        source="staged-diff",
    )
    assert [(hit.source, hit.line_number) for hit in hits] == [("staged-diff", 6)]


def test_builtin_nebius_infra_guard_is_wired_into_existing_gitleaks_ci() -> None:
    workflow = (REPO_ROOT / ".github/workflows/gitleaks.yml").read_text(
        encoding="utf-8"
    )
    gitleaks = (REPO_ROOT / ".gitleaks.toml").read_text(encoding="utf-8")

    assert "--config .gitleaks.toml" in workflow
    assert "[eu]00" in gitleaks
    assert "capacityblockgroup" in gitleaks
    assert "nebius-private-object-location" in gitleaks
    assert "nebius-private-registry-location" in gitleaks
    assert "nebius-dated-operational-name" in gitleaks


def test_tree_scan_skips_binary_files_but_keeps_text_hits(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "public.txt").write_text(
        "contains synthetic-secret\n", encoding="utf-8"
    )
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00synthetic-secret\n")
    subprocess.run(["git", "add", "public.txt", "image.png"], cwd=tmp_path, check=True)

    denylist = compile_denylist(r"synthetic-secret")
    paths = tracked_text_files(tmp_path)
    hits = scan_paths(paths, denylist, repo_root=tmp_path)

    assert {path.name for path in paths} == {"public.txt"}
    assert [(hit.source, hit.line_number) for hit in hits] == [("public.txt", 1)]


def test_confidentiality_matcher_rejects_empty_pattern() -> None:
    with pytest.raises(ValueError, match="empty"):
        compile_denylist("")


def test_confidentiality_matcher_names_empty_pattern_source() -> None:
    with pytest.raises(ValueError, match="INFRA_DENYLIST is empty"):
        compile_denylist("", source="INFRA_DENYLIST")


def test_confidentiality_matcher_surfaces_invalid_regex() -> None:
    with pytest.raises(re.error):
        compile_denylist("[")


def test_denylist_loader_prefers_env_pattern_over_file(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("file-only-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST",
        pattern_file=pattern_file,
        environ={"CUSTOMER_DENYLIST": "env-only-pattern"},
    )

    assert loaded.pattern == "env-only-pattern"
    assert loaded.source == "CUSTOMER_DENYLIST"


def test_denylist_loader_reads_configured_file_env(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("configured-file-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST",
        environ={"CUSTOMER_DENYLIST_FILE": str(pattern_file)},
    )

    assert loaded.pattern == "configured-file-pattern\n"
    assert loaded.source == f"CUSTOMER_DENYLIST_FILE:{pattern_file}"


def test_denylist_loader_reads_explicit_pattern_file(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("explicit-file-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST", pattern_file=pattern_file, environ={}
    )

    assert loaded.pattern == "explicit-file-pattern\n"
    assert loaded.source == f"--pattern-file:{pattern_file}"


def test_denylist_loader_fails_closed_when_source_missing(tmp_path) -> None:
    missing_file = tmp_path / "missing.regex"

    with pytest.raises(ValueError, match="CUSTOMER_DENYLIST is empty"):
        load_denylist_pattern(
            "CUSTOMER_DENYLIST", pattern_file=missing_file, environ={}
        )


def test_confidentiality_scan_cli_fails_closed_when_source_missing(tmp_path) -> None:
    missing_file = tmp_path / "missing.regex"

    assert (
        main(["--repo-root", str(tmp_path), "--pattern-file", str(missing_file)]) == 2
    )


def test_should_skip_unconfigured_fork_pull_request(tmp_path) -> None:
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "repository": {"full_name": "nebius/nebius-physical-ai"},
                "pull_request": {
                    "head": {"repo": {"full_name": "contributor/nebius-physical-ai"}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert should_skip_unconfigured_fork_pull_request(
        "CUSTOMER_DENYLIST",
        environ={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        },
    )


def test_should_not_skip_same_repo_pull_request(tmp_path) -> None:
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "repository": {"full_name": "nebius/nebius-physical-ai"},
                "pull_request": {
                    "head": {"repo": {"full_name": "nebius/nebius-physical-ai"}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert not should_skip_unconfigured_fork_pull_request(
        "CUSTOMER_DENYLIST",
        environ={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        },
    )


def _robomimic_source_repo(tmp_path: Path, *, copy: bool = False) -> Path:
    repo = tmp_path / "robomimic-source"
    lock = repo / ROBOMIMIC_PATH
    lock.parent.mkdir(parents=True)
    lock.write_bytes(ROBOMIMIC_LOCK.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", ROBOMIMIC_PATH], cwd=repo, check=True)
    if copy:
        foreign = repo / "docs/foreign-license-record.txt"
        foreign.parent.mkdir(parents=True)
        foreign.write_text(ROBOMIMIC_ATTRIBUTION_TEXT + "\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", str(foreign.relative_to(repo))], cwd=repo, check=True
        )
    return repo


def _robomimic_proof_directory(tmp_path: Path) -> Path:
    proof = tmp_path / "robomimic-proof"
    proof.mkdir(mode=0o700)
    return proof


def _run_robomimic_tree(
    repo: Path,
    proof: Path,
    monkeypatch: pytest.MonkeyPatch,
    pattern: str = ROBOMIMIC_ATTRIBUTION_PATTERN,
) -> int:
    monkeypatch.setenv("CUSTOMER_DENYLIST", pattern)
    return main(
        [
            "--repo-root",
            str(repo),
            "--tree",
            "--ncore-attribution-proof-directory",
            str(proof),
        ]
    )


def _commit_fixture(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            message,
        ],
        cwd=repo,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def test_exact_robomimic_lock_hit_is_dispositioned_after_official_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    proof = _robomimic_proof_directory(tmp_path)
    calls: list[tuple[bytes, Path]] = []

    def verified(lock: bytes, proof_directory: Path) -> dict[str, object]:
        calls.append((lock, proof_directory))
        return {"copyright_sources": [{}, {}]}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", verified)

    assert _run_robomimic_tree(repo, proof, monkeypatch) == 0
    captured = capsys.readouterr()
    assert f"{ROBOMIMIC_PATH}:249" in captured.err
    assert "raw=1 dispositioned=1 unresolved=0" in captured.out
    assert ROBOMIMIC_ATTRIBUTION_TEXT not in captured.out + captured.err
    assert calls == [(ROBOMIMIC_LOCK.read_bytes(), proof)]


def test_robomimic_attribution_foreign_copy_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _robomimic_source_repo(tmp_path, copy=True)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert _run_robomimic_tree(
        repo, _robomimic_proof_directory(tmp_path), monkeypatch
    ) == 1
    captured = capsys.readouterr()
    assert "docs/foreign-license-record.txt:1" in captured.err
    assert "raw=2 dispositioned=1 unresolved=1" in captured.err


def test_changed_robomimic_lock_and_private_marker_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    marker = "synthetic-private-marker"
    lock = repo / ROBOMIMIC_PATH
    lock.write_bytes(lock.read_bytes() + marker.encode() + b"\n")
    called = False

    def unexpected(_lock: bytes, _proof: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", unexpected)

    assert _run_robomimic_tree(
        repo,
        _robomimic_proof_directory(tmp_path),
        monkeypatch,
        rf"{ROBOMIMIC_ATTRIBUTION_PATTERN}|{marker}",
    ) == 1
    captured = capsys.readouterr()
    assert "raw=2 dispositioned=0 unresolved=2" in captured.err
    assert marker not in captured.out + captured.err
    assert not called


def test_wrong_robomimic_line_cannot_receive_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    called = False

    def unexpected(_lock: bytes, _proof: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", unexpected)
    hit = confidentiality.ScanHit(
        ROBOMIMIC_PATH, 250, repository_path=ROBOMIMIC_PATH
    )

    assert confidentiality._robomimic_disposition_indexes(
        [hit], repo, _robomimic_proof_directory(tmp_path)
    ) == set()
    assert not called


@pytest.mark.parametrize("hostile_git_shape", ["executable", "symlink-alias"])
def test_hostile_robomimic_git_shapes_cannot_receive_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hostile_git_shape: str,
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    if hostile_git_shape == "executable":
        subprocess.run(
            ["git", "update-index", "--chmod=+x", ROBOMIMIC_PATH],
            cwd=repo,
            check=True,
        )
    else:
        alias = repo / "lock-alias"
        alias.symlink_to(ROBOMIMIC_PATH)
        subprocess.run(["git", "add", "lock-alias"], cwd=repo, check=True)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert _run_robomimic_tree(
        repo, _robomimic_proof_directory(tmp_path), monkeypatch
    ) == 1
    assert "raw=1 dispositioned=0 unresolved=1" in capsys.readouterr().err


def test_robomimic_diff_requires_exact_canonical_post_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    lock = repo / ROBOMIMIC_PATH
    lock.unlink()
    subprocess.run(["git", "rm", "--cached", ROBOMIMIC_PATH], cwd=repo, check=True)
    base = _commit_fixture(repo, "Empty base")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(ROBOMIMIC_LOCK.read_bytes())
    head = _commit_fixture(repo, "Canonical lock")
    monkeypatch.setenv("CUSTOMER_DENYLIST", ROBOMIMIC_ATTRIBUTION_PATTERN)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert main(
        [
            "--repo-root",
            str(repo),
            "--diff-range",
            f"{base}..{head}",
            "--ncore-attribution-proof-directory",
            str(_robomimic_proof_directory(tmp_path)),
        ]
    ) == 0
    assert "raw=1 dispositioned=1 unresolved=0" in capsys.readouterr().out

    lines = lock.read_text().splitlines()
    lines[248] += " "
    lock.write_text("\n".join(lines) + "\n")
    changed = _commit_fixture(repo, "Changed lock")
    hits = confidentiality.scan_git_diff(
        repo,
        f"{head}..{changed}",
        compile_denylist(ROBOMIMIC_ATTRIBUTION_PATTERN),
    )
    assert len(hits) == 1
    assert hits[0].repository_path is None


def _synthetic_debian_source_proof(
    *, version: str = "0.11.7-2"
) -> tuple[bytes, bytes]:
    checksum_lines = "\n".join(
        f" {digest} {size} {name}"
        for digest, size, name in robomimic_attribution._SOURCE_FILES
    )
    source_index = lzma.compress(
        (
            "Package: libbsd\n"
            f"Version: {version}\n"
            "Directory: pool/main/libb/libbsd\n"
            f"Checksums-Sha256:\n{checksum_lines}\n"
        ).encode()
    )
    digest = hashlib.sha256(source_index).hexdigest()
    inrelease = (
        "-----BEGIN PGP SIGNED MESSAGE-----\n"
        "Hash: SHA256\n\n"
        "Origin: Debian\n"
        "SHA256:\n"
        f" {digest} {len(source_index)} main/source/Sources.xz\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "synthetic-signature-framing\n"
        "-----END PGP SIGNATURE-----\n"
    ).encode()
    return inrelease, source_index


def _source_proof_record(source_index: bytes) -> robomimic_attribution._PublicProof:
    return robomimic_attribution._PublicProof(
        "source-index",
        "https://snapshot.debian.org/source-index",
        hashlib.sha256(source_index).hexdigest(),
        len(source_index),
        "snapshot.debian.org",
    )


def test_robomimic_signed_snapshot_and_source_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inrelease, source_index = _synthetic_debian_source_proof()
    proof = _source_proof_record(source_index)
    monkeypatch.setattr(robomimic_attribution, "_SOURCES", proof)
    robomimic_attribution._verify_snapshot_source(inrelease, source_index)

    with pytest.raises(ValueError, match="signed source-index pin mismatch"):
        robomimic_attribution._verify_snapshot_source(
            inrelease.replace(proof.sha256.encode(), b"0" * 64), source_index
        )
    with pytest.raises(ValueError, match="signed source-index pin mismatch"):
        robomimic_attribution._verify_snapshot_source(
            inrelease.replace(
                b" main/source/Sources.xz", b" other-main/source/Sources.xz"
            ),
            source_index,
        )

    changed_release, changed_source = _synthetic_debian_source_proof(
        version="0.11.7-3"
    )
    monkeypatch.setattr(
        robomimic_attribution, "_SOURCES", _source_proof_record(changed_source)
    )
    with pytest.raises(ValueError, match="source identity"):
        robomimic_attribution._verify_snapshot_source(
            changed_release, changed_source
        )


def test_robomimic_lock_digest_and_private_proof_mode_fail_closed(
    tmp_path: Path,
) -> None:
    canonical = ROBOMIMIC_LOCK.read_bytes()
    with pytest.raises(ValueError, match="canonical public bytes"):
        robomimic_attribution.verify_public_license_lock(
            canonical + b"\n", _robomimic_proof_directory(tmp_path)
        )

    proof = tmp_path / "public-proof"
    proof.mkdir(mode=0o755)
    proof.chmod(0o755)
    with pytest.raises(ValueError, match="caller-owned and private"):
        robomimic_attribution.verify_public_license_lock(canonical, proof)


def test_robomimic_cached_proof_mismatch_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _robomimic_source_repo(tmp_path)
    proof = _robomimic_proof_directory(tmp_path)
    (proof / robomimic_attribution._INRELEASE.name).write_bytes(b"wrong")

    assert _run_robomimic_tree(repo, proof, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "raw=1 dispositioned=0 unresolved=1" in captured.err
    assert "robomimic public-attribution proof could not be verified" in captured.err


def test_robomimic_independent_copyright_bytes_must_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = _robomimic_proof_directory(tmp_path)
    monkeypatch.setattr(
        robomimic_attribution, "_verify_openpgp_signature", lambda *_args: None
    )
    monkeypatch.setattr(
        robomimic_attribution, "_verify_snapshot_source", lambda *_args: None
    )

    def payload(
        _directory: Path, source: robomimic_attribution._PublicProof
    ) -> bytes:
        if source is robomimic_attribution._FTP_MASTER_COPYRIGHT:
            return b"official-a"
        if source is robomimic_attribution._SOURCES_COPYRIGHT:
            return b"official-b"
        return b"snapshot"

    monkeypatch.setattr(robomimic_attribution, "_proof_payload", payload)
    with pytest.raises(ValueError, match="independent official copyright bytes"):
        robomimic_attribution.verify_public_license_lock(
            ROBOMIMIC_LOCK.read_bytes(), proof
        )


def test_robomimic_openpgp_failure_and_transport_errors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proof = _robomimic_proof_directory(tmp_path)
    monkeypatch.setattr(robomimic_attribution.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="verification tools are unavailable"):
        robomimic_attribution._verify_openpgp_signature(proof)

    marker = "synthetic-sensitive-network-detail"

    def refused(*_args: object, **_kwargs: object) -> None:
        raise PublicDownloadError(marker)

    repo = _robomimic_source_repo(tmp_path)
    second_proof = tmp_path / "second-proof"
    second_proof.mkdir(mode=0o700)
    monkeypatch.setattr(robomimic_attribution, "download_public_https", refused)
    assert _run_robomimic_tree(repo, second_proof, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "raw=1 dispositioned=0 unresolved=1" in captured.err
    assert marker not in captured.out + captured.err


def test_robomimic_invalid_openpgp_signature_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = _robomimic_proof_directory(tmp_path)
    monkeypatch.setattr(
        robomimic_attribution.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        robomimic_attribution, "_proof_payload", lambda *_args: b"key"
    )
    results = iter(
        [
            subprocess.CompletedProcess([], 0, b"", b""),
            subprocess.CompletedProcess([], 0, b"keyring", b""),
            subprocess.CompletedProcess([], 1, b"", b"invalid"),
        ]
    )
    monkeypatch.setattr(
        robomimic_attribution.subprocess,
        "run",
        lambda *_args, **_kwargs: next(results),
    )

    with pytest.raises(ValueError, match="signature verification failed"):
        robomimic_attribution._verify_openpgp_signature(proof)
