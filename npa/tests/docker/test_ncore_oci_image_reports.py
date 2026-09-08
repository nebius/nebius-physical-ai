"""Reject incomplete image reports and authenticate the non-weight Python hook."""

import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import cli, gates  # noqa: E402

HOOK = "opt/venv/lib/python3.12/site-packages/npa-source.pth"


@pytest.mark.parametrize("change", [None, "empty", "missing", "public", "symlink", "wrong-mode"])
def test_explicit_literal_policy_requires_nonempty_private_bound_inventory(tmp_path, change):
    path = tmp_path / "literals.json"
    path.write_text(json.dumps({"literals": [] if change == "empty" else ["private-fixture"]}))
    path.chmod(0o644 if change == "public" else 0o600)
    if change == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    args = SimpleNamespace(policy_mode="ci-regex" if change == "wrong-mode" else "exact-literals",
                           literal_inventory=None if change == "missing" else path)
    with W.authorized_roots(tmp_path, ROOT):
        if change is None:
            cli._policy_input(args)
        else:
            with pytest.raises(ValueError):
                cli._policy_input(args)


@pytest.mark.parametrize("mode", ["ci-regex", "exact-literals"])
def test_byte_authorization_preserves_selected_policy(tmp_path, monkeypatch, mode):
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode=mode,
                           literal_inventory=tmp_path / "literals.json")
    commands = []

    def run(argv, output, **_):
        commands.append(argv)
        if len(commands) == 2:
            target = tmp_path / "bytes"
            target.mkdir()
            (target / "report.json").write_text(json.dumps(dict(complete=True, valid=True, helper_joined=True)))

    monkeypatch.setattr(gates, "run", run)
    gates.byte_scan(args, tmp_path, tmp_path / "image.tar", "sha256:" + "a" * 64, {})
    authorization = commands[0]
    assert authorization[authorization.index("--policy-mode") + 1] == mode
    if mode == "exact-literals":
        assert authorization[authorization.index("--literal-inventory") + 1] == str(args.literal_inventory)
        assert authorization[authorization.index("--literal-matching-policy") + 1] == "exact-substring-v1"
    else:
        assert "--literal-inventory" not in authorization


@pytest.mark.parametrize("change", [None, "weights", "executable", "symlink", "duplicate",
                                    "missing", "incomplete", "history", "payload", "missing-list"])
def test_payload_accepts_only_complete_scan_and_exact_text_import_hook(tmp_path, monkeypatch, change):
    report = dict(format="npa_restricted_payload_scan_v2", scan_complete=True,
                  history_only=False, verdict="clean", entries_scanned=1938,
                  payload_hits=[], history_hits=[], weight_shaped_paths=[HOOK])
    if change == "weights":
        report["weight_shaped_paths"].append("opt/checkpoint.pth")
    elif change == "incomplete":
        report["scan_complete"] = False
    elif change in {"payload", "history"}:
        report[change + "_hits"].append({"path": "vendor-payload"})
    elif change == "missing-list":
        del report["payload_hits"]
    raw = b"import unsafe\n" if change == "executable" else b"/opt/npa/src\n"
    with tarfile.open(tmp_path / "rootfs.tar", "w") as archive:
        for _ in range(0 if change == "missing" else 2 if change == "duplicate" else 1):
            member = tarfile.TarInfo(HOOK)
            if change == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "/elsewhere"
                archive.addfile(member)
            else:
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))

    def run(argv, output, **_):
        assert argv[1].endswith("scan_image_omniverse_payload.py")
        (tmp_path / "payload.json").write_text(json.dumps(report))

    monkeypatch.setattr(gates, "run", run)
    if change is None:
        gates._payload(tmp_path)
    else:
        with pytest.raises(ValueError):
            gates._payload(tmp_path)


@pytest.mark.parametrize("change", [None, "empty-results", "unfixed", "null-results",
                                    "schema", "type", "config", "layers", "metadata",
                                    "fixed-critical", "secret"])
def test_trivy_empty_report_requires_exact_image_binding_and_keeps_security_rejections(
    tmp_path, monkeypatch, change
):
    graph = {"image_config_digest": "sha256:" + "a" * 64,
             "layers": [{"diff_id": "sha256:" + "b" * 64}]}
    report = {"SchemaVersion": 2, "ArtifactType": "container_image",
              "Metadata": {"ImageID": graph["image_config_digest"],
                           "DiffIDs": [graph["layers"][0]["diff_id"]]}}
    if change in {"empty-results", "null-results"}:
        report["Results"] = [] if change == "empty-results" else None
    elif change == "schema":
        report["SchemaVersion"] = 1
    elif change == "type":
        report["ArtifactType"] = "filesystem"
    elif change == "config":
        report["Metadata"]["ImageID"] = "sha256:" + "c" * 64
    elif change == "layers":
        report["Metadata"]["DiffIDs"].append("sha256:" + "c" * 64)
    elif change == "metadata":
        del report["Metadata"]
    elif change in {"fixed-critical", "unfixed"}:
        report["Results"] = [{"Vulnerabilities": [{"Severity": "CRITICAL",
                              "FixedVersion": "2" if change == "fixed-critical" else ""}]}]
    elif change == "secret":
        report["Results"] = [{"Secrets": [{"Severity": "LOW"}]}]
    monkeypatch.setattr(gates, "run", lambda argv, output, **_: output.write_text(json.dumps(report)))
    if change in {None, "empty-results", "unfixed"}:
        gates._security(tmp_path, graph)
    else:
        with pytest.raises(ValueError):
            gates._security(tmp_path, graph)
