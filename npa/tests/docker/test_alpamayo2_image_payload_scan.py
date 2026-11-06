from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "scan_image_alpamayo2_payload.py"
)
SPEC = importlib.util.spec_from_file_location("scan_image_alpamayo2_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def test_dockerfile_parses_torch_arch_flags_as_tokens() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "alpamayo2-super"
        / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert "_cuda_getArchFlags().split()" in dockerfile
    assert "{'sm_90', 'sm_100', 'sm_120'} <= flags" in dockerfile
    # The prepared src/npa context carries the build hook and both catalog tiers.
    assert "COPY src/npa /opt/npa-src/src/npa" in dockerfile
    assert "COPY pyproject.toml README.md /opt/npa-src/" in dockerfile
    assert "chown -R ubuntu:ubuntu /opt/alpamayo2" not in dockerfile
    assert "'uvicorn==0.41.0' /opt/npa-src" in dockerfile
    assert 'npa.version="0.1.0-cu128-r3"' in dockerfile
    assert '"-m", "npa.workbench.alpamayo2_super.healthcheck"' in dockerfile
    assert "urllib.request.urlopen" not in dockerfile
    assert "git linux-libc-dev openssh-server" in dockerfile
    assert "rm -rf /opt/nvidia/nsight-compute" in dockerfile
    assert "test ! -e /opt/nvidia/nsight-compute" in dockerfile


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _saved_image(
    tmp_path: Path, members: dict[str, bytes], *, config: bytes = b"{}"
) -> Path:
    layer = _tar(tmp_path / "layer.tar", members)
    return _tar(
        tmp_path / "image.tar",
        {
            "manifest.json": json.dumps(
                [
                    {
                        "Config": "config.json",
                        "RepoTags": ["test:latest"],
                        "Layers": ["layer.tar"],
                    }
                ]
            ).encode(),
            "config.json": config,
            "layer.tar": layer.read_bytes(),
        },
    )


def test_clean_source_and_empty_cache_pass(tmp_path: Path) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {"opt/alpamayo2/LICENSE": b"Apache License 2.0"})
    )
    assert layers == 1
    assert findings == []


def test_symlink_under_runtime_tree_is_not_read_as_file(tmp_path: Path) -> None:
    layer_path = tmp_path / "layer.tar"
    with tarfile.open(layer_path, "w") as archive:
        link = tarfile.TarInfo("opt/alpamayo2/.venv/bin/python")
        link.type = tarfile.SYMTYPE
        link.linkname = "/usr/bin/python3.12"
        archive.addfile(link)
    assert scanner.scan_layer(layer_path, layer="layer.tar") == []


def test_weight_and_dataset_payload_fail_even_in_lower_layer(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {
                "workspace/.cache/huggingface/hub/models--nvidia/model.safetensors": b"x",
                "opt/alpamayo2/notebooks/clip_ids.parquet": b"x",
            },
        )
    )
    assert {finding.kind for finding in findings} == {
        "model_weight",
        "populated_hf_cache",
        "physical_ai_av_dataset_payload",
    }


def test_secret_content_in_source_fails(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path, {"opt/alpamayo2/config": b"hf_abcdefghijklmnopqrstuvwxyz"}
        )
    )
    assert {finding.kind for finding in findings} == {"credential_content"}


def test_dependency_fixtures_do_not_impersonate_operator_payload(
    tmp_path: Path,
) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {
                "opt/alpamayo2/.venv/site-packages/pkg/testing.py": b"hf_abcdefghijklmnopqrstuvwxyz",
                "opt/alpamayo2/.venv/site-packages/pyarrow/tests/example.parquet": b"x",
            },
        )
    )
    assert findings == []


def test_secret_in_image_environment_fails(tmp_path: Path) -> None:
    findings, _ = scanner.scan_saved_image(
        _saved_image(
            tmp_path,
            {},
            config=b'{"config":{"Env":["HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"]}}',
        )
    )
    assert {finding.kind for finding in findings} == {"credential_content"}


# Fixed, obviously fake PEM block: not a key, opens nothing.
SYNTHETIC_KEY = (
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----\n"
    b"NOT-A-REAL-KEY-SYNTHETIC-CONTROL-FIXTURE-ONLY\n"
    b"-----END OPENSSH PRIVATE KEY-----\n"
)


@pytest.mark.parametrize(
    "member",
    [
        "root/.ssh/id_rsa",
        "home/ubuntu/.ssh/id_ed25519",
        "root/.netrc",
        # Outside opt/alpamayo2 and opt/npa-src, so the pre-existing content
        # check never read it: that check only ran on application content.
        "opt/vendor/secrets/deploy_key",
    ],
)
def test_scan_rejects_credentials_outside_application_content(
    tmp_path: Path, member: str
) -> None:
    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {member: SYNTHETIC_KEY})
    )
    assert layers == 1
    assert findings, f"{member} must be reported"


def test_scan_still_accepts_application_content_with_no_credential(
    tmp_path: Path,
) -> None:
    # Negative control for the application-content path. Note it cannot observe
    # the widened branch, which only runs when a member is *not* application
    # content; the binary controls below are the ones that cover that.
    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {"opt/alpamayo2/README.md": b"Alpamayo 2\n"})
    )
    assert layers == 1
    assert findings == []


# How OpenSSH stores the header it parses: a NUL-terminated constant followed
# immediately by the END constant. Committed rather than read from the host,
# because a scanner test that silently skips when /usr/bin/ssh is absent is
# close to no test. The layout is copied from the real binaries.
OPENSSH_PARSER_CONSTANTS = (
    b"\x7fELF\x02\x01\x01\x00"
    + b"\x00" * 56
    + b"usage: ssh [-46AaCfGgKk]\x00"
    + b"-----BEGIN "
    + b"OPENSSH PRIVATE KEY-----\n\x00\x00\x00\x00"
    b"-----END OPENSSH PRIVATE KEY-----\n\x00\x00\x00\x00\x00\x00"
    b"%s: Could not resolve hostname %.100s: %s\x00"
)


@pytest.mark.parametrize(
    "member",
    ["usr/bin/ssh", "usr/sbin/sshd", "usr/bin/ssh-keygen"],
)
def test_scan_accepts_a_system_openssh_binary(tmp_path: Path, member: str) -> None:
    """A stock openssh binary is not a leaked key.

    alpamayo2-super's Dockerfile installs openssh-server, so this file is in
    the image the scanner runs against in build.sh. These paths are outside the
    application tree, which is exactly what puts them in the widened branch.
    """

    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {member: OPENSSH_PARSER_CONSTANTS})
    )
    assert layers == 1
    assert findings == [], f"{member} reported as {[f.kind for f in findings]}"


@pytest.mark.skipif(
    not Path("/usr/bin/ssh").is_file(), reason="no openssh binary on this host"
)
def test_scan_accepts_the_real_openssh_binary_when_present(tmp_path: Path) -> None:
    # The committed fixture above encodes what the real layout looks like. This
    # checks that belief against the actual bytes wherever they exist, so the
    # fixture cannot drift into testing only itself.
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, {"usr/bin/ssh": Path("/usr/bin/ssh").read_bytes()})
    )
    assert findings == [], f"reported as {[f.kind for f in findings]}"


# Credentials the legacy SECRET_CONTENT list does not describe. It covers only
# RSA/OPENSSH private keys, AKIA ids and hf_ tokens, so each of these was
# rejected under a non-application path and accepted under an application one,
# for byte-identical content.
BEYOND_LEGACY_RULES = {
    "ec_private_key": b"-----BEGIN "
    + b"EC PRIVATE KEY-----\n"
    + b"c3ludGhldGljLWZpeHR1cmUtbm90LWEtcmVhbC1rZXk" * 3
    + b"\n-----END EC PRIVATE KEY-----\n",
    "encrypted_private_key": b"-----BEGIN "
    + b"ENCRYPTED PRIVATE KEY-----\n"
    + b"c3ludGhldGljLWZpeHR1cmUtbm90LWEtcmVhbC1rZXk" * 3
    + b"\n-----END ENCRYPTED PRIVATE KEY-----\n",
    "credential_assignment": b"aws_secret_access_key = c3ludGhldGljLW5vdC1yZWFs\n",
}

# The two families the branch distinguishes. _is_application_content is true
# for the first pair and false for the second.
APPLICATION_PATHS = ("opt/alpamayo2/config/settings.py", "opt/npa-src/npa/tool.py")
NON_APPLICATION_PATHS = ("opt/vendor/secrets/deploy_key", "srv/data/notes.txt")


@pytest.mark.parametrize("payload_kind", sorted(BEYOND_LEGACY_RULES))
@pytest.mark.parametrize("member", APPLICATION_PATHS + NON_APPLICATION_PATHS)
def test_shared_rules_apply_to_both_path_families(
    tmp_path: Path, member: str, payload_kind: str
) -> None:
    """Identical bytes must get the same verdict in either tree.

    The shared rules used to be skipped for application content, on the
    reasoning that the legacy list already covered it. The legacy list is
    narrower, so the two families disagreed. Which directory a key sits in is
    not a property of the key.
    """

    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {member: BEYOND_LEGACY_RULES[payload_kind]})
    )
    assert layers == 1
    assert findings, f"{payload_kind} at {member} must be reported"


@pytest.mark.parametrize("member", APPLICATION_PATHS + NON_APPLICATION_PATHS)
def test_ordinary_content_is_accepted_in_both_path_families(
    tmp_path: Path, member: str
) -> None:
    # Paired negative control, so parity is not reached by rejecting both.
    findings, layers = scanner.scan_saved_image(
        _saved_image(tmp_path, {member: b"# ordinary source file\nvalue = 1\n"})
    )
    assert layers == 1
    assert findings == []


def test_scan_still_reports_a_key_outside_application_content(
    tmp_path: Path,
) -> None:
    # Paired positive control: the false-positive fix must not be a blanket
    # weakening of the widened branch it was applied to.
    key = (
        b"-----BEGIN "
        + b"OPENSSH PRIVATE KEY-----\n"
        + b"c3ludGhldGljLWZpeHR1cmUtbm90LWEtcmVhbC1rZXk" * 3
        + b"\n-----END OPENSSH PRIVATE KEY-----\n"
    )
    findings, _ = scanner.scan_saved_image(
        _saved_image(tmp_path, {"opt/vendor/secrets/deploy_key": key})
    )
    assert findings, "a real key outside the application tree must still be reported"


def test_current_npa_source_review_bindings_are_content_exact() -> None:
    """A changed source member must be re-reviewed rather than silently accepted."""

    root = Path(__file__).resolve().parents[2]
    reviews = scanner.load_reviews()
    current_members = {
        "src/npa/cli/fiftyone/__init__.py": "8a5783505e3778f75669cad2592cac97a53679faadacf4710191d7e2bf3c16f7",
        "src/npa/cli/genesis/__init__.py": "f329e5c06705fe9b3cdd4b0a697609506d530cca14617096ab2490b17e24f302",
        "src/npa/cli/groot/__init__.py": "7b76b736c913693f0207ad60724a7d85c1984bdff5c46b3b099cba22b667df7b",
        "src/npa/cli/isaac_lab/__init__.py": "dcaccb56118e7dae83bee9e9887573d06ffde586effc65f1cd9fecc94aa603cb",
        "src/npa/cli/main.py": "4a607aa7ed98e2210d16ed58867fb98ac98c1903dfc4959a9cb360792c83bde2",
        "src/npa/cli/workbench/health.py": "937cf083dffb784a6b866051ffdc0415ab14a01b4d06f1d6fed50f3021e7547b",
        "src/npa/cli/workbench/lerobot.py": "2e6c2e467e395ebf566f75642f012a768d2ad806689a7615c962a531318170a5",
        "src/npa/cli/workbench/robocasa/deploy.py": "bf2ffee8ca78ba05d125677942d76a40258835b44466919a2b783aa73b746b90",
        "src/npa/cli/workbench/sim2real.py": "a37f80fbd26c66e10eeaa2a11dcc1f04bb360470ad7f12820455b4108f877964",
        "src/npa/cli/workbench/workflow/__init__.py": "6730b4c5750474184faf0e4508fb4beefc34013f206c56ff41d47d09fe3d9d51",
        "src/npa/clients/storage.py": "abbc54e878eccfca0d85391b316be85642b6a417ae25d95ae222c4d0cb4dedd1",
        "src/npa/orchestration/npa_workflow/run_state.py": "866b4b5a1a6e23ea19b6e7a910ffc1f29a3978a04d6cc15762b726fd231abd98",
        "src/npa/orchestration/npa_workflow/sim2real_preflight.py": "b9e4436bc9f4ff1557da3bf60975fd210a1e55c5b206198bc9e51ac5ddd66cd3",
        "src/npa/orchestration/npa_workflow/src_staging.py": "e59bbbd5336a9570e8d67342b49df971b0f106fb3d0cb2296156f9dfa9c54d24",
        "src/npa/orchestration/npa_workflow/submit_matrix.py": "8e3fe7d1db5c9b952a1eacb533d0f456784491371f86c6ecf9e962cde9d0d58f",
        "src/npa/orchestration/skypilot/workflow_state.py": "b23b811dd9b83881f968ef7cd2f5643d3dd27d9b7f9d584f7474922c484112dd",
        "src/npa/smoke/golden_evals.yaml": "5c65176a0a328303e8abb2a53c1842c2fe521d78e52f8973343f1ae9fd69994c",
        "src/npa/workflows/rerun_serve.py": "330d9d817fd2c2d50b2c50e358c5b6a218ebcae130051f73a4c13fe6c51fe117",
        "src/npa/workflows/sim_to_real.py": "bcd3ddc1bbd0696cd3da3a45e173a8d9410837645e88d50b9d9051e2dd61de5d",
    }
    for relative_path, digest in current_members.items():
        kind, receipt = scanner.reviewed_content_credential(
            io.BytesIO((root / relative_path).read_bytes()), reviews
        )
        assert kind is None
        assert receipt and receipt["sha256"] == digest

    # workflow_build stages this root template at the package location scanned
    # in the image. Bind the source template itself, not generated output.
    kind, receipt = scanner.reviewed_content_credential(
        io.BytesIO(
            (root.parent / "workflows/main/nurec-reconstruct.yaml").read_bytes()
        ),
        reviews,
    )
    assert kind is None
    assert receipt and receipt["sha256"] == (
        "366f6ef22133c50a1124aee2c98d9d57984a50f8205e7bff3a62757f89e8e1c4"
    )
