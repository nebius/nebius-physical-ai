"""Consistency gates for the npa-foxglove-embed image.

The pinned `@foxglove/embed` version and its npm integrity digest live in
``npa.workbench.foxglove`` and are duplicated (by necessity) into the Dockerfile
build args, the install script defaults, the image version registry, and
pyproject's supported-tools table. These tests fail when any copy drifts, and
assert the parts of the packaging that make the embedded viewer actually work:
CORS + byte ranges on the data path.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS
from npa.workbench.foxglove import (
    FOXGLOVE_EMBED_SDK_INTEGRITY,
    FOXGLOVE_EMBED_SDK_VERSION,
    FOXGLOVE_SERVICE_PORT,
    SDK_FILES,
    sdk_assets_present,
    sdk_tarball_url,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = REPO_ROOT / "npa" / "docker" / "workbench" / "foxglove-embed"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
CADDYFILE = IMAGE_DIR / "Caddyfile"
INSTALL_SCRIPT = IMAGE_DIR / "install-sdk.sh"
SMOKE_SCRIPT = IMAGE_DIR / "smoke.sh"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _arg_default(text: str, name: str) -> str:
    match = re.search(rf"(?m)^ARG\s+{re.escape(name)}=(.+)$", text)
    assert match, f"Dockerfile is missing a default for ARG {name}"
    return match.group(1).strip()


def test_image_files_exist() -> None:
    for path in (DOCKERFILE, CADDYFILE, INSTALL_SCRIPT, SMOKE_SCRIPT):
        assert path.is_file(), f"missing {path}"


def test_dockerfile_pins_match_python_constants() -> None:
    text = _dockerfile()
    assert _arg_default(text, "FOXGLOVE_EMBED_VERSION") == FOXGLOVE_EMBED_SDK_VERSION
    assert _arg_default(text, "FOXGLOVE_EMBED_INTEGRITY") == FOXGLOVE_EMBED_SDK_INTEGRITY


def test_install_script_defaults_match_python_constants() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert re.search(rf'(?m)^VERSION="{re.escape(FOXGLOVE_EMBED_SDK_VERSION)}"$', text)
    assert re.search(rf'(?m)^INTEGRITY="{re.escape(FOXGLOVE_EMBED_SDK_INTEGRITY)}"$', text)
    # The script must verify, not merely download.
    assert "openssl dgst -sha512" in text
    assert "integrity mismatch" in text


def test_install_script_is_used_by_the_dockerfile() -> None:
    text = _dockerfile()
    assert "docker/workbench/foxglove-embed/install-sdk.sh" in text
    assert "--integrity" in text


def test_supported_tool_version_tracks_the_sdk() -> None:
    assert CONTAINER_IMAGE_NAMES["foxglove-embed"] == "npa-foxglove-embed"
    assert SUPPORTED_TOOL_VERSIONS["foxglove-embed"] == FOXGLOVE_EMBED_SDK_VERSION
    pyproject = tomllib.loads((REPO_ROOT / "npa" / "pyproject.toml").read_text(encoding="utf-8"))
    supported = pyproject["tool"]["npa"]["supported-tools"]
    assert supported["foxglove-embed"] == FOXGLOVE_EMBED_SDK_VERSION


def test_packaging_contract_entry() -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "npa" / "docker" / "workbench" / "packaging-contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    entry = contract["images"]["foxglove-embed"]
    assert entry["tier"] == "service"
    assert entry["ports"] == [FOXGLOVE_SERVICE_PORT]
    assert entry["final_user"] == "nobody"


def test_dockerfile_is_a_non_root_service_with_a_probe() -> None:
    text = _dockerfile()
    assert f"EXPOSE {FOXGLOVE_SERVICE_PORT}" in text
    assert re.search(r"(?m)^USER nobody$", text)
    assert "HEALTHCHECK" in text and "/healthz" in text
    # Digest-pinned public bases (docs/security/image-reproducibility.md).
    for match in re.finditer(r"(?m)^FROM\s+(\S+)", text):
        ref = match.group(1)
        if ref.startswith("${"):
            continue
        assert "@sha256:" in ref, f"base image is not digest-pinned: {ref}"


def test_caddyfile_supports_range_and_cors_for_mcap() -> None:
    text = CADDYFILE.read_text(encoding="utf-8")
    assert f":{FOXGLOVE_SERVICE_PORT}" in text
    data_block = text.split("handle /data/*", 1)
    assert len(data_block) == 2, "Caddyfile must handle /data/* explicitly"
    data_conf = data_block[1]
    assert "Access-Control-Allow-Origin" in data_conf
    assert "Range" in data_conf, "the Range preflight header must be allowed"
    assert "Access-Control-Expose-Headers" in data_conf
    # Compression must not apply to the data path or byte ranges break.
    assert "encode" not in data_conf.split("handle {", 1)[0]


def test_data_path_is_not_browsable_or_world_writable() -> None:
    """A reachable service must not enumerate or accept recordings.

    `/data/` is unauthenticated by necessity (the cross-origin viewer cannot send
    credentials), so it must not also list its contents, and its directory must
    not be world-writable.
    """
    caddy = CADDYFILE.read_text(encoding="utf-8")
    data_conf = caddy.split("handle /data/*", 1)[1].split("handle {", 1)[0]
    assert "file_server browse" not in data_conf
    assert "FOXGLOVE_DATA_BROWSE" in data_conf, "listing must be explicit opt-in"

    dockerfile = _dockerfile()
    assert "chmod 1777 /srv/data" not in dockerfile
    assert re.search(r"chmod 0755 /srv/data", dockerfile)
    assert re.search(r"chown -R 65534:65534 /srv/data", dockerfile)


def test_smoke_script_checks_the_capability_not_just_liveness() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    for marker in (
        "/healthz",
        "/sdk/index.js",
        "class FoxgloveViewer",
        "mountFoxgloveViewer",
        "Range: bytes=0-7",
        "access-control-allow-headers",
    ):
        assert marker in text, f"smoke script does not check {marker}"


def test_golden_eval_entry_runs_the_smoke_script() -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "npa" / "src" / "npa" / "smoke" / "golden_evals.yaml").read_text(
            encoding="utf-8"
        )
    )
    entry = manifest["containers"]["foxglove-embed"]
    assert entry["image"] == "npa-foxglove-embed"
    assert entry["golden_eval"]["kind"] == "server-smoke"
    assert "npa-foxglove-smoke.sh" in entry["golden_eval"]["command"]


def test_sdk_tarball_url_and_asset_probe(tmp_path: Path) -> None:
    url = sdk_tarball_url()
    assert url.endswith(f"embed-{FOXGLOVE_EMBED_SDK_VERSION}.tgz")
    assert url.startswith("https://registry.npmjs.org/@foxglove/embed/")
    assert sdk_tarball_url(registry="https://npm.internal.example/").startswith(
        "https://npm.internal.example/@foxglove/embed/"
    )

    ready, reason = sdk_assets_present(tmp_path / "missing")
    assert not ready and "not installed" in reason

    assets = tmp_path / "sdk"
    assets.mkdir()
    (assets / "index.js").write_text("export {};", encoding="utf-8")
    ready, reason = sdk_assets_present(assets)
    assert not ready and "incomplete" in reason

    for name in SDK_FILES:
        (assets / name).write_text("export {};", encoding="utf-8")
    ready, reason = sdk_assets_present(assets)
    assert ready and reason == ""
