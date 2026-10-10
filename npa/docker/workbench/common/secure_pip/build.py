"""Build a source-identified pip derivative with fixed vendored dependencies.

Run in a disposable build stage, or in the same RUN that installs the wheel and
removes the work directory. Never export the bootstrap/build environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import SplitResult, urlsplit
import venv
import zipfile

HERE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download_https(url: SplitResult) -> bytes:
    """Read one validated HTTPS origin without following redirects."""
    connection = http.client.HTTPSConnection(url.hostname)
    try:
        target = url.path or "/"
        if url.query:
            target += "?" + url.query
        connection.request("GET", target)
        with connection.getresponse() as response:
            if response.status != 200:
                raise ValueError("Build-input HTTPS response is not 200")
            return response.read()
    finally:
        connection.close()


def fetch(row: dict[str, str], directory: Path) -> Path:
    """Only pinned public build inputs; reject mismatches before use."""
    url = urlsplit(row["url"])
    if (
        url.scheme != "https"
        or url.hostname not in {"files.pythonhosted.org", "codeload.github.com"}
        or url.username is not None
        or url.password is not None
        or url.port not in (None, 443)
        or url.fragment
    ):
        raise ValueError("Unexpected public build-input origin")
    name = row["filename"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Invalid build-input filename")
    target = directory / name
    data = _download_https(url)
    if digest(data) != row["sha256"]:
        raise ValueError(f"Build-input digest mismatch: {name}")
    with target.open("xb") as stream:
        stream.write(data)
    return target


def vendor_requirements(original: str, manifest: dict) -> str:
    """Keep the complete upstream population and replace only declared pins."""
    rows = {row["name"].lower(): row for row in manifest["vendors"]}
    if len(rows) != len(manifest["vendors"]):
        raise ValueError("Duplicate vendor input")
    lines = []
    seen = set()
    for line in original.splitlines():
        name, old = line.strip().split("==")
        key = name.lower()
        row = rows[key]
        replacement = manifest["replacements"].get(key)
        expected = replacement["from"] if replacement else row["version"]
        if old != expected or key in seen:
            raise ValueError(f"Unexpected upstream vendor version: {name}")
        if replacement and row["version"] != replacement["to"]:
            raise ValueError(f"Mismatched replacement version: {name}")
        seen.add(key)
        lines.append(f"{name}=={row['version']} --hash=sha256:{row['sha256']}")
    if seen != rows.keys():
        raise ValueError("Vendor population mismatch")
    return "\n".join(lines) + "\n"


def extract_source(archive: Path, destination: Path, commit: str) -> Path:
    root = f"pip-{commit}"
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != root or ".." in parts:
                raise ValueError("Unexpected upstream archive member")
        stream.extractall(destination, filter="data")
    return destination / root


def run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    # Wheel ZIP member modes otherwise inherit the caller's umask and change
    # the intermediate digest recorded by the final derivative. Only this
    # child uses the fixed mask; task-owned work/output directories stay 0700.
    subprocess.run(argv, cwd=cwd, env=env, check=True, umask=0o022)


def _require_fresh_directories(work: Path, output: Path) -> None:
    """Refuse stale task paths before checking the optional build runtime."""
    if work.exists() or output.exists():
        raise FileExistsError("secure-pip build paths must be fresh")


def build(work: Path, output: Path) -> Path:
    # A stale task path is always a pre-side-effect refusal, including on a
    # host that cannot perform the Python-3.11-only derivative build.
    _require_fresh_directories(work, output)
    if sys.version_info < (3, 11):
        raise RuntimeError("The pinned vendoring build tool requires Python 3.11+")
    manifest = json.loads((HERE / "inputs.json").read_text())
    # Exclusive fresh directories prevent contamination by a prior build.
    work.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    inputs = work / "inputs"
    inputs.mkdir()
    vendors = work / "vendors"
    vendors.mkdir()
    temp = work / "tmp"
    temp.mkdir()
    source_archive = fetch(manifest["source"], inputs)
    bootstrap = fetch(manifest["bootstrap"], inputs)
    vendor_files = {
        row["name"].lower(): fetch(row, vendors) for row in manifest["vendors"]
    }
    source = extract_source(source_archive, work, manifest["source"]["commit"])
    environment = work / "build-venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = str(environment / "bin/python")
    # Do not inherit credentials, alternate package indexes or user pip config.
    env = {
        "PATH": str(environment / "bin") + os.pathsep + os.environ["PATH"],
        "TMPDIR": str(temp),
        "LANG": "C.UTF-8",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_INDEX_URL": "https://pypi.org/simple",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": str(manifest["source_date_epoch"]),
    }
    run(
        [
            python,
            str(bootstrap / "pip"),
            "install",
            "--require-hashes",
            "-r",
            str(HERE / "build-tools.lock"),
        ],
        cwd=work,
        env=env,
    )
    # Build the pure-Python msgpack wheel with the already hash-locked tooling.
    # Explicit --no-build-isolation prevents an unpinned nested backend install.
    offline = dict(
        env, PIP_NO_INDEX="1", PIP_FIND_LINKS=str(vendors), MSGPACK_PUREPYTHON="1"
    )
    run(
        [
            python,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(vendors),
            str(vendor_files["msgpack"]),
        ],
        cwd=work,
        env=offline,
    )
    built_msgpack = list(vendors.glob("msgpack-*-py3-none-any.whl"))
    if len(built_msgpack) != 1:
        raise ValueError("Expected exactly one pure-Python msgpack wheel")
    resolved = copy.deepcopy(manifest)
    msgpack_row = next(row for row in resolved["vendors"] if row["name"] == "msgpack")
    msgpack_row["sha256"] = digest(built_msgpack[0].read_bytes())
    msgpack_row["filename"] = built_msgpack[0].name
    msgpack_row["url"] = "built-from-pinned-sdist"
    vendor_file = source / "src/pip/_vendor/vendor.txt"
    vendor_file.write_text(vendor_requirements(vendor_file.read_text(), resolved))
    shutil.copyfile(
        HERE / "pkg_resources.patch",
        source / "tools/vendoring/patches/pkg_resources.patch",
    )
    version_file = source / "src/pip/__init__.py"
    old = f'__version__ = "{manifest["upstream_version"]}"'
    text = version_file.read_text()
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream pip version source")
    version_file.write_text(
        text.replace(old, f'__version__ = "{manifest["derived_version"]}"')
    )
    offline["PIP_REQUIRE_HASHES"] = "1"
    run([str(environment / "bin/vendoring"), "sync", "-v"], cwd=source, env=offline)
    # Upstream's license collector flattens all setuptools-bundled licenses.
    # Explicitly preserve the actual donor's root notice without that collision.
    with zipfile.ZipFile(vendor_files["setuptools"]) as wheel:
        donor = next(row for row in manifest["vendors"] if row["name"] == "setuptools")
        prefix = f"setuptools-{donor['version']}.dist-info/"
        names = [
            n
            for n in wheel.namelist()
            if n in {prefix + "LICENSE", prefix + "licenses/LICENSE"}
        ]
        if len(names) != 1:
            raise ValueError("Missing unambiguous setuptools distribution license")
        license_bytes = wheel.read(names[0])
    (source / "src/pip/_vendor/pkg_resources/LICENSE.setuptools").write_bytes(
        license_bytes
    )
    provenance = {
        "schema": "npa.secure-pip-derivative.v1",
        "inputs": manifest,
        "built_msgpack": {
            "filename": built_msgpack[0].name,
            "sha256": msgpack_row["sha256"],
            "source_sha256": digest(vendor_files["msgpack"].read_bytes()),
        },
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "build_tools_lock_sha256": digest((HERE / "build-tools.lock").read_bytes()),
        "namespace_patch_sha256": digest((HERE / "pkg_resources.patch").read_bytes()),
        "scope": "Released pip source, upstream vendoring machinery, three pinned vendor replacements; not upstream endorsement or image qualification.",
    }
    (source / "src/pip/NPA_VENDOR_REPAIR.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    run(
        [
            python,
            "-c",
            "from flit_core.buildapi import build_wheel; import sys; build_wheel(sys.argv[1])",
            str(output),
        ],
        cwd=source,
        env=env,
    )
    wheels = list(output.glob("*.whl"))
    if (
        len(wheels) != 1
        or wheels[0].name != f"pip-{manifest['derived_version']}-py3-none-any.whl"
    ):
        raise ValueError("Unexpected derived wheel population")
    receipt = {
        "wheel": wheels[0].name,
        "sha256": digest(wheels[0].read_bytes()),
        "bytes": wheels[0].stat().st_size,
        "provenance": provenance,
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    wheel = build(args.work_dir.absolute(), args.output_dir.absolute())
    print(json.dumps({"wheel": wheel.name, "sha256": digest(wheel.read_bytes())}))


if __name__ == "__main__":
    main()
