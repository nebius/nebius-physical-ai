"""Build the separate Ed25519 review verifier with the pinned scanner toolchain."""

from __future__ import annotations

import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_byte_scan import core as W
from image_byte_scan.go_helper import build as B

SCHEMA = "npa.image-review-signature-tools.v1"


def _sources():
    checkout = W._ROOTS.get()[1]
    names = ("review_signature/main.go", "review_signature/main_test.go", "review_signature/go.mod",
             "review_signature_build.py", "go_helper/build.py", "go_helper/LICENSE-GO")
    return {name: B.read_regular(checkout / "npa/scripts/image_byte_scan" / name)
            for name in names}


def _toolchain(output, supplied):
    archive = output / "go.tar.gz"
    data = B.read_regular(supplied)
    B.write_new(archive, data)
    return B.extract_toolchain(archive, B.private_dir(output / "toolchain"))


def _compile(output, go, sources):
    stage, logs = B.private_dir(output / "source"), B.private_dir(output / "logs")
    for name in ("main.go", "main_test.go", "go.mod"):
        B.write_new(stage / name, sources["review_signature/" + name])
    env = B.isolated_environment(output, output / "unused-config")
    env.update(GO111MODULE="on", GOPROXY="off", GOSUMDB="off")
    version = B.run_step([str(go), "version"], "version", stage, env, logs)
    W.require(version.decode().strip() == f"go version go{B.GO_VERSION} linux/amd64",
              "review_toolchain_version")
    raw = B.run_step([str(go), "test", "-count=1", "-json", "."], "tests", stage, env, logs)
    counts = B.verify_native_tests(raw, sources["review_signature/main_test.go"])
    binary = output / "verify-review-signature"
    B.run_step([str(go), "build", "-trimpath", "-buildvcs=false", "-o", str(binary), "."],
               "build", stage, env, logs)
    binary.chmod(0o500)
    for name in ("main.go", "main_test.go", "go.mod"):
        W.require(B.read_regular(stage / name) == sources["review_signature/" + name],
                  "review_staged_source_changed")
    return binary, counts


def _build(output, archive):
    sources = _sources()
    go = _toolchain(output, archive)
    W.require(B.read_regular(go.parent.parent / "LICENSE") == sources["go_helper/LICENSE-GO"],
              "review_toolchain_license")
    B.write_new(output / "LICENSE-GO", sources["go_helper/LICENSE-GO"])
    binary, counts = _compile(output, go, sources)
    W.require(_sources() == sources, "review_signature_source_changed")
    return {"schema_version": SCHEMA,
            "binary": {"path": str(binary), "sha256": W.sha(B.read_regular(binary))},
            "sources": {name: W.sha(data) for name, data in sources.items()},
            "toolchain_archive_sha256": B.GO_ARCHIVE_SHA256,
            "toolchain_version": B.GO_VERSION, "cgo_enabled": False,
            "external_modules": [], "native_tests": counts}


def verified_binary(path):
    """Bind the separately built verifier to its current source and toolchain.

    Args:
        path: Owner-only terminal build receipt beneath the authorized root.
    Returns:
        Exact binary path and digest for sealed execution.
    Raises:
        ScanError: Build evidence or source bindings are inconsistent.
        OSError: A required private input cannot be read.
    """
    raw = W.bound_json({"path": str(path), "sha256": W.sha(B.read_regular(path))})
    W.require(raw.get("schema_version") == SCHEMA, "review_signature_build_schema")
    W.require(raw.get("sources") == {name: W.sha(data) for name, data in _sources().items()},
              "review_signature_build_sources")
    W.require(raw.get("toolchain_archive_sha256") == B.GO_ARCHIVE_SHA256
              and raw.get("toolchain_version") == B.GO_VERSION
              and raw.get("cgo_enabled") is False and raw.get("external_modules") == [],
              "review_signature_toolchain")
    W.require(raw.get("native_tests") == {"passed": 2, "declared": 2, "skipped": 0, "failed": 0},
              "review_signature_tests")
    W.bound_file(raw["binary"])
    return raw["binary"]


def main(argv=None):
    """Prepare a verifier without changing the existing native detector bytes.

    Args:
        argv: Explicit arguments, or the process argument vector.
    Returns:
        Zero only after compilation, real tests and input verification succeed.
    Raises:
        None.
    """
    os.umask(0o077)
    try:
        parser = W.SanitizedArgumentParser(description=__doc__)
        for name in ("analysis-root", "trusted-root", "output-dir"):
            parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--toolchain-archive", type=Path, required=True)
        args = parser.parse_args(argv)
        with B.cancellation_scope(), W.authorized_roots(args.analysis_root, args.trusted_root):
            output, fd = W.create_output(args.output_dir)
            try:
                receipt = _build(output, args.toolchain_archive)
                W.write_private_json(output, "dependency-receipt.json", receipt)
            finally:
                os.close(fd)
    except (*W.INPUT_ERRORS, B.BuildError, B.BuildCancelled):
        print("review signature verifier preparation failed")
        return 1
    print("review signature verifier preparation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
