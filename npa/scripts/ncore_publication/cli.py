"""Provide the NCore-only prepare, committed build, gate and development push CLI."""

import json
import os
from pathlib import Path

from image_byte_scan import core as W, prepare as P
from . import artifact, gates, registry
from .process import ROOT, PYTHON, committed_source, public_environment, run, write_json


def _parser():
    parser = W.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "check", "publish"))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New check/publish evidence directory")
    parser.add_argument("--builder", help="Buildx docker-container builder name")
    parser.add_argument("--annex", type=Path, help="Delivered corresponding-source annex")
    parser.add_argument("--native-source", type=Path, help="Verified native corresponding-source inputs")
    parser.add_argument("--metadata", type=Path, help="Separate signed Debian build metadata")
    parser.add_argument("--keyring", type=Path, default=Path("/usr/share/keyrings/debian-archive-keyring.gpg"))
    parser.add_argument("--bootstrap-source", type=Path, help="Two hash-pinned SkyPilot source files")
    parser.add_argument("--authfile", type=Path, help="Private Docker/containers registry auth file; publish only")
    parser.add_argument("--policy-mode", choices=("ci-regex", "exact-literals"), default="ci-regex")
    parser.add_argument("--literal-inventory", type=Path,
                        help="Owner-only exact private literal inventory; exact-literals mode only")
    return parser


def _inputs(args):
    args.analysis_root = args.analysis_root.absolute()
    W.C.compile_policy(os.environ.get("CUSTOMER_DENYLIST"), os.environ.get("INFRA_DENYLIST"))
    _policy_input(args)
    committed_source(args.source_sha)
    gates.eligibility(args.source_sha)
    if args.action in {"check", "publish"}:
        for name in ("output_dir", "annex", "native_source", "metadata", "bootstrap_source"):
            path = getattr(args, name)
            W.require(path is not None, "missing_gate_input")
            path = path.absolute()
            W.require(path.is_relative_to(args.analysis_root), "gate_input_outside_private_root")
            fd = W.directory_fd(path.parent if name == "output_dir" else path)
            os.close(fd)
            setattr(args, name, path)
        W.require(not args.metadata.is_relative_to(args.annex)
                  and not args.annex.is_relative_to(args.metadata)
                  and not args.metadata.is_relative_to(args.native_source)
                  and not args.native_source.is_relative_to(args.metadata), "metadata_must_be_separate")
    if args.action == "publish":
        W.require(args.authfile is not None, "private_registry_authfile_required")
        args.authfile = args.authfile.absolute()
        P.binding(args.authfile)


def _policy_input(args):
    if args.policy_mode == "exact-literals":
        W.require(args.literal_inventory is not None, "literal_inventory_required")
        args.literal_inventory = args.literal_inventory.absolute()
        values = W.bound_json(P.binding(args.literal_inventory)).get("literals")
        W.require(isinstance(values, list) and values
                  and all(type(value) is str and value for value in values), "nonempty_literal_inventory_required")
    else:
        W.require(args.literal_inventory is None, "literal_inventory_requires_exact_mode")


def _prepare(args):
    common = ["--analysis-root", str(args.analysis_root), "--trusted-root", str(ROOT)]
    run([str(PYTHON), "npa/scripts/image_byte_scan/go_helper/build.py", *common,
         "--output-dir", str(args.analysis_root / "tools")], args.analysis_root / "tools.log")
    run([str(PYTHON), "npa/scripts/image_byte_scan/prepare.py", "dependencies", *common,
         "--output-dir", str(args.analysis_root / "native")], args.analysis_root / "native.log")
    run([str(PYTHON), "npa/scripts/image_byte_scan/real_helper_checks.py", *common,
         "--tools-receipt", str(args.analysis_root / "tools/dependency-receipt.json"),
         "--native-receipt", str(args.analysis_root / "native/dependencies.json"),
         "--output-dir", str(args.analysis_root / "native-checks")], args.analysis_root / "native-checks.log")
    _source_inputs(args.analysis_root)


def _source_inputs(root):
    from npa._public_https import download_public_https
    from .process import file_sha

    env = public_environment()
    env["TMPDIR"] = str(root)
    run([str(PYTHON), "npa/docker/workbench/ncore/base_sources.py", "assemble",
         "--lock", str(ROOT / "npa/docker/workbench/ncore/base-source-lock.json"),
         "--annex", str(root / "sources"), "--native", str(root / "sources"),
         "--metadata", str(root / "metadata")], root / "source-inputs.log", env=env)
    upstream = root / "bootstrap-source"
    upstream.mkdir(mode=0o700)
    lock = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    for item in lock["bootstrap"]["upstream"]:
        target = upstream / item["url"].rsplit("/", 1)[1]
        with target.open("xb") as output:
            download_public_https(item["url"], output, allowed_hosts=frozenset({"raw.githubusercontent.com"}))
        W.require(file_sha(target) == item["sha256"], "bootstrap_upstream_sha256")


def _build(args):
    directory = args.analysis_root / "build"
    directory.mkdir(mode=0o700)
    context_sha = committed_source(args.source_sha)
    image = gates.eligibility(args.source_sha)
    gates.source_guards(directory)
    argv = ["bash", str(ROOT / "npa/docker/workbench/ncore/build.sh"),
            "--source-sha", args.source_sha, "--image", image,
            "--oci-output", str(directory / "image.oci.tar"),
            "--metadata-file", str(directory / "buildx.json")]
    if args.builder:
        argv.extend(["--builder", args.builder])
    env = public_environment()
    env["TMPDIR"] = str(args.analysis_root)
    run(argv, directory / "build.log", env=env)
    digest = json.loads((directory / "buildx.json").read_bytes())["containerimage.digest"]
    _, verification = artifact.inspect(directory / "image.oci.tar", digest)
    W.require(committed_source(args.source_sha) == context_sha, "source_changed_during_build")
    write_json(directory / "build.json", {
        "schema": "npa.ncore.committed-oci-build.v1", "source_sha": args.source_sha,
        "context_sha256": context_sha, "image": image, "image_digest": digest,
        "archive_sha256": verification["archive_sha256"], "argv": argv,
        "metadata": P.binding(directory / "buildx.json"),
    })


def _check_or_publish(args):
    build = W.bound_json(P.binding(args.analysis_root / "build/build.json"))
    W.require(build["schema"] == "npa.ncore.committed-oci-build.v1"
              and build["source_sha"] == args.source_sha
              and build["image"] == gates.eligibility(args.source_sha), "build_receipt_source")
    metadata = W.bound_json(build["metadata"])
    W.require(metadata["containerimage.digest"] == build["image_digest"], "build_metadata_digest")
    args.output_dir.mkdir(mode=0o700)
    graph, verification = gates.verify(args, args.output_dir, build)
    write_json(args.output_dir / "prepublication.json", {
        "status": "pass", "source_sha": args.source_sha,
        "image_digest": build["image_digest"], "archive_sha256": build["archive_sha256"],
        "release_acceptance": False,
    })
    if args.action == "publish":
        transfer = args.output_dir / "transfer"
        transfer.mkdir(mode=0o700)
        registry.transfer(args, transfer, build, graph, verification)


def main(argv=None):
    """Execute the scoped command and print no private scanner/process output.

    Args:
        argv: Optional CLI argument list.
    Returns:
        Zero only when the requested operation and every required gate passed.
    Raises:
        SystemExit: Argument parsing terminates for help or invalid arguments.
    """
    os.umask(0o077)
    try:
        args = _parser().parse_args(argv)
        with W.cancellation_scope(), W.authorized_roots(args.analysis_root, ROOT):
            _inputs(args)
            if args.action == "prepare":
                _prepare(args)
            elif args.action == "build":
                _build(args)
            else:
                _check_or_publish(args)
        print("NCore OCI operation passed; release acceptance and quarantine are unchanged")
        return 0
    except W.INPUT_ERRORS:
        print("NCore OCI operation failed; inspect private evidence")
        return 1
