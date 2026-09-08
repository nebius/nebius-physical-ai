"""Provide the NCore-only prepare, committed build, gate and development push CLI."""

import json
import os
from pathlib import Path
import tarfile

from image_byte_scan import core as W, prepare as P
from . import artifact, registry
from .diagnostics import phase, run_phase
from .process import ROOT, PYTHON, committed_source, file_sha, public_environment, run, write_json
from .process import committed_npa_imports


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
    parser.add_argument("--keyring", type=Path,
                        help="Locked Debian keyring; defaults to analysis-root/keyring/debian-archive-keyring.gpg")
    parser.add_argument("--bootstrap-source", type=Path, help="Two hash-pinned SkyPilot source files")
    parser.add_argument("--authfile", type=Path, help="Private Docker/containers registry auth file; publish only")
    parser.add_argument("--policy-mode", choices=("ci-regex", "exact-literals"), default="ci-regex")
    parser.add_argument("--literal-inventory", type=Path,
                        help="Owner-only exact private literal inventory; exact-literals mode only")
    return parser


def _inputs(args):
    args.analysis_root = args.analysis_root.absolute()
    _policy_input(args)
    committed_source(args.source_sha)
    from . import gates

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
    _keyring_input(args)


def _keyring_input(args):
    args.keyring = (getattr(args, "keyring", None)
                    or args.analysis_root / "keyring/debian-archive-keyring.gpg").absolute()
    W.require(args.keyring.is_relative_to(args.analysis_root), "keyring_outside_private_root")
    if args.action != "prepare":
        lock = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
        W.require(P.binding(args.keyring)["sha256"] == lock["debian_keyring_sha256"],
                  "locked_debian_keyring_required_before_build_or_gates")


def _policy_input(args):
    if args.policy_mode == "exact-literals":
        W.require(args.literal_inventory is not None, "literal_inventory_required")
        args.literal_inventory = args.literal_inventory.absolute()
        values = W.bound_json(P.binding(args.literal_inventory)).get("literals")
        W.require(isinstance(values, list) and values
                  and all(type(value) is str and value for value in values), "nonempty_literal_inventory_required")
    else:
        W.require(args.literal_inventory is None, "literal_inventory_requires_exact_mode")
        W.C.compile_policy(os.environ.get("CUSTOMER_DENYLIST"), os.environ.get("INFRA_DENYLIST"))


def _prepare(args):
    _prepare_keyring(args.analysis_root, args.keyring)
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


def _prepare_keyring(root, keyring):
    from npa._public_https import download_public_https

    lock = json.loads((ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_bytes())
    packages = [item for item in lock["debian_binaries"] if item["name"] == "debian-archive-keyring"]
    W.require(len(packages) == 1, "locked_keyring_package_required")
    package = packages[0]
    directory = root / "keyring-package"
    directory.mkdir(mode=0o700)
    downloaded = directory / "keyring.deb"
    with downloaded.open("xb") as output:
        download_public_https(package["url"], output, allowed_hosts=frozenset({"snapshot.debian.org"}))
    W.require(file_sha(downloaded) == package["sha256"], "locked_keyring_package_digest")
    archive = directory / "data.tar"
    run(["dpkg-deb", "--fsys-tarfile", str(downloaded)], archive, env=public_environment())
    _extract_keyring(archive, keyring, lock["debian_keyring_sha256"])
    write_json(directory / "receipt.json", {
        "schema": "npa.ncore.public-debian-keyring.v1", "package_sha256": package["sha256"],
        "base_lock_sha256": file_sha(ROOT / "npa/docker/workbench/ncore/base-source-lock.json"),
        "keyring": P.binding(keyring), "source": "hash-locked-public-debian-package",
    })


def _extract_keyring(archive, keyring, digest):
    with tarfile.open(archive) as stream:
        members = [member for member in stream
                   if member.name.removeprefix("./") == "usr/share/keyrings/debian-archive-keyring.gpg"]
        W.require(len(members) == 1 and members[0].isfile(), "public_keyring_member_required")
        raw = stream.extractfile(members[0]).read()
    W.require(W.sha(raw) == digest, "locked_debian_keyring_digest")
    keyring.parent.mkdir(mode=0o700, exist_ok=True)
    P.save_bytes(keyring.parent, keyring.name, raw)


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
    from . import gates

    directory = args.analysis_root / "build"
    directory.mkdir(mode=0o700)
    context_sha = committed_source(args.source_sha)
    image = gates.eligibility(args.source_sha)
    gates.source_guards(directory, args.source_sha)
    argv = ["bash", str(ROOT / "npa/docker/workbench/ncore/build.sh"),
            "--source-sha", args.source_sha, "--image", image,
            "--oci-output", str(directory / "image.oci.tar"),
            "--metadata-file", str(directory / "buildx.json")]
    if args.builder:
        argv.extend(["--builder", args.builder])
    env = public_environment()
    env["TMPDIR"] = str(args.analysis_root)
    run(argv, directory / "build.log", env=env)
    metadata = _build_metadata(directory / "buildx.json")
    digest = W.bound_json(metadata)["containerimage.digest"]
    _, verification = artifact.inspect(directory / "image.oci.tar", digest)
    W.require(committed_source(args.source_sha) == context_sha, "source_changed_during_build")
    write_json(directory / "build.json", {
        "schema": "npa.ncore.committed-oci-build.v1", "source_sha": args.source_sha,
        "context_sha256": context_sha, "image": image, "image_digest": digest,
        "archive_sha256": verification["archive_sha256"], "argv": argv,
        "metadata": metadata,
    })


def _build_metadata(path):
    # Buildx writes mode 0644 despite the caller's umask. Its new output is
    # inside the private build directory; restrict the opened inode before
    # making it an input to any later gate. Never chmod a link or shared inode.
    path, fd, before = W.open_private_fd(path, secret=False)
    try:
        W.require(before.st_nlink == 1, "build_metadata_must_be_unshared")
        digest = W.descriptor_digest(fd)
        os.fchmod(fd, 0o600)
        restricted = os.fstat(fd)
        binding = P.binding(path)
        after = os.fstat(fd)
        selected = path.lstat()
        W.require(binding["sha256"] == digest and after.st_nlink == selected.st_nlink == 1
                  and W.stat_fingerprint(restricted) == W.stat_fingerprint(after)
                  and W.stat_fingerprint(selected) == W.stat_fingerprint(after),
                  "build_metadata_changed")
        return binding
    finally:
        os.close(fd)


def _build_receipt(args):
    from . import gates

    build = W.bound_json(P.binding(args.analysis_root / "build/build.json"))
    W.require(build["schema"] == "npa.ncore.committed-oci-build.v1"
              and build["source_sha"] == args.source_sha
              and build["image"] == gates.eligibility(args.source_sha), "build_receipt_source")
    metadata = W.bound_json(build["metadata"])
    W.require(metadata["containerimage.digest"] == build["image_digest"], "build_metadata_digest")
    return build


def _check_or_publish(args):
    from . import gates

    build = run_phase("build-receipt", _build_receipt, args)
    args.output_dir.mkdir(mode=0o700)
    graph, verification = run_phase("prepublication", gates.verify, args, args.output_dir, build)
    write_json(args.output_dir / "prepublication.json", {
        "status": "pass", "source_sha": args.source_sha,
        "image_digest": build["image_digest"], "archive_sha256": build["archive_sha256"],
        "release_acceptance": False,
    })
    if args.action == "publish":
        transfer = args.output_dir / "transfer"
        transfer.mkdir(mode=0o700)
        run_phase("registry-transfer", registry.transfer, args, transfer, build, graph, verification)


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
        with (phase(args.action), W.cancellation_scope(), W.authorized_roots(args.analysis_root, ROOT),
              committed_npa_imports(args.source_sha)):
            run_phase("inputs", _inputs, args)
            if args.action == "prepare":
                _prepare(args)
            elif args.action == "build":
                _build(args)
            else:
                _check_or_publish(args)
        print("NCore OCI operation passed; release acceptance and quarantine are unchanged")
        return 0
    except (Exception, KeyboardInterrupt):
        # Unexpected library errors can also contain private paths or process output.
        print("NCore OCI operation failed; inspect private evidence")
        return 1
