#!/usr/bin/env python3
"""Prove mechanically that a BUILT image carries no NVIDIA Omniverse Kit payload.

This is the check the whole redistribution reclassification rests on. Reading a
Dockerfile is not enough: the claim is about bytes in layers, so this inspects the built
image's filesystem and its layer history.

    npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py \
        cr.eu-north1.nebius.cloud/<registry-id>/npa-isaac-lab:2.3.2.post1

    # or from a local docker save tarball, with no registry access
    docker save npa-isaac-lab:rc1 -o /tmp/img.tar
    npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py --tarball /tmp/img.tar

Why it keys on payload signatures rather than the string "isaac"
---------------------------------------------------------------
The re-architected images deliberately keep a ``/isaac-sim/python.sh`` shim, because ~30
call sites in this repo already invoke Isaac through that path and pods override
ENTRYPOINT, so the shim is the only reliable bootstrap trigger. A naive
``tar -tf | grep isaac`` therefore reports a hit on a 40-line shell script of ours, and
"grep found nothing" is not available as a proof strategy.

Instead this looks for the things only a real Kit install produces - ``libcarb``, Kit's
``kernel/``, ``omni.*`` extension directories, ``extscache``, ``.kit`` app files,
``site-packages/isaacsim`` - and pairs that with a short, explicit ALLOWLIST of the paths
we do ship. Anything matching a signature and not on the allowlist fails the scan, so an
unexpected path fails closed rather than being waved through.

The classifier is unit-tested offline against synthetic listings in
npa/tests/docker/test_image_payload_scan.py, so its logic is covered in CI without a
registry.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

# Path signatures that only a real Omniverse Kit / Isaac Sim install produces.
PAYLOAD_SIGNATURES: tuple[tuple[str, str], ...] = (
    (r"(?i)site-packages/isaacsim/", "the isaacsim wheel's installed package tree"),
    (r"(?i)site-packages/isaaclab/", "the isaaclab wheel's installed package tree"),
    (r"(?i)(^|/)isaac-sim/(kit|exts|extscache|extsPhysics|apps)/", "an Isaac Sim install tree"),
    (r"(?i)(^|/)kit/kernel/", "Omniverse Kit's kernel"),
    (r"(?i)libcarb", "carb, Omniverse Kit's core runtime library"),
    (r"(?i)libomni[a-z0-9_.]*\.so", "an Omniverse Kit shared library"),
    (r"(?i)(^|/)omni\.[a-z0-9_.]+(-[^/]*)?/", "an Omniverse Kit extension directory"),
    (r"(?i)extscache", "Omniverse Kit's extension cache"),
    (r"(?i)\.kit$", "a Kit app configuration file"),
    (r"(?i)omniverse", "an Omniverse-branded path"),
    (r"(?i)isaac.?sim.?assets", "Isaac Sim's bundled assets"),
)

# Gated model weights are a separate licence axis from Omniverse Kit, and the workbench
# rule is the same for both: never baked, always fetched at run time by the operator with
# their own token. gear_sonic's weights sit behind git LFS, so a plain checkout leaves
# ~130-byte pointer stubs; a pointer is a reference the operator resolves, not a weight.
# This scanner only sees a tar listing (names, not contents), so it reports weight-shaped
# paths for a human to eyeball rather than failing on them - the authoritative
# content-based check runs inside the image build, where the bytes are available.
WEIGHT_SUFFIXES: tuple[str, ...] = (".pt", ".pth", ".safetensors", ".ckpt", ".onnx", ".gguf")

# Paths we DO ship that a loose name filter would flag. Deliberately short and exact: an
# unlisted path that matches a signature fails the scan.
ALLOWED_EXACT: frozenset[str] = frozenset(
    {
        "isaac-sim/python.sh",
        "opt/npa/bin/isaac-python",
        "opt/npa/bin/isaac-bootstrap",
    }
)
ALLOWED_PREFIXES: tuple[str, ...] = (
    "opt/npa/docker/workbench/common/",
    "opt/npa/docker/workbench/isaac-lab/",
    "opt/npa/docker/workbench/sonic/",
    "opt/npa/docker/workbench/groot/",
)
# Directory entries that are legitimately present but empty (mount points, workdirs).
ALLOWED_DIRS: frozenset[str] = frozenset(
    {"isaac-sim", "opt/isaac-cache", "opt/isaac-cache/v", "workspace/isaaclab", "opt/isaac-lab"}
)

# Layer commands that would mean Isaac was installed during the build. Kept in step with
# packaging-contract.yaml's omniverse_bake_patterns, but applied to the image's own
# recorded history rather than to a Dockerfile, so it also catches an image built from a
# Dockerfile nobody reviewed.
HISTORY_BAKE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?i)pip[^\n]*install[^\n]*\bisaacsim\b", "a build layer pip-installed isaacsim"),
    (r"(?i)pip[^\n]*install[^\n]*\bisaaclab\b", "a build layer pip-installed isaaclab"),
    (r"(?i)nvcr\.io/nvidia/(isaac-lab|isaac-sim|omniverse)", "a layer references an NVIDIA vendor image"),
    (
        r"(?i)\b(isaac-bootstrap|isaac_bootstrap\.sh)\s+(ensure|warm|verify)\b",
        "a build layer ran the runtime bootstrap, materialising Isaac into the image",
    ),
    (
        r"(?i)(OMNI_KIT_ACCEPT_EULA|ISAACSIM_ACCEPT_EULA|PRIVACY_CONSENT)=",
        "a layer bakes EULA acceptance",
    ),
)


def _normalize(member: str) -> str:
    """Normalise a tar member path for matching (no leading ./ or /)."""
    return member.lstrip("./").lstrip("/")


def is_allowed(path: str) -> bool:
    normalized = _normalize(path)
    if normalized in ALLOWED_EXACT or normalized.rstrip("/") in ALLOWED_DIRS:
        return True
    return any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def classify_path(path: str) -> str | None:
    """Return why ``path`` looks like Kit payload, or ``None`` if it is fine."""
    if is_allowed(path):
        return None
    normalized = _normalize(path)
    for pattern, why in PAYLOAD_SIGNATURES:
        if re.search(pattern, normalized):
            return why
    return None


def _history_instructions(command: str) -> str:
    """Strip comment lines from a recorded layer command.

    buildkit records the *whole* RUN, heredocs included, so a Python comment inside an
    inlined script ends up in the image history. That bit for real: sonic's build-time
    check contains the line

        # happens on GPU (isaac-bootstrap verify / the golden eval).

    which made the scanner report "a build layer ran the runtime bootstrap" against an
    image that had done no such thing. A false positive here is not harmless - it would
    block a legitimate reclassification, and the obvious "fix" is to loosen the pattern.
    Same prose-versus-instruction distinction the packaging guard makes on Dockerfiles.
    """
    return "\n".join(
        line
        for line in command.splitlines()
        if not line.lstrip().startswith("#")
    )


def classify_history(command: str) -> str | None:
    """Return why a layer command looks like a build-time Isaac install, or ``None``."""
    instructions = _history_instructions(command)
    for pattern, why in HISTORY_BAKE_PATTERNS:
        if re.search(pattern, instructions):
            return why
    return None


@dataclass
class ScanReport:
    image: str
    source: str
    digest: str | None = None
    entries_scanned: int = 0
    allowlisted_hits: list[str] = field(default_factory=list)
    payload_hits: list[dict[str, str]] = field(default_factory=list)
    history_hits: list[dict[str, str]] = field(default_factory=list)
    weight_shaped_paths: list[str] = field(default_factory=list)
    #: True when only the layer history was inspected. Recorded in the JSON report so a
    #: consumer can never mistake a fast gate result for a full-filesystem proof.
    history_only: bool = False

    @property
    def clean(self) -> bool:
        return not self.payload_hits and not self.history_hits

    def to_dict(self) -> dict:
        return {
            "format": "npa_omniverse_payload_scan_v1",
            "image": self.image,
            "source": self.source,
            "digest": self.digest,
            # Recorded so a consumer can never mistake a fast pre-publish gate result for
            # a full-filesystem proof: a history-only "clean" says the build ran no Isaac
            # install, not that the image ships no Isaac bytes.
            "history_only": self.history_only,
            "entries_scanned": self.entries_scanned,
            "verdict": "clean" if self.clean else "omniverse-payload-detected",
            "payload_hits": self.payload_hits,
            "history_hits": self.history_hits,
            "allowlisted_paths_present": sorted(self.allowlisted_hits),
            "weight_shaped_paths": sorted(self.weight_shaped_paths),
        }


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found on PATH")
    return path


def _iter_crane_export(image: str):
    """Stream the flattened filesystem of a remote image, member by member."""
    crane = _require("crane")
    process = subprocess.Popen(  # noqa: S603 - fixed argv
        [crane, "export", image, "-"], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    try:
        # r|* streams without seeking, so a multi-GB image never lands on disk.
        with tarfile.open(fileobj=process.stdout, mode="r|*") as archive:
            for member in archive:
                yield member.name + ("/" if member.isdir() else "")
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait()


def _iter_tarball(tarball: Path):
    """Yield member names from a `docker save` tarball, including inside layer blobs."""
    with tarfile.open(tarball, mode="r") as archive:
        for member in archive:
            name = member.name
            if not (
                name.endswith(("/layer.tar", ".tar"))
                or name.startswith("blobs/")
                or "/blobs/" in name
            ):
                yield name + ("/" if member.isdir() else "")
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                with tarfile.open(fileobj=handle, mode="r|*") as layer:
                    for entry in layer:
                        yield entry.name + ("/" if entry.isdir() else "")
            except tarfile.TarError:
                # Not a tar (config JSON, manifest); the outer name was already yielded.
                yield name


def _image_history(image: str) -> tuple[list[str], str | None]:
    crane = _require("crane")
    config = subprocess.run(  # noqa: S603 - fixed argv
        [crane, "config", image], capture_output=True, text=True, check=False
    )
    if config.returncode != 0:
        return [], None
    payload = json.loads(config.stdout)
    commands = [
        entry.get("created_by", "")
        for entry in payload.get("history", [])
        if entry.get("created_by")
    ]
    digest = subprocess.run(  # noqa: S603 - fixed argv
        [crane, "digest", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        check=False,
    )
    return commands, (digest.stdout.strip() or None)


def scan(
    image: str | None,
    tarball: Path | None,
    *,
    max_report: int = 40,
    history_only: bool = False,
) -> ScanReport:
    """Scan an image for Omniverse payload.

    ``history_only`` skips the filesystem walk and inspects only the layer history, which
    needs the config blob (a few KB) instead of streaming the whole image (tens of GB).
    That makes it usable as a pre-publish gate in CI, where streaming 69 GB is not.

    It is strictly weaker: it catches a build that RAN an Isaac install, not a payload
    that arrived some other way (a COPY from a vendor stage, an ADD of a tarball). Use it
    as a fast gate in front of an irreversible action, never as the proof itself -- the
    full scan is what the redistribution claim actually rests on.
    """
    if tarball is not None:
        report = ScanReport(image=str(tarball), source="tarball")
        entries = _iter_tarball(tarball)
        history: list[str] = []
    else:
        assert image is not None
        report = ScanReport(image=image, source="registry")
        history, report.digest = _image_history(image)
        entries = () if history_only else _iter_crane_export(image)
    report.history_only = history_only

    for path in entries:
        report.entries_scanned += 1
        if is_allowed(path):
            report.allowlisted_hits.append(_normalize(path))
            continue
        if path.endswith(WEIGHT_SUFFIXES) and len(report.weight_shaped_paths) < max_report:
            report.weight_shaped_paths.append(_normalize(path))
        why = classify_path(path)
        if why and len(report.payload_hits) < max_report:
            report.payload_hits.append({"path": _normalize(path), "why": why})

    for command in history:
        why = classify_history(command)
        if why:
            report.history_hits.append({"command": command.strip()[:400], "why": why})

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", help="Image reference to scan with crane.")
    parser.add_argument(
        "--tarball", type=Path, help="Scan a `docker save` tarball instead of a registry."
    )
    parser.add_argument("--json", type=Path, help="Write the JSON report here.")
    parser.add_argument(
        "--history-only",
        action="store_true",
        help=(
            "Inspect only the layer history (config blob, a few KB) instead of streaming "
            "the whole filesystem. Fast enough for a CI pre-publish gate; strictly weaker "
            "than a full scan, so never treat it as the redistribution proof itself."
        ),
    )
    args = parser.parse_args(argv)

    if not args.image and not args.tarball:
        parser.error("pass an image reference or --tarball")

    report = scan(args.image, args.tarball, history_only=args.history_only)
    payload = report.to_dict()

    print(f"image            {payload['image']}")
    if payload["digest"]:
        print(f"digest           {payload['digest']}")
    if report.history_only:
        print("mode             history-only (layer commands; filesystem NOT scanned)")
    print(f"entries scanned  {payload['entries_scanned']}")
    print(f"allowlisted      {len(payload['allowlisted_paths_present'])} path(s) we do ship:")
    for path in payload["allowlisted_paths_present"][:20]:
        print(f"                   {path}")
    if report.payload_hits:
        print(f"\nOMNIVERSE PAYLOAD DETECTED ({len(report.payload_hits)} path(s)):")
        for hit in report.payload_hits:
            print(f"  {hit['path']}\n      -> {hit['why']}")
    if report.history_hits:
        print(f"\nBUILD-TIME ISAAC INSTALL DETECTED ({len(report.history_hits)} layer(s)):")
        for hit in report.history_hits:
            print(f"  {hit['why']}\n      {hit['command']}")
    if report.weight_shaped_paths:
        print(
            f"\nFYI - {len(report.weight_shaped_paths)} weight-shaped path(s) present. "
            f"A tar listing has no contents, so these may be git-LFS pointers (fine) or "
            f"real tensors (not fine). The image build checks this by content:"
        )
        for path in report.weight_shaped_paths[:15]:
            print(f"  {path}")
    print(f"\nVERDICT: {payload['verdict']}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"report written to {args.json}")

    return 0 if report.clean else 1


if __name__ == "__main__":
    sys.exit(main())
