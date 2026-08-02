#!/usr/bin/env python3
"""Regenerate the hash-pinned NVIDIA Isaac wheel list used by the runtime bootstrap.

The four Isaac workbench images ship **no** NVIDIA Isaac bytes. Isaac Sim and Isaac
Lab are fetched on first run from ``https://pypi.nvidia.com`` under the operator's own
EULA acceptance (see ``isaac_bootstrap.sh``). Because that fetch happens at *run* time
on a customer's machine, it is attack surface: this script produces the
``--require-hashes`` manifest that pins every wheel to a sha256 the repo has reviewed,
so a compromised or man-in-the-middled index cannot substitute a different wheel.

No credentials are needed. ``pypi.nvidia.com`` serves an anonymous PEP-503 "simple"
index whose ``href`` carries the wheel's sha256 as a URL fragment, e.g.

    <a href="isaaclab-2.3.2.post1-cp311-none-manylinux_2_35_x86_64.whl#sha256=53f0...">

so the digest is published alongside the artifact and we can pin it without downloading
4.5 GB of wheels.

The package set is exactly ``isaacsim[all,extscache]`` plus ``isaacsim-kernel`` and
``isaaclab`` — i.e. precisely what these images used to bake. It is deliberately not
trimmed: ``isaacsim-extscache-kit`` and ``-kit-sdk`` are 4.16 GB of the 4.46 GB total,
so dropping the small optional members (ros1/ros2/test/benchmark/code-editor/cortex/
example/template) would save ~3% of the download in exchange for a risk of subtle
import failures inside ``isaaclab_tasks``. Keeping the set identical to what was baked
makes this re-architecture behaviour-preserving.

Usage:

    npa/.venv/bin/python npa/docker/workbench/common/refresh_isaac_wheel_pins.py \
        --isaacsim 5.1.0.0 --isaaclab 2.3.2.post1

    # check-only, for CI / pre-commit
    npa/.venv/bin/python npa/docker/workbench/common/refresh_isaac_wheel_pins.py --check
"""

from __future__ import annotations

import argparse
import html.parser
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INDEX_URL = "https://pypi.nvidia.com"
DEFAULT_ISAAC_SIM_VERSION = "5.1.0.0"
DEFAULT_ISAAC_LAB_VERSION = "2.3.2.post1"
# isaaclab and isaacsim both declare Requires-Python: ==3.11.*
DEFAULT_PYTHON_TAG = "cp311"
DEFAULT_PLATFORM = "x86_64"

OUTPUT_PATH = Path(__file__).resolve().parent / "isaac-nvidia-wheels.txt"

# ``isaacsim[all,extscache]`` expanded from the isaacsim 5.1.0.0 wheel's own METADATA,
# plus the unconditional ``isaacsim-kernel`` dependency. Keep this list in the same
# order the metadata declares it so a diff against a new Isaac Sim release is readable.
ISAAC_SIM_PACKAGES = (
    "isaacsim",
    "isaacsim-kernel",
    # extra == "extscache"
    "isaacsim-extscache-kit",
    "isaacsim-extscache-kit-sdk",
    "isaacsim-extscache-physics",
    # extra == "all"
    "isaacsim-app",
    "isaacsim-asset",
    "isaacsim-benchmark",
    "isaacsim-code-editor",
    "isaacsim-core",
    "isaacsim-cortex",
    "isaacsim-example",
    "isaacsim-gui",
    "isaacsim-replicator",
    "isaacsim-rl",
    "isaacsim-robot",
    "isaacsim-robot-motion",
    "isaacsim-robot-setup",
    "isaacsim-ros1",
    "isaacsim-ros2",
    "isaacsim-sensor",
    "isaacsim-storage",
    "isaacsim-template",
    "isaacsim-test",
    "isaacsim-utils",
)
ISAAC_LAB_PACKAGES = ("isaaclab",)


@dataclass(frozen=True)
class Wheel:
    project: str
    filename: str
    sha256: str

    @property
    def distribution(self) -> str:
        """PEP-427 distribution name (underscored) from the wheel filename."""
        return self.filename.split("-", 1)[0]


class _SimpleIndexParser(html.parser.HTMLParser):
    """Collect ``(filename, sha256)`` pairs from a PEP-503 simple index page."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        name, _, fragment = href.partition("#")
        name = name.rsplit("/", 1)[-1]
        match = re.fullmatch(r"sha256=([0-9a-f]{64})", fragment)
        if name and match:
            self.entries.append((name, match.group(1)))


def _fetch_index(index_url: str, project: str) -> list[tuple[str, str]]:
    url = f"{index_url.rstrip('/')}/{project}/"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https index
        payload = response.read().decode("utf-8", "replace")
    parser = _SimpleIndexParser()
    parser.feed(payload)
    if not parser.entries:
        raise RuntimeError(f"no hashed wheel links found at {url}")
    return parser.entries


def _select_wheel(
    project: str,
    entries: list[tuple[str, str]],
    version: str,
    python_tag: str,
    platform: str,
) -> Wheel:
    """Pick the one wheel for this version/interpreter/platform.

    Isaac wheels come in two shapes: interpreter-specific
    (``isaacsim-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl``) and pure-python
    (``isaaclab-3.0.0b2-py3-none-any.whl``). Accept either, but require an exact
    version match so a newer release on the index can never be picked up silently.
    """
    normalized = project.replace("-", "_")
    version_prefix = f"{normalized}-{version}-"
    candidates = [
        Wheel(project=project, filename=name, sha256=digest)
        for name, digest in entries
        if name.startswith(version_prefix) and name.endswith(".whl")
    ]
    if not candidates:
        available = sorted(
            {
                name[len(normalized) + 1 :].split("-", 1)[0]
                for name, _ in entries
                if name.startswith(f"{normalized}-")
            }
        )
        raise RuntimeError(
            f"{project}: no wheel for version {version!r}; index offers {available[-8:]}"
        )

    def _matches(wheel: Wheel) -> bool:
        tags = wheel.filename[: -len(".whl")].split("-")
        # {distribution}-{version}(-{build})?-{python}-{abi}-{platform}
        py_tag, plat_tag = tags[-3], tags[-1]
        if py_tag == "py3" and plat_tag == "any":
            return True
        return py_tag == python_tag and platform in plat_tag

    matching = [wheel for wheel in candidates if _matches(wheel)]
    if len(matching) != 1:
        raise RuntimeError(
            f"{project}: expected exactly one {python_tag}/{platform} wheel for "
            f"{version}, found {[w.filename for w in matching]}"
        )
    return matching[0]


def build_pin_file(
    *,
    index_url: str = DEFAULT_INDEX_URL,
    isaac_sim_version: str = DEFAULT_ISAAC_SIM_VERSION,
    isaac_lab_version: str = DEFAULT_ISAAC_LAB_VERSION,
    python_tag: str = DEFAULT_PYTHON_TAG,
    platform: str = DEFAULT_PLATFORM,
) -> str:
    wheels: list[tuple[str, str, Wheel]] = []
    for project in ISAAC_SIM_PACKAGES:
        entries = _fetch_index(index_url, project)
        wheels.append(
            (project, isaac_sim_version, _select_wheel(project, entries, isaac_sim_version, python_tag, platform))
        )
    for project in ISAAC_LAB_PACKAGES:
        entries = _fetch_index(index_url, project)
        wheels.append(
            (project, isaac_lab_version, _select_wheel(project, entries, isaac_lab_version, python_tag, platform))
        )

    lines = [
        "# GENERATED — do not edit by hand.",
        "#   npa/.venv/bin/python npa/docker/workbench/common/refresh_isaac_wheel_pins.py",
        "#",
        "# NVIDIA Isaac wheels fetched at FIRST RUN by isaac_bootstrap.sh, never baked into",
        "# any image layer. Installed with `pip install --no-deps --require-hashes` against",
        f"# --index-url {index_url} so the set cannot be shadowed from PyPI and a wheel cannot",
        "# be substituted. Every OSS transitive dependency is baked at build time from PyPI",
        "# instead (see install_isaac_runtime_base.sh), which is what keeps --no-deps honest.",
        "#",
        "# Package set: isaacsim[all,extscache] + isaacsim-kernel + isaaclab (unabridged —",
        "# see the module docstring for why it is not trimmed).",
        f"# isaacsim  {isaac_sim_version}",
        f"# isaaclab  {isaac_lab_version}",
        f"# python    {python_tag}   platform {platform}",
        "",
    ]
    for project, version, wheel in wheels:
        lines.append(f"{project}=={version} \\")
        lines.append(f"    --hash=sha256:{wheel.sha256}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--isaacsim", default=DEFAULT_ISAAC_SIM_VERSION)
    parser.add_argument("--isaaclab", default=DEFAULT_ISAAC_LAB_VERSION)
    parser.add_argument("--python-tag", default=DEFAULT_PYTHON_TAG)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed file differs from a freshly resolved one.",
    )
    args = parser.parse_args(argv)

    rendered = build_pin_file(
        index_url=args.index_url,
        isaac_sim_version=args.isaacsim,
        isaac_lab_version=args.isaaclab,
        python_tag=args.python_tag,
        platform=args.platform,
    )
    if args.check:
        if not args.output.is_file():
            print(f"{args.output} does not exist", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.output} is stale; re-run without --check to regenerate",
                file=sys.stderr,
            )
            return 1
        print(f"{args.output} is up to date")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    count = rendered.count("--hash=sha256:")
    print(f"wrote {args.output} ({count} hash-pinned wheels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
