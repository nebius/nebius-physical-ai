"""Select the reviewed MoviePy/Pillow closure from a strict requirements lock.

The shared Isaac installer must not infer package identity with line-oriented
``grep``/``sed`` matching.  Parse the repository's intentionally narrow lock
grammar, canonicalize project names, reject ambiguous requirement forms and
duplicates, and emit the ordinary closure only after the whole input validates.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EXACT_REQUIREMENT = re.compile(
    r"(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"=="
    r"(?P<version>[A-Za-z0-9](?:[A-Za-z0-9.!+_-]*[A-Za-z0-9])?)"
)
QUALIFIED_MOVIEPY = "2.2.1"
QUALIFIED_PILLOW = "13.0.0"


def canonical_name(name: str) -> str:
    """Return the PEP 503 comparison form for a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def select_closure(requirements: str, python_minor: str) -> tuple[str, bool]:
    """Validate *requirements* and remove packages installed by bounded paths."""
    parsed: list[tuple[str | None, str]] = []
    versions: dict[str, str] = {}

    for line_number, line in enumerate(requirements.splitlines(keepends=True), start=1):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            parsed.append((None, line))
            continue

        match = EXACT_REQUIREMENT.fullmatch(stripped)
        if match is None:
            raise ValueError(
                f"line {line_number}: expected one exact name==version requirement; "
                "direct references, includes, constraints, markers, options, hashes, "
                "extras, continuations, and alternate comparison forms are forbidden"
            )
        name = canonical_name(match.group("name"))
        if name in versions:
            raise ValueError(
                f"line {line_number}: duplicate canonical requirement for {name}"
            )
        versions[name] = match.group("version")
        parsed.append((name, line))

    qualified = python_minor == "3.11" and "moviepy" in versions
    if qualified:
        if versions["moviepy"] != QUALIFIED_MOVIEPY:
            raise ValueError(
                "Python 3.11 MoviePy closure requires exactly moviepy==2.2.1"
            )
        if versions.get("pillow") != QUALIFIED_PILLOW:
            raise ValueError(
                "Python 3.11 MoviePy closure requires exactly pillow==13.0.0"
            )

    omitted = {"imageio-ffmpeg"}
    if qualified:
        omitted.update(("moviepy", "pillow"))
    selected = "".join(line for name, line in parsed if name not in omitted)
    return selected, qualified


def main() -> None:
    """Write a validated closure and an explicit qualified-state marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualified-marker", type=Path, required=True)
    parser.add_argument("--python-minor", required=True)
    args = parser.parse_args()

    args.output.unlink(missing_ok=True)
    args.qualified_marker.unlink(missing_ok=True)
    selected, qualified = select_closure(
        args.requirements.read_text(encoding="utf-8"),
        args.python_minor,
    )
    args.output.write_text(selected, encoding="utf-8")
    if qualified:
        args.qualified_marker.write_text(
            f"moviepy=={QUALIFIED_MOVIEPY}\npillow=={QUALIFIED_PILLOW}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
