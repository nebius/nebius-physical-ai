"""Backport Setuptools' official Unicode manifest exclusion fix.

The exact upstream dd9f436a36486b4cb8a4c70a2321548b0be09b8f change preserves
vLLM's Setuptools<81 constraint. Unknown source bytes are rejected. Installed
license, version, APIs, RECORD hashes, and checked-hash bytecode are preserved.

The patch fragments below are derived from Setuptools under its MIT license:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path
import py_compile

SOURCE_HASHES = {
    "setuptools/command/egg_info.py": (
        "e7177956d81f43a53a873f1702ce2971eb6fd4212f56b68a7d9a53131c67bc99",
        "cb71f019b6e6c0fca6d0d4739a323e27c9a60d074e2a0f0f1f485d379747ce78",
    ),
    "setuptools/unicode_utils.py": (
        "ba430687ca44030e85fc4cdbf8ae43ddcfb4efc46003f19c174a16ea5838952b",
        "0e400678e34254f1f7e8ef29a85515473e9df98699ea4d4314098ee414721d52",
    ),
}

MATCHER = '''class _NormalizedMatcher:
    """
    Wrap a compiled pattern so that matching is insensitive to Unicode
    normalization form.

    File names walked from disk (NFD on macOS APFS/HFS+) and patterns from
    ``MANIFEST.in`` (typically NFC) can denote the same file while differing
    byte-for-byte. Normalizing both sides before matching keeps an exclusion
    (or inclusion) from silently failing. See GHSA-h35f-9h28-mq5c.
    """

    def __init__(self, pattern: re.Pattern) -> None:
        self._pattern = pattern

    def match(self, path):
        return self._pattern.match(unicode_utils.normalize(path))

    def search(self, path):
        return self._pattern.search(unicode_utils.normalize(path))


'''

NORMALIZE_PATTERN = """    # Normalize the pattern so it matches paths regardless of the Unicode
    # normalization form used on disk (GHSA-h35f-9h28-mq5c). Candidate paths
    # are normalized to the same form by ``_NormalizedMatcher``.
    glob = unicode_utils.normalize(glob)

"""

NORMALIZE = '''def normalize(text):
    """
    Return *text* in a canonical Unicode form (NFC) so that names which are
    visually identical but encoded differently compare equal.

    macOS APFS/HFS+ store file names in decomposed form (NFD), while patterns
    in ``MANIFEST.in`` are typically authored composed (NFC). The two denote
    the same file but differ byte-for-byte, so matching them directly lets an
    exclusion silently fail. Normalizing both the walked path and the pattern
    to a single form before matching avoids that (GHSA-h35f-9h28-mq5c).
    """
    return unicodedata.normalize('NFC', text) if isinstance(text, str) else text


'''


def patched_source(name: str, source: bytes) -> bytes:
    original, patched = SOURCE_HASHES[name]
    digest = hashlib.sha256(source).hexdigest()
    if digest == patched:
        return source
    if digest != original:
        raise RuntimeError("Setuptools source differs from the reviewed upstream bytes")
    text = source.decode()
    if name.endswith("egg_info.py"):
        text = text.replace(
            "def translate_pattern(", MATCHER + "def translate_pattern(", 1
        )
        text = text.replace(
            "    # This will split", NORMALIZE_PATTERN + "    # This will split", 1
        )
        text = text.replace(
            "return re.compile(pat, flags=re.MULTILINE | re.DOTALL)",
            "return _NormalizedMatcher(re.compile(pat, flags=re.MULTILINE | re.DOTALL))",
            1,
        )
    else:
        text = text.replace("def filesys_decode(", NORMALIZE + "def filesys_decode(", 1)
    result = text.encode()
    if hashlib.sha256(result).hexdigest() != patched:
        raise RuntimeError("Setuptools backport does not match the reviewed patch")
    return result


def main() -> None:
    distribution = metadata.distribution("setuptools")
    if distribution.version != "80.10.2":
        raise RuntimeError("the manifest backport requires exact Setuptools 80.10.2")
    root = Path(distribution.locate_file(""))
    record = Path(
        distribution.locate_file(
            next(
                path
                for path in distribution.files or ()
                if str(path) == "setuptools-80.10.2.dist-info/RECORD"
            )
        )
    )
    rows = list(csv.reader(io.StringIO(record.read_text())))
    recorded = {row[0] for row in rows}
    if not set(SOURCE_HASHES) <= recorded:
        raise RuntimeError("Setuptools RECORD does not contain both reviewed modules")
    # Validate both originals before replacing either file.
    patched = {
        name: patched_source(name, (root / name).read_bytes()) for name in SOURCE_HASHES
    }
    updates = {}
    for name, content in patched.items():
        source = root / name
        source.write_bytes(content)
        compiled = Path(
            py_compile.compile(
                str(source),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
        )
        for path in (source, compiled):
            content = path.read_bytes()
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                .rstrip(b"=")
                .decode()
            )
            relative = path.relative_to(root).as_posix()
            updates[relative] = [relative, "sha256=" + digest, str(len(content))]
    rows = [updates.pop(row[0], row) for row in rows]
    rows.extend(updates.values())
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    record.write_text(output.getvalue())


if __name__ == "__main__":
    main()
