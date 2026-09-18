#!/usr/bin/env bash
# Verify lock-selected Ubuntu artifacts without resolving or fetching them.
set -euo pipefail

emit_direct_package_records() {
  local lock="$1"
  local output="$2"
  local package_rows
  [[ -f "$lock" ]]
  package_rows="$(awk '$0 == "packages:" { active=1; next } active && /^  - / { count++; next } active && /^[^ ]/ { active=0 } END { print count + 0 }' "$lock")"
  [[ "$package_rows" -gt 0 ]]
  sed -n '/^packages:$/,/^[^ ]/ s/^  - {binary: \([^,]*\), version: \("[^"]*"\|[^,]*\), source: [^,]*, sha256: \([0-9a-f]\{64\}\)}$/\1|\2|\3/p' "$lock" |
    tr -d '"' > "$output"
  [[ "$(wc -l < "$output")" -eq "$package_rows" ]]
  [[ "$(cut -d '|' -f 1 "$output" | LC_ALL=C sort -u | wc -l)" -eq "$package_rows" ]]
  while IFS='|' read -r package version digest; do
    [[ "$package" =~ ^[a-z0-9+.-]+$ ]]
    [[ -n "$version" && ! "$version" =~ [[:space:]\|/] ]]
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
  done < "$output"
}

verify_direct_debs() {
  local records="$1"
  local archive_dir="$2"
  local expected="$archive_dir/.expected"
  local observed="$archive_dir/.observed"
  local archive package version architecture digest
  LC_ALL=C sort -t '|' -k 1,1 "$records" > "$expected"
  : > "$observed"
  for archive in "$archive_dir"/*.deb; do
    [[ -f "$archive" ]]
    package="$(dpkg-deb -f "$archive" Package)"
    version="$(dpkg-deb -f "$archive" Version)"
    architecture="$(dpkg-deb -f "$archive" Architecture)"
    [[ "$architecture" == "amd64" || "$architecture" == "all" ]]
    digest="$(sha256sum "$archive" | cut -d ' ' -f 1)"
    printf '%s|%s|%s\n' "$package" "$version" "$digest" >> "$observed"
  done
  LC_ALL=C sort -t '|' -k 1,1 -o "$observed" "$observed"
  cmp "$expected" "$observed"
  rm -f "$expected" "$observed"
}

emit_signed_source_index_record() {
  local lock="$1"
  local output="$2"
  local suite path bytes digest
  [[ -f "$lock" ]]
  for field in suite path bytes sha256; do
    [[ "$(sed -n "/^    signed_index:\$/,/^    artifacts:/ s/^      $field: .*\$/x/p" "$lock" | wc -l)" -eq 1 ]]
  done
  suite="$(sed -n '/^    signed_index:$/,/^    artifacts:/ s/^      suite: //p' "$lock")"
  path="$(sed -n '/^    signed_index:$/,/^    artifacts:/ s/^      path: //p' "$lock")"
  bytes="$(sed -n '/^    signed_index:$/,/^    artifacts:/ s/^      bytes: //p' "$lock")"
  digest="$(sed -n '/^    signed_index:$/,/^    artifacts:/ s/^      sha256: //p' "$lock")"
  [[ "$suite" == "jammy-updates" ]]
  [[ "$path" == "dists/$suite/main/source/Sources.xz" ]]
  [[ "$bytes" =~ ^[1-9][0-9]*$ ]]
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
  printf '%s|%s|%s|%s\n' "$suite" "$path" "$bytes" "$digest" > "$output"
}

verify_locked_file() {
  local file="$1"
  local bytes="$2"
  local digest="$3"
  [[ -f "$file" ]]
  [[ "$(stat -c %s "$file")" -eq "$bytes" ]]
  printf '%s  %s\n' "$digest" "$file" | sha256sum -c -
}

verify_source_package() {
  /usr/bin/python3 - "$@" <<'PY'
import hashlib
import lzma
from pathlib import Path
import re
import sys


def require(condition, reason):
    if not condition:
        raise SystemExit("source package refused: " + reason)


def one(pattern, text):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    require(len(matches) == 1, "lock field cardinality")
    return matches[0]


def paragraphs(text):
    for block in text.strip().split("\n\n"):
        fields = {}
        previous = None
        for line in block.splitlines():
            if line.startswith((" ", "\t")):
                require(previous is not None, "orphan continuation")
                fields[previous] += "\n" + line.strip()
                continue
            require(":" in line, "invalid index field")
            key, value = line.split(":", 1)
            require(key not in fields, "duplicate index field")
            fields[key] = value.strip()
            previous = key
        yield fields


def checksums(value, digits):
    rows = {}
    for line in value.strip().splitlines():
        parts = line.split()
        require(len(parts) == 3, "checksum record shape")
        digest, size, name = parts
        require(re.fullmatch(r"[0-9a-f]{%d}" % digits, digest), "checksum syntax")
        require(size.isdecimal() and int(size) > 0, "checksum size")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_~-]*", name), "unsafe name")
        require(name not in rows, "duplicate checksum record")
        rows[name] = (int(size), digest)
    require(rows, "empty checksums")
    return rows


def locked_records(lock):
    rows = re.findall(
        r'^      - \{filename: ([^,]+), bytes: ([0-9]+), sha256: ([0-9a-f]{64}), locator: "([^"]+)"\}$',
        lock, flags=re.MULTILINE,
    )
    require(len(rows) == len(re.findall(r"^      - ", lock, re.MULTILINE)), "lock artifact syntax")
    require(len(rows) == 4 and len({row[0] for row in rows}) == 4, "lock artifact population")
    directory = one(r"^    directory: (.+)$", lock)
    snapshot = one(r'^snapshot: "([0-9]{8}T[0-9]{6}Z)"$', lock)
    for name, size, digest, locator in rows:
        require(locator == f"https://snapshot.ubuntu.com/ubuntu/{snapshot}/{directory}/{name}", "artifact locator")
    return {name: (int(size), digest) for name, size, digest, _ in rows}


def verified_stanza(lock, index):
    require(index.is_file() and not index.is_symlink(), "index type")
    size = int(one(r"^      bytes: ([0-9]+)$", lock))
    require(index.stat().st_size == size, "index size")
    compressed = index.read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == one(r"^      sha256: ([0-9a-f]{64})$", lock), "index digest")
    decoder = lzma.LZMADecompressor()
    decoded = decoder.decompress(compressed, max_length=64 * 1024 * 1024)
    require(decoder.eof and not decoder.unused_data, "index compression boundary")
    source = one(r"^    source: (.+)$", lock)
    version = one(r"^    version: (.+)$", lock)
    selected = [row for row in paragraphs(decoded.decode("utf-8"))
                if row.get("Package") == source and row.get("Version") == version]
    require(len(selected) == 1, "selected source cardinality")
    row = selected[0]
    require(row.get("Directory") == one(r"^    directory: (.+)$", lock), "source directory")
    return row


def verify_delivered(directory, sha256_rows, md5_rows):
    require(directory.is_dir() and not directory.is_symlink(), "source directory type")
    require({path.name for path in directory.iterdir()} == set(sha256_rows), "delivered population")
    for name, (size, digest) in sha256_rows.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "artifact type")
        require(path.stat().st_size == size, "artifact size")
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest, "artifact SHA256")
        # MD5 is only an additional signed-index consistency check, never trust.
        require(hashlib.md5(payload, usedforsecurity=False).hexdigest() == md5_rows[name][1], "artifact Files checksum")


lock = Path(sys.argv[1]).read_text(encoding="utf-8")
stanza = verified_stanza(lock, Path(sys.argv[2]))
sha256_rows = checksums(stanza.get("Checksums-Sha256", ""), 64)
md5_rows = checksums(stanza.get("Files", ""), 32)
require(sha256_rows == locked_records(lock), "index-to-lock artifact identity")
require({name: row[0] for name, row in sha256_rows.items()} ==
        {name: row[0] for name, row in md5_rows.items()}, "Files population/size")
if len(sys.argv) == 4:
    verify_delivered(Path(sys.argv[3]), sha256_rows, md5_rows)
PY
}

case "${1:-}" in
  direct-records)
    [[ $# -eq 3 ]]
    emit_direct_package_records "$2" "$3"
    ;;
  verify-direct)
    [[ $# -eq 3 ]]
    verify_direct_debs "$2" "$3"
    ;;
  source-index-record)
    [[ $# -eq 3 ]]
    emit_signed_source_index_record "$2" "$3"
    ;;
  verify-file)
    [[ $# -eq 4 ]]
    verify_locked_file "$2" "$3" "$4"
    ;;
  verify-source)
    [[ $# -eq 3 || $# -eq 4 ]]
    shift
    verify_source_package "$@"
    ;;
  *)
    echo "usage: verify_apt_artifacts.sh {direct-records|verify-direct|source-index-record|verify-file|verify-source} ..." >&2
    exit 2
    ;;
esac
