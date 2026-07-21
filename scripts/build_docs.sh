#!/usr/bin/env bash
#
# Regenerate the CLI reference under docs/cli/ from `npa --help`.
#
# Usage:
#   scripts/build_docs.sh            # regenerate docs/cli/ in place
#   scripts/build_docs.sh --check    # verify docs/cli/ is up to date (CI drift gate)
#
set -euo pipefail

NPA_BIN="${NPA_BIN:-npa}"
export COLUMNS="${COLUMNS:-120}"
export NO_COLOR=1

CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
fi

DOCS_DIR="docs/cli"
if [ "$CHECK" -eq 1 ]; then
  DOCS_DIR="$(mktemp -d)"
  trap 'rm -rf "$DOCS_DIR"' EXIT
fi

discover_commands() {
  python3 -c '
import re
import sys

text = sys.stdin.read()
names = []
for line in text.splitlines():
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    match = re.match(r"\s*[│|]\s*([a-z][a-z0-9-]*)\s{2,}", line)
    if match:
        names.append(match.group(1))
print("\n".join(sorted(set(names))))
'
}

# A Typer *group* accepts subcommands; its usage line ends with
# "COMMAND [ARGS]...". Leaf commands do not, so we only recurse into groups.
# Capture the help text fully before matching: piping straight into `grep -q`
# races with SIGPIPE on large help output (e.g. `workbench`) and can return a
# spurious non-match.
is_group() {
  local help_text
  help_text="$("$@" --help 2>&1 || true)"
  case "$help_text" in
    *"COMMAND [ARGS]"*) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$DOCS_DIR"
# Regenerate from a clean slate so pages for commands that were hidden or
# removed from `--help` do not linger as orphans (keeps in-place output
# identical to `--check`).
rm -f "$DOCS_DIR"/*.md

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

document_command() {
  local output_name="$1"
  shift
  local command_path=("$@")
  local output="${DOCS_DIR}/${output_name}.md"
  "${command_path[@]}" --help > "$tmp"
  python3 scripts/_help_to_markdown.py "$tmp" "$output_name" "${command_path[*]}" > "$output"
}

# Document a group and, recursively, every nested subgroup (e.g. the
# `retargeting` subgroup under `workbench sonic`). Leaf commands are skipped so
# that per-tool commands like `deploy`/`status` do not collide across tools.
document_group_recursive() {
  local output_name="$1"
  shift
  local command_path=("$@")
  document_command "$output_name" "${command_path[@]}"
  local child_help child
  child_help="$("${command_path[@]}" --help 2>&1)"
  for child in $(printf "%s" "$child_help" | discover_commands); do
    if is_group "${command_path[@]}" "$child"; then
      document_group_recursive "$child" "${command_path[@]}" "$child"
    fi
  done
}

top_help="$("$NPA_BIN" --help 2>&1)"
groups="$(printf "%s" "$top_help" | discover_commands)"

for group in $groups; do
  if is_group "$NPA_BIN" "$group"; then
    document_group_recursive "$group" "$NPA_BIN" "$group"
  else
    document_command "$group" "$NPA_BIN" "$group"
  fi
done

python3 scripts/_generate_docs_index.py "$DOCS_DIR/" > "$DOCS_DIR/README.md"

if [ "$CHECK" -eq 1 ]; then
  if ! diff -ruN docs/cli "$DOCS_DIR" > "$tmp" 2>&1; then
    echo "docs/cli is out of date. Run 'scripts/build_docs.sh' and commit the result." >&2
    echo >&2
    cat "$tmp" >&2
    exit 1
  fi
  echo "docs/cli is up to date."
else
  echo "Docs generated for groups: $groups"
fi
