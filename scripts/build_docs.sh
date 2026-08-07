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
# Execution may use an absolute path from an isolated venv, but generated docs
# must show the stable public command name rather than leaking that checkout.
NPA_DISPLAY_BIN="${NPA_DOCS_DISPLAY_BIN:-$(basename "$NPA_BIN")}"
# Typer/Rich reads COLUMNS when rendering help. Do not inherit a shell or tmux
# width: the generated Markdown must be identical in CI and an interactive TTY.
DOCS_COLUMNS="${NPA_DOCS_COLUMNS:-200}"
export NO_COLOR=1

run_with_docs_width() {
  # Bash treats COLUMNS specially and can reset an exported value to the TTY's
  # current width. Put it directly in each child environment instead.
  env COLUMNS="$DOCS_COLUMNS" NO_COLOR=1 "$@"
}

CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
fi

# Single cleanup handler: bash keeps only the last `trap ... EXIT`, so both the
# scratch help file and the --check temp dir must be removed from one place.
TMP_FILE=""
TEMP_DOCS_DIR=""
HELP_CACHE_DIR=""
cleanup() {
  [ -n "$TMP_FILE" ] && rm -f "$TMP_FILE"
  [ -n "$TEMP_DOCS_DIR" ] && rm -rf "$TEMP_DOCS_DIR"
  [ -n "$HELP_CACHE_DIR" ] && rm -rf "$HELP_CACHE_DIR"
  # An EXIT trap's final command can replace an otherwise successful status.
  # Keep normal regeneration successful when TEMP_DOCS_DIR is intentionally empty.
  return 0
}
trap cleanup EXIT

DOCS_DIR="docs/cli"
if [ "$CHECK" -eq 1 ]; then
  TEMP_DOCS_DIR="$(mktemp -d)"
  DOCS_DIR="$TEMP_DOCS_DIR"
fi

HELP_CACHE_DIR="$(mktemp -d)"

# `npa <path> --help` costs a full Python + Typer startup, and the walk below
# needs the same text three times per node (is_group, the page itself, and the
# child listing). Fetch each once.
help_for() {
  local key cache
  key="$(printf '%s_' "$@" | tr -c 'A-Za-z0-9_' '.')"
  cache="${HELP_CACHE_DIR}/${key}.help"
  if [ ! -f "$cache" ]; then
    local staging="${cache}.$$"
    run_with_docs_width "$@" --help > "$staging" 2>&1 || true
    mv -f "$staging" "$cache"
  fi
  cat "$cache"
}

# Warm the cache for several command paths at once; the fetches are independent
# and each is dominated by interpreter startup rather than CPU.
prefetch_help() {
  local path
  for path in "$@"; do
    # shellcheck disable=SC2086 - deliberate word split: the path is a command.
    ( help_for $path >/dev/null ) &
  done
  wait
}

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
  help_text="$(help_for "$@")"
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

TMP_FILE="$(mktemp)"
tmp="$TMP_FILE"

document_command() {
  local output_name="$1"
  shift
  local command_path=("$@")
  local display_path=("$NPA_DISPLAY_BIN" "${command_path[@]:1}")
  local output="${DOCS_DIR}/${output_name}.md"
  # Pages are keyed by leaf group name. Generation starts from a clean slate, so
  # a pre-existing file here means two distinct subgroups share a leaf name and
  # would silently overwrite each other. Fail loudly instead.
  if [ -f "$output" ]; then
    echo "ERROR: doc name collision for '${output_name}' (${command_path[*]}); a" \
         "same-named subgroup already generated ${output}. Use a unique group name." >&2
    exit 1
  fi
  help_for "${command_path[@]}" > "$tmp"
  python3 scripts/_help_to_markdown.py "$tmp" "$output_name" "${display_path[*]}" > "$output"
}

# Document a group and, recursively, every nested subgroup (e.g. the
# `retargeting` subgroup under `workbench sonic`). Leaf commands are skipped so
# that per-tool commands like `deploy`/`status` do not collide across tools.
document_group_recursive() {
  local output_name="$1"
  shift
  local command_path=("$@")
  local prefetch_paths=()
  document_command "$output_name" "${command_path[@]}"
  local child_help child children
  child_help="$(help_for "${command_path[@]}")"
  children="$(printf "%s" "$child_help" | discover_commands)"
  for child in $children; do
    prefetch_paths+=("${command_path[*]} $child")
  done
  prefetch_help "${prefetch_paths[@]}"
  prefetch_paths=()
  for child in $children; do
    if is_group "${command_path[@]}" "$child"; then
      document_group_recursive "$child" "${command_path[@]}" "$child"
    fi
  done
}

top_help="$(help_for "$NPA_BIN")"
groups="$(printf "%s" "$top_help" | discover_commands)"

# The top level is the widest layer of the walk; warming it in parallel turns the
# dominant cost (one interpreter start per command) into wall-clock we overlap.
top_paths=()
for group in $groups; do
  top_paths+=("$NPA_BIN $group")
done
prefetch_help "${top_paths[@]}"

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
