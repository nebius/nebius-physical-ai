#!/usr/bin/env bash
#
# Regenerate the CLI reference under docs/cli/ from `npa --help`.
#
# Usage:
#   scripts/build_docs.sh            # regenerate docs/cli/ in place
#   scripts/build_docs.sh --check    # verify docs/cli/ is up to date (CI drift gate)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Every docs path below is relative, including the staging directory that has to
# sit beside docs/cli for the swap to be a rename. Anchor them rather than
# inheriting the caller's directory.
cd "$REPO_ROOT"

# Prefer the repository venv when NPA_BIN is not set explicitly: `npa` is only on
# PATH when that venv is activated, and the repo convention (AGENTS.md, the CI
# guardrail jobs) is npa/.venv.
if [ -z "${NPA_BIN:-}" ] && [ -x "${REPO_ROOT}/npa/.venv/bin/npa" ]; then
  NPA_BIN="${REPO_ROOT}/npa/.venv/bin/npa"
fi
NPA_BIN="${NPA_BIN:-npa}"
# Execution may use an absolute path from an isolated venv, but generated docs
# must show the stable public command name rather than leaking that checkout.
NPA_DISPLAY_BIN="${NPA_DOCS_DISPLAY_BIN:-$(basename "$NPA_BIN")}"

# Typer/Rich reads COLUMNS when rendering help. Do not inherit a shell or tmux
# width: the generated Markdown must be identical in CI and an interactive TTY.
# Set before the probe below so one width governs every invocation.
DOCS_COLUMNS="${NPA_DOCS_COLUMNS:-200}"
export NO_COLOR=1

# Every help fetch below tolerates a non-zero exit, because a leaf command may
# legitimately fail --help. That tolerance used to hide a missing interpreter
# entirely: with no `npa` on PATH the walk documented nothing, `--check` reported
# every page as drifted, and an in-place run deleted docs/cli/ and wrote nothing
# back. Resolve the binary once, up front, and fail with a usable message.
if ! env COLUMNS="$DOCS_COLUMNS" NO_COLOR=1 "$NPA_BIN" --help >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: cannot run '${NPA_BIN}'.

The CLI reference is generated from live \`npa --help\` output, so this script
needs a working npa. Install it and retry, or point NPA_BIN at an interpreter's
console script:

  python3 -m venv npa/.venv && npa/.venv/bin/pip install -e npa
  bash scripts/build_docs.sh${1:+ $1}

  # or
  NPA_BIN=/path/to/venv/bin/npa bash scripts/build_docs.sh${1:+ $1}
EOF
  exit 1
fi

run_with_docs_width() {
  # Bash treats COLUMNS specially and can reset an exported value to the TTY's
  # current width. Put it directly in each child environment instead.
  env COLUMNS="$DOCS_COLUMNS" NO_COLOR=1 "$@"
}

CHECK=0
if [ "${1:-}" = "--check" ]; then
  CHECK=1
fi

# Single cleanup handler: bash keeps only the last `trap ... EXIT`, so the scratch
# help file and every staging directory must be removed from one place. A
# successful in-place swap renames the staging dir away, so removing it is a no-op.
TMP_FILE=""
TEMP_DOCS_DIR=""
PREV_DOCS_DIR=""
HELP_CACHE_DIR=""
cleanup() {
  local status=$?
  [ -n "$TMP_FILE" ] && rm -f "$TMP_FILE"
  [ -n "$TEMP_DOCS_DIR" ] && rm -rf "$TEMP_DOCS_DIR"
  [ -n "$PREV_DOCS_DIR" ] && rm -rf "$PREV_DOCS_DIR"
  [ -n "$HELP_CACHE_DIR" ] && rm -rf "$HELP_CACHE_DIR"
  return "$status"
}
trap cleanup EXIT

# Always generate into a staging directory, including for an in-place run. The
# generator used to clear docs/cli first and write pages as it walked, so any walk
# that produced nothing left the reference deleted. Staging means no destructive
# step happens until a complete, non-empty result exists.
#
# Stage inside docs/ so the swap below is a same-filesystem rename.
TEMP_DOCS_DIR="$(mktemp -d "docs/.cli-stage-XXXXXX")"
DOCS_DIR="$TEMP_DOCS_DIR"

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

# A fresh staging directory is inherently a clean slate, so pages for commands
# that were hidden or removed from `--help` cannot linger as orphans.
mkdir -p "$DOCS_DIR"

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

# `discover_commands` scrapes Rich's box-drawing table out of --help text. A Typer
# or Rich upgrade, or a COLUMNS value that makes Rich wrap differently, can leave
# that regex matching nothing while npa itself is perfectly healthy. Silently
# publishing an empty reference is the failure this guards.
if [ -z "$groups" ]; then
  cat >&2 <<EOF
ERROR: found no commands in '${NPA_BIN} --help'.

The command table is parsed out of Rich's help output, so this usually means the
help rendering changed rather than that the CLI is broken. Compare:

  COLUMNS=${DOCS_COLUMNS} NO_COLOR=1 ${NPA_BIN} --help

against the parser in scripts/build_docs.sh:discover_commands. docs/cli was left
untouched.
EOF
  exit 1
fi

# The top level is the widest layer of the walk; warming it in parallel turns the
# dominant cost (one interpreter start per command) into wall-clock we overlap.
top_paths=()
for group in $groups; do
  top_paths+=("$NPA_BIN $group")
done
prefetch_help "${top_paths[@]}"

for group in $groups; do
  # A nested group may share a leaf name with a top-level command. The public
  # reference page for that filename belongs to the top-level command.
  rm -f "$DOCS_DIR/${group}.md"
  if is_group "$NPA_BIN" "$group"; then
    document_group_recursive "$group" "$NPA_BIN" "$group"
  else
    document_command "$group" "$NPA_BIN" "$group"
  fi
done

# Belt and braces behind the empty-groups guard: never install a tree with no
# pages. This has to run BEFORE the index is generated -- the index is itself a
# .md file in this directory, so checking afterwards always finds one page and the
# guard never fires.
staged_pages=("$DOCS_DIR"/*.md)
if [ ! -e "${staged_pages[0]}" ]; then
  echo "ERROR: generated no pages for groups: $groups. docs/cli was left untouched." >&2
  exit 1
fi

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
  # Two same-filesystem renames, so docs/cli is only ever replaced by a complete
  # tree and an interrupted run cannot leave it half-written.
  PREV_DOCS_DIR="$(mktemp -d "docs/.cli-prev-XXXXXX")"
  if [ -d docs/cli ]; then
    mv docs/cli "$PREV_DOCS_DIR/cli"
  fi
  if ! mv "$TEMP_DOCS_DIR" docs/cli; then
    if [ -d "$PREV_DOCS_DIR/cli" ]; then
      mv "$PREV_DOCS_DIR/cli" docs/cli
    fi
    echo "ERROR: could not install generated docs; docs/cli restored." >&2
    exit 1
  fi
  echo "Docs generated for groups: $groups"
fi
