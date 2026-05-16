#!/usr/bin/env bash
set -euo pipefail

NPA_BIN="${NPA_BIN:-npa}"
export COLUMNS="${COLUMNS:-120}"
export NO_COLOR=1

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

mkdir -p docs/cli

top_help="$("$NPA_BIN" --help 2>&1)"
groups="$(printf "%s" "$top_help" | discover_commands)"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

document_command() {
  local output_name="$1"
  shift
  local command_path=("$@")
  local output="docs/cli/${output_name}.md"
  "${command_path[@]}" --help > "$tmp"
  python3 scripts/_help_to_markdown.py "$tmp" "$output_name" "${command_path[*]}" > "$output"
}

for group in $groups; do
  document_command "$group" "$NPA_BIN" "$group"
  if [ "$group" = "workbench" ]; then
    workbench_help="$("$NPA_BIN" workbench --help 2>&1)"
    workbench_groups="$(printf "%s" "$workbench_help" | discover_commands)"
    for child in $workbench_groups; do
      document_command "$child" "$NPA_BIN" workbench "$child"
    done
  fi
done

python3 scripts/_generate_docs_index.py docs/cli/ > docs/cli/README.md
echo "Docs generated for groups: $groups"
