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
  local scratch expected observed archive package version architecture digest
  local status=0 cleanup_status=0 archive_found=0
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/npa-habitat-apt.XXXXXX")" || return $?
  chmod 700 "$scratch" || status=$?
  expected="$scratch/expected"
  observed="$scratch/observed"
  if (( status == 0 )); then
    LC_ALL=C sort -t '|' -k 1,1 "$records" > "$expected" || status=$?
  fi
  if (( status == 0 )); then
    : > "$observed" || status=$?
  fi
  if (( status == 0 )); then
    for archive in "$archive_dir"/*.deb; do
      [[ -f "$archive" ]] || continue
      archive_found=1
      package="$(dpkg-deb -f "$archive" Package)" || status=$?
      version="$(dpkg-deb -f "$archive" Version)" || status=$?
      architecture="$(dpkg-deb -f "$archive" Architecture)" || status=$?
      if [[ "$architecture" != "amd64" && "$architecture" != "all" ]]; then
        status=1
      fi
      digest="$(sha256sum "$archive" | cut -d ' ' -f 1)" || status=$?
      printf '%s|%s|%s\n' "$package" "$version" "$digest" >> "$observed" || status=$?
      (( status == 0 )) || break
    done
    (( archive_found == 1 )) || status=1
  fi
  if (( status == 0 )); then
    LC_ALL=C sort -t '|' -k 1,1 -o "$observed" "$observed" || status=$?
  fi
  if (( status == 0 )); then
    cmp "$expected" "$observed" || status=$?
  fi
  cleanup_verify_direct_scratch "$scratch" || cleanup_status=$?
  if (( status != 0 )); then
    if (( cleanup_status != 0 )); then
      printf 'verify-direct scratch cleanup failed after status %d\n' "$status" >&2
    fi
    return "$status"
  fi
  if (( cleanup_status != 0 )); then
    printf 'verify-direct scratch cleanup failed\n' >&2
    return "$cleanup_status"
  fi
}

cleanup_verify_direct_scratch() {
  local scratch="$1"
  local status=0
  rm -f -- "$scratch/expected" "$scratch/observed" || status=$?
  rmdir -- "$scratch" || status=$?
  return "$status"
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
  local script_directory
  script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  /usr/bin/python3 "$script_directory/verify_apt_source.py" "$@"
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
