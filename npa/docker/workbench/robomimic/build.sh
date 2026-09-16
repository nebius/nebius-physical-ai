#!/usr/bin/env bash
# Build the neutral robomimic image with serialized tag assignment and durable owner receipts.
set -euo pipefail
umask 077

failure_reason="pre-image-failure"
image_created=0
image_id=""
iid_file=""
tag_assignment_attempted=0
tag_assignment_completed=0
transaction_complete=0
failure_receipt_published=0
failure_receipt=""
receipt_tmp=""
receipt_tmp_name=""
receipt_tmp_identity=""
scratch_root=""
scratch_root_fd=""
scratch_root_identity=""
scratch_anchor=""
context=""
context_fd=""
context_identity=""
context_anchor=""
receipt_dir=""
receipt_dir_fd=""
receipt_dir_identity=""
receipt_anchor=""

fail() {
  failure_reason="$1"
  echo "robomimic neutral build refused: $1" >&2
  exit 1
}

require_owner_private_directory() {
  local path="$1"
  [[ -d "${path}" && ! -L "${path}" ]] \
    || fail "owner directory is absent or unsafe"
  [[ "$(stat -c '%u' -- "${path}")" == "$(id -u)" ]] \
    || fail "owner directory has the wrong owner"
  [[ "$(stat -c '%a' -- "${path}")" == "700" ]] \
    || fail "owner directory must have mode 0700"
}

require_trusted_directory_chain() {
  local path="$1"
  local current=""
  local component=""
  local owner=""
  local mode=""
  local mode_value=0
  local -a components=()
  [[ "${path}" == /* && "${path}" != "/" ]] \
    || fail "owner directory path must be absolute"
  IFS='/' read -r -a components <<< "${path#/}"
  for component in "${components[@]}"; do
    [[ -n "${component}" && "${component}" != "." && "${component}" != ".." ]] \
      || fail "owner directory path is malformed"
    current="${current}/${component}"
    [[ -d "${current}" && ! -L "${current}" ]] \
      || fail "owner directory chain is absent or unsafe"
    IFS=: read -r owner mode < <(stat -c '%u:%a' -- "${current}")
    [[ "${owner}" == "0" || "${owner}" == "$(id -u)" ]] \
      || fail "owner directory chain has the wrong owner"
    mode_value=$((8#${mode}))
    if (( (mode_value & 0022) != 0 )); then
      (( owner == 0 && (mode_value & 01000) != 0 )) \
        || fail "owner directory chain is writable by another user"
    fi
  done
  require_owner_private_directory "${path}"
}

directory_identity() {
  stat -c '%d:%i:%u:%a:%F' -- "$1"
}

descriptor_identity() {
  stat -Lc '%d:%i:%u:%a:%F' -- "/proc/$$/fd/$1"
}

directory_binding_matches() {
  local descriptor="$1"
  local path="$2"
  local expected="$3"
  [[ -d "${path}" && ! -L "${path}" ]] || return 1
  [[ "$(descriptor_identity "${descriptor}" 2>/dev/null)" == "${expected}" ]] \
    || return 1
  [[ "$(directory_identity "${path}" 2>/dev/null)" == "${expected}" ]]
}

context_bindings_match() {
  directory_binding_matches \
    "${scratch_root_fd}" "${scratch_root}" "${scratch_root_identity}" \
    || return 1
  directory_binding_matches \
    "${context_fd}" "${context}" "${context_identity}" \
    || return 1
  [[ "$(directory_identity "${scratch_anchor}/${transaction_id}" 2>/dev/null)" \
    == "${context_identity}" ]]
}

transaction_bindings_match() {
  context_bindings_match || return 1
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}"
}

require_transaction_bindings() {
  transaction_bindings_match \
    || fail "build transaction directory identity changed"
}

file_identity() {
  stat -c '%d:%i:%u:%a:regular file' -- "$1"
}

receipt_staging_matches() {
  [[ -n "${receipt_tmp_name}" && -n "${receipt_tmp_identity}" ]] || return 1
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || return 1
  local anchored="${receipt_anchor}/${receipt_tmp_name}"
  [[ -f "${anchored}" && ! -L "${anchored}" ]] || return 1
  [[ "$(file_identity "${anchored}" 2>/dev/null)" == "${receipt_tmp_identity}" ]]
}

start_receipt_staging() {
  local purpose="$1"
  local created=""
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || return 1
  created="$(mktemp "${receipt_anchor}/.${transaction_id}.${purpose}.XXXXXXXX")" \
    || return 1
  receipt_tmp_name="$(basename "${created}")"
  receipt_tmp="${receipt_dir}/${receipt_tmp_name}"
  chmod 600 -- "${receipt_anchor}/${receipt_tmp_name}" || return 1
  receipt_tmp_identity="$(file_identity "${receipt_anchor}/${receipt_tmp_name}")" \
    || return 1
  receipt_staging_matches
}

clear_receipt_staging() {
  [[ -n "${receipt_tmp_name}" ]] || return 0
  receipt_staging_matches || return 1
  rm -f -- "${receipt_anchor}/${receipt_tmp_name}" || return 1
  receipt_tmp=""
  receipt_tmp_name=""
  receipt_tmp_identity=""
}

fd_relative_link_receipt() {
  local target_name="$1" inherited_directory_fd=9
  python3 - receipt-link "${inherited_directory_fd}" "${receipt_tmp_name}" \
    "${target_name}" "${receipt_tmp_identity}" "${receipt_dir_identity}" \
    9<&"${receipt_dir_fd}" <<'PY'
import os
import stat
import sys
_, directory_fd, source_name, target_name, source_identity, directory_identity = sys.argv[1:]
directory_fd = int(directory_fd)

def identity(value: os.stat_result, kind: str) -> str:
    mode = stat.S_IMODE(value.st_mode)
    return f"{value.st_dev}:{value.st_ino}:{value.st_uid}:{mode:o}:{kind}"

def matches_source(value: os.stat_result, links: int) -> bool:
    return (
        stat.S_ISREG(value.st_mode) and identity(value, "regular file") == source_identity
        and value.st_nlink == links
    )

if any(not name or os.path.basename(name) != name for name in (source_name, target_name)):
    raise SystemExit(1)
if identity(os.fstat(directory_fd), "directory") != directory_identity:
    raise SystemExit(1)
source = os.stat(source_name, dir_fd=directory_fd, follow_symlinks=False)
if not matches_source(source, 1):
    raise SystemExit(1)
try:
    os.link(source_name, target_name, src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd, follow_symlinks=False)
except OSError:
    raise SystemExit(1) from None
source = os.stat(source_name, dir_fd=directory_fd, follow_symlinks=False)
target = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
if not matches_source(source, 2) or not matches_source(target, 2):
    raise SystemExit(1)
PY
}

link_receipt_target() {
  local target_name="$1"
  local target_anchor="${receipt_anchor}/${target_name}"
  receipt_staging_matches || return 1
  [[ ! -e "${target_anchor}" && ! -L "${target_anchor}" ]] || return 1
  fd_relative_link_receipt "${target_name}" || return 1
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || return 1
  receipt_staging_matches || return 1
  [[ -f "${target_anchor}" && ! -L "${target_anchor}" ]] || return 1
  [[ "$(file_identity "${target_anchor}" 2>/dev/null)" == "${receipt_tmp_identity}" ]] \
    || return 1
  [[ "$(stat -c '%u:%a:%h' -- "${receipt_anchor}/${receipt_tmp_name}" 2>/dev/null)" \
    == "$(id -u):600:2" ]] || return 1
  [[ "$(stat -c '%u:%a:%h' -- "${target_anchor}" 2>/dev/null)" \
    == "$(id -u):600:2" ]] || return 1
  clear_receipt_staging || return 2
  [[ "$(stat -c '%u:%a:%h' -- "${target_anchor}" 2>/dev/null)" \
    == "$(id -u):600:1" ]]
}

write_failure_receipt() {
  local destination="$1"
  local context_disposition="$2"
  local shared_tag_disposition="$3"
  printf '{"context_evidence_disposition":"%s","failure_reason":"%s","immutable_image_disposition":"retained-no-exclusive-ownership-proof","immutable_image_id":"%s","intended_full_sha_tag":"%s","revision":"%s","schema":"npa.robomimic.neutral-build-failure.v1","shared_tag_disposition":"%s","status":"failure","transaction_id":"%s"}\n' \
    "${context_disposition}" "${failure_reason}" "${image_id}" "${image}" \
    "${revision}" "${shared_tag_disposition}" "${transaction_id}" \
    > "${destination}"
}

shared_tag_disposition() {
  if [[ "${tag_assignment_attempted}" -eq 1 \
    && "${tag_assignment_completed}" -eq 1 ]]; then
    printf '%s\n' "assignment-completed-no-cleanup-modification"
  elif [[ "${tag_assignment_attempted}" -eq 1 ]]; then
    printf '%s\n' "assignment-outcome-unknown-no-cleanup-modification"
  else
    printf '%s\n' "not-modified-by-cleanup"
  fi
}

read_image_id_file() {
  local path="$1"
  local -a lines=()
  [[ -f "${path}" && ! -L "${path}" ]] \
    || fail "Docker iidfile is absent or unsafe"
  [[ "$(stat -c '%u:%h' -- "${path}")" == "$(id -u):1" ]] \
    || fail "Docker iidfile ownership is unsafe"
  mapfile -t lines < "${path}"
  [[ "${#lines[@]}" -eq 1 && "${lines[0]}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || fail "Docker iidfile is malformed"
  printf '%s\n' "${lines[0]}"
}

recover_image_id_file() {
  local path="$1"
  local -a lines=()
  [[ -n "${path}" && -f "${path}" && ! -L "${path}" ]] || return 1
  [[ "$(stat -c '%u:%h' -- "${path}" 2>/dev/null)" == "$(id -u):1" ]] \
    || return 1
  mapfile -t lines < "${path}" || return 1
  [[ "${#lines[@]}" -eq 1 && "${lines[0]}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || return 1
  printf '%s\n' "${lines[0]}"
}

inspect_image_id() {
  local reference="$1"
  local output_path="$2"
  local error_path="$3"
  local -a lines=()
  if ! docker image inspect --format '{{.Id}}' "${reference}" \
    > "${output_path}" 2> "${error_path}"; then
    return 1
  fi
  mapfile -t lines < "${output_path}"
  if [[ "${#lines[@]}" -ne 1 || ! "${lines[0]}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    return 2
  fi
  printf '%s\n' "${lines[0]}"
}

confirm_tag_absent() {
  local reference="$1"
  local output_path="$2"
  local error_path="$3"
  local -a lines=()
  if ! docker image ls --quiet --no-trunc --filter "reference=${reference}" \
    > "${output_path}" 2> "${error_path}"; then
    return 1
  fi
  mapfile -t lines < "${output_path}"
  [[ "${#lines[@]}" -eq 0 ]]
}

inspect_shared_tag() {
  local reference="$1"
  local prefix="$2"
  local inspected_id=""
  local inspect_status=0
  if inspected_id="$(inspect_image_id "${reference}" "${prefix}.out" "${prefix}.err")"; then
    printf '%s\n' "${inspected_id}"
    return 0
  else
    inspect_status=$?
  fi
  [[ "${inspect_status}" -ne 2 ]] || return 2
  confirm_tag_absent "${reference}" "${prefix}.list.out" "${prefix}.list.err" \
    || return 3
  return 1
}

write_success_receipt() {
  local destination="$1"
  python3 - "${revision}" "${transaction_id}" "${image_id}" "${image}" "${cas_outcome}" \
    > "${destination}" <<'PY'
import json
import sys

revision, transaction_id, image_id, intended_tag, cas_outcome = sys.argv[1:]
print(json.dumps({
    "schema": "npa.robomimic.neutral-build-receipt.v1",
    "revision": revision,
    "transaction_id": transaction_id,
    "immutable_image_id": image_id,
    "consumer_image_ref": image_id,
    "immutable_image_disposition": "retained-consumer-reference",
    "intended_full_sha_tag": intended_tag,
    "shared_tag_disposition": "serialized-and-verified",
    "tag_compare_and_set": cas_outcome,
    "transaction_evidence_disposition": "removed-after-receipt-publication",
}, sort_keys=True, separators=(",", ":")))
PY
}

publish_success_receipt() {
  local publish_status=0
  require_transaction_bindings
  start_receipt_staging "receipt" \
    || fail "transaction receipt staging refused"
  write_success_receipt "${receipt_anchor}/${receipt_tmp_name}"
  receipt_staging_matches \
    || fail "transaction receipt staging identity changed"
  if link_receipt_target "${receipt_name}"; then
    :
  else
    publish_status=$?
    [[ "${publish_status}" -ne 2 ]] \
      || fail "transaction receipt staging cleanup failed"
    fail "transaction receipt publication refused"
  fi
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || fail "receipt directory identity changed after publication"
  [[ "$(stat -c '%u:%a:%h' -- "${receipt_anchor}/${receipt_name}")" \
    == "$(id -u):600:1" ]] \
    || fail "transaction receipt is not owner-only"
}

publish_failure_receipt() {
  local context_disposition="$1"
  local disposition=""
  local published_name=""
  local publish_status=0
  local target_name="${transaction_id}.failure.json"
  local recovery_name="${transaction_id}.failure-recovery.json"
  [[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
  [[ "${failure_reason}" =~ ^[A-Za-z0-9][A-Za-z0-9\ .:_/-]*$ ]] \
    || failure_reason="unclassified-post-build-failure"
  clear_receipt_staging || return 1
  disposition="$(shared_tag_disposition)" || return 1
  start_receipt_staging "failure" || return 1
  write_failure_receipt \
    "${receipt_anchor}/${receipt_tmp_name}" "${context_disposition}" "${disposition}" \
    || return 1
  if link_receipt_target "${target_name}"; then
    failure_receipt="${receipt_dir}/${target_name}"
    published_name="${target_name}"
  else
    publish_status=$?
    [[ "${publish_status}" -eq 1 ]] || return 1
    if link_receipt_target "${recovery_name}"; then
      failure_receipt="${receipt_dir}/${recovery_name}"
      published_name="${recovery_name}"
    else
      return 1
    fi
  fi
  failure_receipt_published=1
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || return 1
  [[ -f "${receipt_anchor}/${published_name}" \
    && ! -L "${receipt_anchor}/${published_name}" ]] || return 1
  [[ "$(stat -c '%u:%a' -- "${receipt_anchor}/${published_name}")" \
    == "$(id -u):600" ]] \
    || return 1
}

recover_created_image() {
  local recovered_id=""
  [[ "${image_created}" -eq 0 && -n "${iid_file}" ]] || return 0
  if recovered_id="$(recover_image_id_file "${iid_file}")"; then
    image_id="${recovered_id}"
    image_created=1
  fi
}

remove_transaction_context() {
  local -a entries=()
  context_bindings_match || return 1
  shopt -s dotglob nullglob
  entries=("${context_anchor}"/*)
  shopt -u dotglob nullglob
  if [[ "${#entries[@]}" -gt 0 ]]; then
    rm -rf -- "${entries[@]}" || return 1
  fi
  context_bindings_match || return 1
  rmdir -- "${scratch_anchor}/${transaction_id}" || return 1
  [[ ! -e "${context}" && ! -L "${context}" ]]
}

retain_failed_image() {
  local cleanup_failed=0
  publish_failure_receipt "retained-owner-private-for-reconciliation" \
    || cleanup_failed=1
  if [[ "${failure_receipt_published}" -eq 1 ]]; then
    echo "robomimic neutral build retained image and transaction evidence; failure receipt ${failure_receipt}" >&2
  else
    echo "robomimic neutral build could not publish a failure receipt; transaction evidence remains at ${context}" >&2
  fi
  return "${cleanup_failed}"
}

discard_preimage_context() {
  local cleanup_failed=0
  clear_receipt_staging || cleanup_failed=1
  remove_transaction_context || cleanup_failed=1
  return "${cleanup_failed}"
}

discard_success_context() {
  remove_transaction_context && return 0
  failure_reason="transaction-context-cleanup-unresolved"
  publish_failure_receipt "cleanup-unresolved-after-success-receipt" || true
  return 1
}

cleanup() {
  local original_status=$?
  local cleanup_failed=0
  trap - EXIT HUP INT TERM
  set +e
  recover_created_image
  if [[ "${original_status}" -ne 0 && "${image_created}" -eq 1 ]]; then
    retain_failed_image || cleanup_failed=1
  elif [[ "${original_status}" -ne 0 ]]; then
    discard_preimage_context || cleanup_failed=1
  else
    discard_success_context || cleanup_failed=1
  fi
  [[ "${cleanup_failed}" -eq 0 ]] \
    || echo "robomimic neutral build cleanup or failure-receipt publication was incomplete" >&2
  [[ "${original_status}" -ne 0 ]] && exit "${original_status}"
  [[ "${cleanup_failed}" -eq 0 ]] || exit 1
  exit 0
}

repo_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || fail "expected a full Git revision"
registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"
[[ "${registry}" =~ ^[A-Za-z0-9][A-Za-z0-9./:_-]*$ ]] \
  || fail "local registry name is malformed"
image="${registry}/npa-robomimic:dev-${revision}"

if [[ -n "${TMPDIR+x}" ]]; then
  scratch_root="${TMPDIR}"
else
  scratch_root="/tmp/npa-robomimic-build-$(id -u)"
  mkdir -p -m 700 -- "${scratch_root}"
fi
require_trusted_directory_chain "${scratch_root}"
exec {scratch_root_fd}< "${scratch_root}" \
  || fail "scratch directory could not be opened"
scratch_root_identity="$(descriptor_identity "${scratch_root_fd}")" \
  || fail "scratch directory descriptor could not be verified"
scratch_anchor="/proc/$$/fd/${scratch_root_fd}"
directory_binding_matches \
  "${scratch_root_fd}" "${scratch_root}" "${scratch_root_identity}" \
  || fail "scratch directory changed while opening"
context_anchor="$(mktemp -d "${scratch_anchor}/npa-robomimic-context.XXXXXXXX")"
transaction_id="$(basename "${context_anchor}")"
context="${scratch_root}/${transaction_id}"
context_anchor="${scratch_anchor}/${transaction_id}"
exec {context_fd}< "${context_anchor}" \
  || fail "transaction context could not be opened"
context_identity="$(descriptor_identity "${context_fd}")" \
  || fail "transaction context descriptor could not be verified"
context_anchor="/proc/$$/fd/${context_fd}"
receipt=""
trap cleanup EXIT
trap 'failure_reason="signal-HUP"; exit 129' HUP
trap 'failure_reason="signal-INT"; exit 130' INT
trap 'failure_reason="signal-TERM"; exit 143' TERM

receipt_dir="${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-receipts}"
mkdir -p -m 700 -- "${receipt_dir}"
require_trusted_directory_chain "${receipt_dir}"
exec {receipt_dir_fd}< "${receipt_dir}" \
  || fail "receipt directory could not be opened"
receipt_dir_identity="$(descriptor_identity "${receipt_dir_fd}")" \
  || fail "receipt directory descriptor could not be verified"
receipt_anchor="/proc/$$/fd/${receipt_dir_fd}"
directory_binding_matches \
  "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
  || fail "receipt directory changed while opening"
receipt_name="${transaction_id}.json"
receipt="${receipt_dir}/${receipt_name}"
[[ ! -e "${receipt_anchor}/${receipt_name}" \
  && ! -L "${receipt_anchor}/${receipt_name}" ]] \
  || fail "transaction receipt already exists"
require_transaction_bindings

# Build only the exact committed image context named by the immutable tag.
require_transaction_bindings
git -C "${repo_root}" archive "${revision}:npa/docker/workbench/robomimic" \
  | tar -x --same-permissions -C "${context_anchor}"
require_transaction_bindings
python3 "${context_anchor}/verify_image.py" prepare-build-inputs \
  --output-root "${context_anchor}/build-inputs" \
  --debian-lock "${context_anchor}/debian-packages.lock" \
  --source-manifest "${context_anchor}/source-manifest.json"

require_transaction_bindings
iid_file="${context_anchor}/image.iid"
docker build --platform linux/amd64 --pull=false --iidfile "${iid_file}" "${context_anchor}"
require_transaction_bindings
image_id="$(read_image_id_file "${iid_file}")"
image_created=1
failure_reason="post-build-identity-verification-failed"
observed_id="$(inspect_image_id "${image_id}" "${context_anchor}/inspect-built.txt" \
  "${context_anchor}/inspect-built.err")" \
  || fail "built immutable image ID could not be verified"
require_transaction_bindings
[[ "${observed_id}" == "${image_id}" ]] \
  || fail "built immutable image ID changed during inspection"

owner_runtime_dir="/run/user/$(id -u)"
require_owner_private_directory "${owner_runtime_dir}"
lock_dir="${owner_runtime_dir}/npa-robomimic-build-locks"
mkdir -p -m 700 -- "${lock_dir}" \
  || fail "canonical shared-tag lock directory could not be created"
require_owner_private_directory "${lock_dir}"
tag_lock_name="$(printf '%s' "${image}" | sha256sum | cut -d ' ' -f 1)"
lock_path="${lock_dir}/${tag_lock_name}.lock"
if [[ ! -e "${lock_path}" && ! -L "${lock_path}" ]]; then
  ( set -o noclobber; : > "${lock_path}" ) 2>/dev/null || true
fi
[[ -f "${lock_path}" && ! -L "${lock_path}" ]] \
  || fail "shared-tag lock is unsafe"
[[ "$(stat -c '%u:%a:%h' -- "${lock_path}")" == "$(id -u):600:1" ]] \
  || fail "shared-tag lock ownership is unsafe"
exec {tag_lock_fd}<> "${lock_path}" \
  || fail "shared-tag lock could not be opened"
lock_fd_identity="$(stat -Lc '%d:%i:%u:%a:%h' -- "/proc/$$/fd/${tag_lock_fd}")" \
  || fail "shared-tag lock descriptor could not be verified"
lock_path_identity="$(stat -c '%d:%i:%u:%a:%h' -- "${lock_path}")" \
  || fail "shared-tag lock path could not be verified"
[[ "${lock_fd_identity}" == "${lock_path_identity}" ]] \
  || fail "shared-tag lock changed while opening"
flock -x "${tag_lock_fd}" \
  || fail "exclusive shared-tag coordination could not be established"
[[ "$(stat -c '%d:%i:%u:%a:%h' -- "${lock_path}")" == "${lock_fd_identity}" ]] \
  || fail "shared-tag lock changed during coordination"

cas_outcome=""
require_transaction_bindings
if existing_id="$(inspect_shared_tag "${image}" "${context_anchor}/inspect-tag-before")"; then
  require_transaction_bindings
  [[ "${existing_id}" == "${image_id}" ]] \
    || fail "full-SHA tag already names different bytes"
  cas_outcome="existing-identical"
else
  inspect_status=$?
  require_transaction_bindings
  [[ "${inspect_status}" -ne 2 ]] \
    || fail "existing full-SHA tag identity is malformed"
  [[ "${inspect_status}" -eq 1 ]] \
    || fail "existing full-SHA tag absence could not be proven"
  require_transaction_bindings
  tag_assignment_attempted=1
  docker image tag "${image_id}" "${image}" \
    || fail "full-SHA tag assignment failed or was interrupted"
  require_transaction_bindings
  tag_assignment_completed=1
  cas_outcome="assigned"
fi

require_transaction_bindings
tagged_id="$(inspect_shared_tag "${image}" "${context_anchor}/inspect-tag-after")" \
  || fail "full-SHA tag could not be verified after serialized assignment"
require_transaction_bindings
[[ "${tagged_id}" == "${image_id}" ]] \
  || fail "full-SHA tag changed during serialized assignment"
final_iid="$(read_image_id_file "${iid_file}")"
final_built_id="$(inspect_image_id "${image_id}" "${context_anchor}/inspect-built-final.txt" \
  "${context_anchor}/inspect-built-final.err")" \
  || fail "built immutable image ID could not be reverified"
require_transaction_bindings
final_tagged_id="$(inspect_shared_tag "${image}" "${context_anchor}/inspect-tag-final")" \
  || fail "full-SHA tag could not be reverified"
require_transaction_bindings
[[ "${final_iid}" == "${image_id}" \
  && "${final_built_id}" == "${image_id}" \
  && "${final_tagged_id}" == "${image_id}" ]] \
  || fail "image identity changed before receipt publication"

publish_success_receipt

transaction_complete=1
echo "built local candidate ${image_id}; receipt ${receipt}; this helper does not push or publish"
