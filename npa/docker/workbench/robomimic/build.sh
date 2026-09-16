#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() {
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

record_unresolved_partial_state() {
  local reason="$1"
  unresolved_tmp="$(mktemp "${receipt_dir}/.${transaction_id}.unresolved.XXXXXXXX")" \
    || return 1
  printf '{"immutable_image_id":"%s","intended_full_sha_tag":"%s","reason":"%s","revision":"%s","schema":"npa.robomimic.neutral-build-unresolved.v1","transaction_id":"%s"}\n' \
    "${image_id}" "${image}" "${reason}" "${revision}" "${transaction_id}" \
    > "${unresolved_tmp}" || return 1
  chmod 600 -- "${unresolved_tmp}" || return 1
  ln -- "${unresolved_tmp}" "${unresolved_receipt}" || return 1
  rm -f -- "${unresolved_tmp}" || return 1
  unresolved_tmp=""
}

rollback_transaction_tag() {
  local current_id=""
  local inspect_status=0
  if current_id="$(inspect_shared_tag "${image}" "${context}/rollback-before")"; then
    if [[ "${current_id}" != "${image_id}" ]]; then
      echo "robomimic neutral build rollback skipped: shared tag identity changed" >&2
      return 0
    fi
  else
    inspect_status=$?
    [[ "${inspect_status}" -eq 1 ]] && return 0
    record_unresolved_partial_state "rollback-inspection-unresolved"
    return 1
  fi
  if ! docker image rm "${image}" \
    > "${context}/rollback-remove.out" 2> "${context}/rollback-remove.err"; then
    record_unresolved_partial_state "rollback-remove-failed"
    return 1
  fi
  if inspect_shared_tag "${image}" "${context}/rollback-after" > /dev/null; then
    record_unresolved_partial_state "rollback-verification-failed"
    return 1
  else
    inspect_status=$?
  fi
  [[ "${inspect_status}" -eq 1 ]] && return 0
  record_unresolved_partial_state "rollback-verification-unresolved"
  return 1
}

repo_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || fail "expected a full Git revision"
registry="${NPA_BYOF_ROBOMIMIC_REGISTRY:-local.invalid}"
[[ "${registry}" =~ ^[A-Za-z0-9][A-Za-z0-9./:_-]*$ ]] \
  || fail "local registry name is malformed"
image="${registry}/npa-robomimic:dev-${revision}"

context="$(mktemp -d "${TMPDIR:-/tmp}/npa-robomimic-context.XXXXXXXX")"
transaction_id="$(basename "${context}")"
receipt_tmp=""
unresolved_tmp=""
receipt=""
unresolved_receipt=""
receipt_published=0
tag_assigned=0
transaction_complete=0
cleanup() {
  local original_status=$?
  local cleanup_failed=0
  trap - EXIT
  set +e
  if [[ "${transaction_complete}" -eq 0 && "${tag_assigned}" -eq 1 ]]; then
    rollback_transaction_tag || cleanup_failed=1
  fi
  if [[ "${transaction_complete}" -eq 0 && "${receipt_published}" -eq 1 ]]; then
    rm -f -- "${receipt}" || cleanup_failed=1
  fi
  if [[ -n "${receipt_tmp}" ]]; then
    rm -f -- "${receipt_tmp}" || cleanup_failed=1
  fi
  if [[ -n "${unresolved_tmp}" ]]; then
    rm -f -- "${unresolved_tmp}" || cleanup_failed=1
  fi
  rm -rf -- "${context}" || cleanup_failed=1
  [[ "${cleanup_failed}" -eq 0 ]] \
    || echo "robomimic neutral build cleanup was incomplete" >&2
  if [[ "${original_status}" -ne 0 ]]; then
    exit "${original_status}"
  fi
  [[ "${cleanup_failed}" -eq 0 ]] || exit 1
  exit 0
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

receipt_dir="${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-receipts}"
lock_dir="${NPA_BYOF_ROBOMIMIC_LOCK_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-locks}"
mkdir -p -m 700 -- "${receipt_dir}" "${lock_dir}"
require_owner_private_directory "${receipt_dir}"
require_owner_private_directory "${lock_dir}"
receipt="${receipt_dir}/${transaction_id}.json"
unresolved_receipt="${receipt_dir}/${transaction_id}.unresolved.json"
[[ ! -e "${receipt}" && ! -L "${receipt}" ]] \
  || fail "transaction receipt already exists"
[[ ! -e "${unresolved_receipt}" && ! -L "${unresolved_receipt}" ]] \
  || fail "unresolved transaction receipt already exists"

# Build only the exact committed image context named by the immutable tag.
git -C "${repo_root}" archive "${revision}:npa/docker/workbench/robomimic" \
  | tar -x --same-permissions -C "${context}"
python3 "${context}/verify_image.py" prepare-build-inputs \
  --output-root "${context}/build-inputs" \
  --debian-lock "${context}/debian-packages.lock" \
  --source-manifest "${context}/source-manifest.json"

iid_file="${context}/image.iid"
docker build --platform linux/amd64 --pull=false --iidfile "${iid_file}" "${context}"
image_id="$(read_image_id_file "${iid_file}")"
observed_id="$(inspect_image_id "${image_id}" "${context}/inspect-built.txt" \
  "${context}/inspect-built.err")" \
  || fail "built immutable image ID could not be verified"
[[ "${observed_id}" == "${image_id}" ]] \
  || fail "built immutable image ID changed during inspection"

tag_lock_name="$(printf '%s' "${image}" | sha256sum | cut -d ' ' -f 1)"
lock_path="${lock_dir}/${tag_lock_name}.lock"
[[ ! -e "${lock_path}" || ( -f "${lock_path}" && ! -L "${lock_path}" ) ]] \
  || fail "shared-tag lock is unsafe"
exec {tag_lock_fd}> "${lock_path}"
flock -x "${tag_lock_fd}"

cas_outcome=""
if existing_id="$(inspect_shared_tag "${image}" "${context}/inspect-tag-before")"; then
  [[ "${existing_id}" == "${image_id}" ]] \
    || fail "full-SHA tag already names different bytes"
  cas_outcome="existing-identical"
else
  inspect_status=$?
  [[ "${inspect_status}" -ne 2 ]] \
    || fail "existing full-SHA tag identity is malformed"
  [[ "${inspect_status}" -eq 1 ]] \
    || fail "existing full-SHA tag absence could not be proven"
  # Treat the assignment as transaction-owned before Docker runs so a signal or
  # ambiguous command failure still enters the guarded identity rollback.
  tag_assigned=1
  docker image tag "${image_id}" "${image}"
  cas_outcome="assigned"
fi

tagged_id="$(inspect_shared_tag "${image}" "${context}/inspect-tag-after")" \
  || fail "full-SHA tag could not be verified after compare-and-set"
[[ "${tagged_id}" == "${image_id}" ]] \
  || fail "full-SHA tag changed during compare-and-set"
final_iid="$(read_image_id_file "${iid_file}")"
final_built_id="$(inspect_image_id "${image_id}" "${context}/inspect-built-final.txt" \
  "${context}/inspect-built-final.err")" \
  || fail "built immutable image ID could not be reverified"
final_tagged_id="$(inspect_shared_tag "${image}" "${context}/inspect-tag-final")" \
  || fail "full-SHA tag could not be reverified"
[[ "${final_iid}" == "${image_id}" \
  && "${final_built_id}" == "${image_id}" \
  && "${final_tagged_id}" == "${image_id}" ]] \
  || fail "image identity changed before receipt publication"

receipt_tmp="$(mktemp "${receipt_dir}/.${transaction_id}.receipt.XXXXXXXX")"
python3 - "${revision}" "${transaction_id}" "${image_id}" "${image}" "${cas_outcome}" \
  > "${receipt_tmp}" <<'PY'
import json
import sys

revision, transaction_id, image_id, intended_tag, cas_outcome = sys.argv[1:]
print(
    json.dumps(
        {
            "schema": "npa.robomimic.neutral-build-receipt.v1",
            "revision": revision,
            "transaction_id": transaction_id,
            "immutable_image_id": image_id,
            "consumer_image_ref": image_id,
            "intended_full_sha_tag": intended_tag,
            "tag_compare_and_set": cas_outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
chmod 600 -- "${receipt_tmp}"
ln -- "${receipt_tmp}" "${receipt}" \
  || fail "transaction receipt publication refused"
receipt_published=1
rm -f -- "${receipt_tmp}"
receipt_tmp=""
[[ "$(stat -c '%u:%a:%h' -- "${receipt}")" == "$(id -u):600:1" ]] \
  || fail "transaction receipt is not owner-only"

echo "built local candidate ${image_id}; receipt ${receipt}; this helper does not push or publish"
transaction_complete=1
