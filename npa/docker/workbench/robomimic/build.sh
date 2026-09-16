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

publish_failure_receipt() {
  local context_disposition="$1"
  local shared_tag_disposition="not-modified-by-cleanup"
  local target="${receipt_dir}/${transaction_id}.failure.json"
  local recovery_target="${receipt_dir}/${transaction_id}.failure-recovery.json"
  local terminal_tmp=""

  [[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || return 1
  [[ "${failure_reason}" =~ ^[A-Za-z0-9][A-Za-z0-9\ .:_/-]*$ ]] \
    || failure_reason="unclassified-post-build-failure"
  if [[ "${tag_assignment_attempted}" -eq 1 && "${tag_assignment_completed}" -eq 1 ]]; then
    shared_tag_disposition="assignment-completed-no-cleanup-modification"
  elif [[ "${tag_assignment_attempted}" -eq 1 ]]; then
    shared_tag_disposition="assignment-outcome-unknown-no-cleanup-modification"
  fi
  [[ ! -e "${target}" && ! -L "${target}" ]] || return 1
  [[ ! -e "${recovery_target}" && ! -L "${recovery_target}" ]] || return 1
  terminal_tmp="$(mktemp "${receipt_dir}/.${transaction_id}.failure.XXXXXXXX")" \
    || return 1
  if ! printf '{"context_evidence_disposition":"%s","failure_reason":"%s","immutable_image_disposition":"retained-no-exclusive-ownership-proof","immutable_image_id":"%s","intended_full_sha_tag":"%s","revision":"%s","schema":"npa.robomimic.neutral-build-failure.v1","shared_tag_disposition":"%s","status":"failure","transaction_id":"%s"}\n' \
    "${context_disposition}" "${failure_reason}" "${image_id}" "${image}" \
    "${revision}" "${shared_tag_disposition}" "${transaction_id}" \
    > "${terminal_tmp}"; then
    return 1
  fi
  chmod 600 -- "${terminal_tmp}" || return 1
  if ln -- "${terminal_tmp}" "${target}"; then
    failure_receipt="${target}"
  elif ln -- "${terminal_tmp}" "${recovery_target}"; then
    failure_receipt="${recovery_target}"
  else
    return 1
  fi
  failure_receipt_published=1
  if ! rm -f -- "${terminal_tmp}"; then
    echo "robomimic neutral build retained a private failure-receipt staging link" >&2
  fi
  [[ -f "${failure_receipt}" && ! -L "${failure_receipt}" ]] || return 1
  [[ "$(stat -c '%u:%a' -- "${failure_receipt}")" == "$(id -u):600" ]] \
    || return 1
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
receipt=""
receipt_dir=""
cleanup() {
  local original_status=$?
  local cleanup_failed=0
  local recovered_id=""
  trap - EXIT HUP INT TERM
  set +e

  if [[ "${image_created}" -eq 0 && -n "${iid_file}" ]]; then
    if recovered_id="$(recover_image_id_file "${iid_file}")"; then
      image_id="${recovered_id}"
      image_created=1
    fi
  fi

  if [[ "${original_status}" -ne 0 && "${image_created}" -eq 1 ]]; then
    publish_failure_receipt "retained-owner-private-for-reconciliation" \
      || cleanup_failed=1
    if [[ "${failure_receipt_published}" -eq 1 ]]; then
      echo "robomimic neutral build retained image and transaction evidence; failure receipt ${failure_receipt}" >&2
    else
      echo "robomimic neutral build could not publish a failure receipt; transaction evidence remains at ${context}" >&2
    fi
  elif [[ "${original_status}" -ne 0 ]]; then
    if [[ -n "${receipt_tmp}" ]]; then
      rm -f -- "${receipt_tmp}" || cleanup_failed=1
    fi
    rm -rf -- "${context}" || cleanup_failed=1
  else
    if ! rm -rf -- "${context}"; then
      failure_reason="transaction-context-cleanup-unresolved"
      publish_failure_receipt "cleanup-unresolved-after-success-receipt" \
        || cleanup_failed=1
      cleanup_failed=1
    fi
  fi

  [[ "${cleanup_failed}" -eq 0 ]] \
    || echo "robomimic neutral build cleanup or failure-receipt publication was incomplete" >&2
  if [[ "${original_status}" -ne 0 ]]; then
    exit "${original_status}"
  fi
  [[ "${cleanup_failed}" -eq 0 ]] || exit 1
  exit 0
}
trap cleanup EXIT
trap 'failure_reason="signal-HUP"; exit 129' HUP
trap 'failure_reason="signal-INT"; exit 130' INT
trap 'failure_reason="signal-TERM"; exit 143' TERM

receipt_dir="${NPA_BYOF_ROBOMIMIC_RECEIPT_DIR:-${TMPDIR:-/tmp}/npa-robomimic-build-receipts}"
mkdir -p -m 700 -- "${receipt_dir}"
require_owner_private_directory "${receipt_dir}"
receipt="${receipt_dir}/${transaction_id}.json"
[[ ! -e "${receipt}" && ! -L "${receipt}" ]] \
  || fail "transaction receipt already exists"

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
image_created=1
failure_reason="post-build-identity-verification-failed"
observed_id="$(inspect_image_id "${image_id}" "${context}/inspect-built.txt" \
  "${context}/inspect-built.err")" \
  || fail "built immutable image ID could not be verified"
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
  tag_assignment_attempted=1
  docker image tag "${image_id}" "${image}" \
    || fail "full-SHA tag assignment failed or was interrupted"
  tag_assignment_completed=1
  cas_outcome="assigned"
fi

tagged_id="$(inspect_shared_tag "${image}" "${context}/inspect-tag-after")" \
  || fail "full-SHA tag could not be verified after serialized assignment"
[[ "${tagged_id}" == "${image_id}" ]] \
  || fail "full-SHA tag changed during serialized assignment"
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
            "immutable_image_disposition": "retained-consumer-reference",
            "intended_full_sha_tag": intended_tag,
            "shared_tag_disposition": "serialized-and-verified",
            "tag_compare_and_set": cas_outcome,
            "transaction_evidence_disposition": "removed-after-receipt-publication",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
chmod 600 -- "${receipt_tmp}"
ln -- "${receipt_tmp}" "${receipt}" \
  || fail "transaction receipt publication refused"
rm -f -- "${receipt_tmp}" \
  || fail "transaction receipt staging cleanup failed"
receipt_tmp=""
[[ "$(stat -c '%u:%a:%h' -- "${receipt}")" == "$(id -u):600:1" ]] \
  || fail "transaction receipt is not owner-only"

transaction_complete=1
echo "built local candidate ${image_id}; receipt ${receipt}; this helper does not push or publish"
