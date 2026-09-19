#!/usr/bin/env bash
# Build the neutral robomimic image with daemon-serialized tags and durable receipts.
set -euo pipefail
umask 077

failure_reason="pre-image-failure"
image_created=0
image_id=""
iid_file=""
tag_assignment_attempted=0
tag_assignment_completed=0
transaction_complete=0
success_receipt_identity=""
failure_receipt_published=0
failure_receipt=""
tag_lock_acquisition_attempted=0
tag_lock_held=0
tag_lock_id=""
tag_lock_name=""
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
tag_daemon_id=""

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

forget_receipt_staging() {
  receipt_tmp=""
  receipt_tmp_name=""
  receipt_tmp_identity=""
}

receipt_link_program="$(cat <<'PY'
import os
import stat
import sys
_, directory_fd, source_name, target_name, source_identity, directory_identity = sys.argv[1:]
directory_fd = int(directory_fd)

def identity(value: os.stat_result, kind: str) -> str:
    return f"{value.st_dev}:{value.st_ino}:{value.st_uid}:{stat.S_IMODE(value.st_mode):o}:{kind}"

def matches_source(value: os.stat_result, links: int) -> bool:
    return (stat.S_ISREG(value.st_mode)
            and identity(value, "regular file") == source_identity
            and value.st_nlink == links)

if any(not name or os.path.basename(name) != name for name in (source_name, target_name)):
    raise SystemExit(1)
if identity(os.fstat(directory_fd), "directory") != directory_identity:
    raise SystemExit(1)
source_fd = os.open(source_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
try:
    if not matches_source(os.fstat(source_fd), 1):
        raise SystemExit(1)
    os.fsync(source_fd)
    os.link(source_name, target_name, src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd, follow_symlinks=False)
finally:
    os.close(source_fd)
if not all(matches_source(os.stat(name, dir_fd=directory_fd, follow_symlinks=False), 2)
           for name in (source_name, target_name)):
    raise SystemExit(1)
os.unlink(source_name, dir_fd=directory_fd)
if not matches_source(os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False), 1):
    raise SystemExit(1)
os.fsync(directory_fd)
if identity(os.fstat(directory_fd), "directory") != directory_identity:
    raise SystemExit(1)
if not matches_source(os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False), 1):
    raise SystemExit(1)
PY
)"

fd_relative_link_receipt() {
  local target_name="$1" inherited_directory_fd=9
  python3 - receipt-link "${inherited_directory_fd}" "${receipt_tmp_name}" \
    "${target_name}" "${receipt_tmp_identity}" "${receipt_dir_identity}" \
    9<&"${receipt_dir_fd}" <<< "${receipt_link_program}"
}

link_receipt_target() {
  local target_name="$1"
  local target_anchor="${receipt_anchor}/${target_name}"
  local target_identity="${receipt_tmp_identity}"
  receipt_staging_matches || return 1
  [[ ! -e "${target_anchor}" && ! -L "${target_anchor}" ]] || return 1
  fd_relative_link_receipt "${target_name}" || return 1
  forget_receipt_staging
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" \
    || return 1
  [[ -f "${target_anchor}" && ! -L "${target_anchor}" ]] || return 1
  [[ "$(file_identity "${target_anchor}" 2>/dev/null)" == "${target_identity}" ]] \
    || return 1
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

canonical_registry_reference() {
  local reference="$1"
  local first=""
  local path=""
  [[ "${reference}" =~ ^[a-z0-9]+([._-][a-z0-9]+)*(:[0-9]+)?(/[a-z0-9]+([._-][a-z0-9]+)*)*$ ]] \
    || return 1
  first="${reference%%/*}"
  case "${first}" in
    docker.io|index.docker.io)
      path="${reference#${first}/}"
      [[ "${path}" != "${reference}" && -n "${path}" ]] || return 1
      [[ "${path}" == */* ]] || path="library/${path}"
      printf 'docker.io/%s\n' "${path}"
      ;;
    *.*|*:*|localhost)
      printf '%s\n' "${reference}"
      ;;
    *)
      printf 'docker.io/%s\n' "${reference}"
      ;;
  esac
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
    "transaction_evidence_disposition": "cleanup-pending-terminal-result",
}, sort_keys=True, separators=(",", ":")))
PY
}

write_cleanup_receipt() {
  local destination="$1" status="$2" disposition="$3"
  python3 - "${revision}" "${transaction_id}" "${image_id}" "${status}" \
    "${disposition}" "${transaction_id}.cleanup-journal.json" \
    > "${destination}" <<'PY'
import json
import sys

revision, transaction_id, image_id, status, disposition, journal_name = sys.argv[1:]
print(json.dumps({
    "schema": "npa.robomimic.neutral-build-cleanup.v1",
    "revision": revision,
    "transaction_id": transaction_id,
    "immutable_image_id": image_id,
    "cleanup_journal": journal_name,
    "status": status,
    "transaction_evidence_disposition": disposition,
}, sort_keys=True, separators=(",", ":")))
PY
}

write_cleanup_journal() {
  local destination="$1"
  python3 - "${revision}" "${transaction_id}" "${image_id}" \
    "${transaction_id}.cleanup.json" > "${destination}" <<'PY'
import json
import sys

revision, transaction_id, image_id, terminal_name = sys.argv[1:]
print(json.dumps({
    "schema": "npa.robomimic.neutral-build-cleanup-journal.v1",
    "revision": revision,
    "transaction_id": transaction_id,
    "immutable_image_id": image_id,
    "cleanup_intent": "remove-transaction-context",
    "intent_state": "durably-recorded-before-context-deletion",
    "terminal_outcome_record": terminal_name,
}, sort_keys=True, separators=(",", ":")))
PY
}

publish_cleanup_journal() {
  local target_name="${transaction_id}.cleanup-journal.json"
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" || return 1
  [[ -z "${receipt_tmp_name}" ]] || return 1
  start_receipt_staging "cleanup-journal" || return 1
  write_cleanup_journal "${receipt_anchor}/${receipt_tmp_name}" || return 1
  receipt_staging_matches || return 1
  link_receipt_target "${target_name}" || return 1
  [[ "$(stat -c '%u:%a:%h' -- "${receipt_anchor}/${target_name}" 2>/dev/null)" \
    == "$(id -u):600:1" ]]
}

publish_cleanup_receipt() {
  local status="$1" disposition="$2"
  local target_name="${transaction_id}.cleanup.json"
  directory_binding_matches \
    "${receipt_dir_fd}" "${receipt_dir}" "${receipt_dir_identity}" || return 1
  [[ -z "${receipt_tmp_name}" ]] || return 1
  start_receipt_staging "cleanup" || return 1
  write_cleanup_receipt \
    "${receipt_anchor}/${receipt_tmp_name}" "${status}" "${disposition}" || return 1
  receipt_staging_matches || return 1
  link_receipt_target "${target_name}" || return 1
  [[ "$(stat -c '%u:%a:%h' -- "${receipt_anchor}/${target_name}" 2>/dev/null)" \
    == "$(id -u):600:1" ]]
}

publish_success_receipt() {
  local publish_status=0
  require_transaction_bindings
  start_receipt_staging "receipt" \
    || fail "transaction receipt staging refused"
  write_success_receipt "${receipt_anchor}/${receipt_tmp_name}"
  receipt_staging_matches \
    || fail "transaction receipt staging identity changed"
  success_receipt_identity="${receipt_tmp_identity}"
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
  [[ -z "${receipt_tmp_name}" ]] || return 1
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

# This is a local-owner terminal proof, not a lease expiry or cross-host oracle.
tag_terminal_program="$(cat <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

SCHEMA = "npa.robomimic.tag-terminal.v1"

def host_identity():
    return [Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            os.readlink("/proc/self/ns/pid")]

def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]

def daemon_output(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise ValueError("ambiguous daemon response")
    return lines[0]

def daemon_identity():
    value = daemon_output("info", "--format", "{{.ID}}")
    if not re.fullmatch(r"[A-Za-z0-9:_-]{10,128}", value):
        raise ValueError("invalid daemon identity")
    return value

def terminal_record(values):
    pid, daemon, revision, transaction, lock_id, tag, image = values
    if daemon_identity() != daemon:
        raise ValueError("daemon changed")
    return dict(schema=SCHEMA, phase="no-further-tag-writes", host=host_identity(),
                uid=os.geteuid(), pid=int(pid), start_ticks=process_start(int(pid)),
                daemon=daemon, revision=revision, transaction=transaction,
                lock_id=lock_id, tag=tag, image=image)

def private_parent(path):
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe receipt path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            value = os.fstat(descriptor)
            mode = stat.S_IMODE(value.st_mode)
            trusted = value.st_uid in {0, os.geteuid()}
            sticky_root = value.st_uid == 0 and bool(mode & stat.S_ISVTX)
            if not trusted or mode & 0o022 and not sticky_root:
                raise ValueError("unsafe receipt parent")
        value = os.fstat(descriptor)
        if value.st_uid != os.geteuid() or stat.S_IMODE(value.st_mode) != 0o700:
            raise ValueError("receipt parent is not owner-private")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def unique_fields(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("ambiguous terminal fields")
    return result

def stable_file_identity(value):
    return (value.st_dev, value.st_ino, value.st_uid, value.st_mode,
            value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

def read_terminal(path):
    parent = private_parent(path)
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1
                    or not 0 < before.st_size <= 4096):
                raise ValueError("unsafe terminal receipt")
            raw = os.read(descriptor, 4097)
            if (len(raw) != before.st_size
                    or stable_file_identity(os.fstat(descriptor)) != stable_file_identity(before)):
                raise ValueError("terminal receipt changed")
            identity = (stable_file_identity(before), os.fstat(parent).st_dev, os.fstat(parent).st_ino)
            return json.loads(raw, object_pairs_hook=unique_fields), hashlib.sha256(raw).hexdigest(), identity
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)

def require_terminal_owner(record, revision, tag):
    required = {"schema", "phase", "host", "uid", "pid", "start_ticks", "daemon",
                "revision", "transaction", "lock_id", "tag", "image"}
    if (set(record) != required or record["schema"] != SCHEMA
            or record["phase"] != "no-further-tag-writes"
            or record["host"] != host_identity() or record["uid"] != os.geteuid()
            or record["revision"] != revision or record["tag"] != tag):
        raise ValueError("terminal ownership binding refused")
    if (type(record["pid"]) is not int or record["pid"] <= 1
            or not re.fullmatch(r"[0-9]+", record["start_ticks"])
            or not re.fullmatch(r"npa-robomimic-context\.[A-Za-z0-9]{8}", record["transaction"])
            or not re.fullmatch(r"[0-9a-f]{64}", record["lock_id"])
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", record["image"])):
        raise ValueError("malformed terminal binding")
    try:
        process_start(record["pid"])
    except FileNotFoundError:
        return
    raise ValueError("owner live, unreaped, or PID reused")

def require_daemon_binding(record):
    if daemon_identity() != record["daemon"]:
        raise ValueError("daemon changed")
    name = "npa-robomimic-tag-lock-" + hashlib.sha256(record["tag"].encode()).hexdigest()
    template = '{{.Id}}|{{index .Config.Labels "org.nebius.npa.robomimic.lock.transaction"}}|{{.Image}}|{{.State.Running}}|{{.Name}}'
    expected = f'{record["lock_id"]}|{record["transaction"]}|{record["image"]}|false|/{name}'
    if daemon_output("container", "inspect", "--format", template, name) != expected:
        raise ValueError("lock changed or not stopped")
    if daemon_output("image", "inspect", "--format", "{{.Id}}", record["tag"]) != record["image"]:
        raise ValueError("tag changed")

def reconcile(path, revision, tag):
    record, digest, identity = read_terminal(path)
    require_terminal_owner(record, revision, tag)
    if path.name != record["transaction"] + ".tag-terminal.json":
        raise ValueError("transaction receipt name differs")
    require_daemon_binding(record)
    if read_terminal(path) != (record, digest, identity):
        raise ValueError("receipt changed")
    require_terminal_owner(record, revision, tag)
    require_daemon_binding(record)
    # The terminal fence follows synchronous completion of every tag writer.
    # Only this immutable stopped lock object can be removed; never a tag/image.
    subprocess.run(["docker", "container", "rm", record["lock_id"]],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("reconciled exact owner-terminal tag mutex; no image or tag modified")

def main():
    action, *values = sys.argv[1:]
    try:
        if action == "daemon-id":
            print(daemon_identity())
        elif action == "record":
            print(json.dumps(terminal_record(values), sort_keys=True, separators=(",", ":")))
        elif action == "reconcile":
            reconcile(Path(values[0]), values[1], values[2])
        else:
            raise ValueError("unsupported terminal operation")
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        raise SystemExit("robomimic tag-terminal reconciliation refused; retain owner evidence") from None

if __name__ == "__main__":
    main()
PY
)"

publish_tag_terminal_receipt() {
  local target_name="${transaction_id}.tag-terminal.json"
  require_transaction_bindings
  start_receipt_staging "tag-terminal" || return 1
  python3 - record "$$" "${tag_daemon_id}" "${revision}" "${transaction_id}" \
    "${tag_lock_id}" "${image}" "${image_id}" \
    > "${receipt_anchor}/${receipt_tmp_name}" <<< "${tag_terminal_program}" || return 1
  link_receipt_target "${target_name}"
}

inspect_tag_lock() {
  local reference="$1" output_path="$2" error_path="$3"
  local -a lines=()
  docker container inspect --format \
    '{{.Id}}|{{index .Config.Labels "org.nebius.npa.robomimic.lock.transaction"}}|{{.Image}}' \
    "${reference}" > "${output_path}" 2> "${error_path}" || return 1
  mapfile -t lines < "${output_path}"
  [[ "${#lines[@]}" -eq 1 \
    && "${lines[0]}" == "${tag_lock_id}|${transaction_id}|${image_id}" ]]
}

acquire_tag_lock() {
  local created_id=""
  tag_lock_name="npa-robomimic-tag-lock-$(printf '%s' "${image}" | sha256sum | cut -d ' ' -f 1)"
  tag_lock_acquisition_attempted=1
  created_id="$(docker container create --name "${tag_lock_name}" \
    --label "org.nebius.npa.robomimic.lock.transaction=${transaction_id}" \
    --entrypoint /bin/true "${image_id}")" || return 1
  [[ "${created_id}" =~ ^[0-9a-f]{64}$ ]] || return 1
  tag_lock_id="${created_id}"
  inspect_tag_lock "${tag_lock_id}" "${context_anchor}/inspect-lock.out" \
    "${context_anchor}/inspect-lock.err" || return 1
  tag_lock_held=1
}

recover_tag_lock() {
  local -a lines=()
  [[ "${tag_lock_acquisition_attempted}" -eq 1 && "${tag_lock_held}" -eq 0 ]] \
    || return 0
  docker container inspect --format \
    '{{.Id}}|{{index .Config.Labels "org.nebius.npa.robomimic.lock.transaction"}}|{{.Image}}' \
    "${tag_lock_name}" > "${context_anchor}/recover-lock.out" \
    2> "${context_anchor}/recover-lock.err" || return 0
  mapfile -t lines < "${context_anchor}/recover-lock.out" || return 1
  [[ "${#lines[@]}" -eq 1 ]] || return 1
  IFS='|' read -r tag_lock_id observed_transaction observed_image <<< "${lines[0]}"
  [[ "${tag_lock_id}" =~ ^[0-9a-f]{64}$ \
    && "${observed_transaction}" == "${transaction_id}" \
    && "${observed_image}" == "${image_id}" ]] || return 0
  tag_lock_held=1
}

release_tag_lock() {
  [[ "${tag_lock_held}" -eq 1 ]] || return 0
  inspect_tag_lock "${tag_lock_id}" "${context_anchor}/release-lock.out" \
    "${context_anchor}/release-lock.err" || return 1
  docker container rm "${tag_lock_id}" \
    > "${context_anchor}/release-lock-rm.out" \
    2> "${context_anchor}/release-lock-rm.err" || return 1
  tag_lock_acquisition_attempted=0
  tag_lock_held=0
  tag_lock_id=""
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
  publish_cleanup_journal || return 1
  if remove_transaction_context; then
    publish_cleanup_receipt "completed" "removed" && return 0
  else
    publish_cleanup_receipt "unresolved" \
      "retained-owner-private-for-reconciliation" || true
    return 1
  fi
}

success_receipt_is_published() {
  [[ -n "${success_receipt_identity}" ]] || return 1
  [[ "$(descriptor_identity "${receipt_dir_fd}" 2>/dev/null)" \
    == "${receipt_dir_identity}" ]] || return 1
  local target="${receipt_anchor}/${receipt_name}"
  [[ -f "${target}" && ! -L "${target}" ]] || return 1
  [[ "$(file_identity "${target}" 2>/dev/null)" \
    == "${success_receipt_identity}" ]]
}

retain_success_context() {
  # Publication is the build outcome even if its caller is interrupted before
  # observing completion. Never append a contradictory build-failure receipt.
  # Retain the context and record cleanup separately from that immutable outcome.
  if [[ -n "${receipt_tmp_name}" ]]; then
    [[ ! -e "${receipt_anchor}/${receipt_tmp_name}" \
      && ! -L "${receipt_anchor}/${receipt_tmp_name}" ]] || return 1
    forget_receipt_staging
  fi
  publish_cleanup_journal || return 1
  publish_cleanup_receipt "unresolved" "retained-owner-private-for-reconciliation"
}

cleanup() {
  local original_status=$?
  local cleanup_failed=0
  trap - EXIT HUP INT TERM
  set +e
  recover_created_image
  recover_tag_lock || cleanup_failed=1
  release_tag_lock || cleanup_failed=1
  if [[ "${original_status}" -ne 0 ]] \
    && { [[ "${transaction_complete}" -eq 1 ]] || success_receipt_is_published; }; then
    retain_success_context || cleanup_failed=1
  elif [[ "${original_status}" -ne 0 && "${image_created}" -eq 1 ]]; then
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
registry="$(canonical_registry_reference "${registry}")" \
  || fail "local registry name is malformed or noncanonicalizable"
image="${registry}/npa-robomimic:dev-${revision}"

if [[ "$#" -ne 0 ]]; then
  [[ "$#" -eq 2 && "$1" == "--reconcile-tag-lock" ]] \
    || fail "expected --reconcile-tag-lock with an owner terminal receipt"
  exec python3 - reconcile "$2" "${revision}" "${image}" <<< "${tag_terminal_program}"
fi

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

tag_daemon_id="$(python3 - daemon-id <<< "${tag_terminal_program}")" \
  || fail "daemon identity could not be established"
acquire_tag_lock \
  || fail "daemon-wide shared-tag coordination could not be established"

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
publish_tag_terminal_receipt \
  || fail "owner tag-terminal fence could not be published"
release_tag_lock \
  || fail "daemon-wide shared-tag coordination could not be released"

publish_success_receipt

transaction_complete=1
echo "built local candidate ${image_id}; receipt ${receipt}; this helper does not push or publish"
