#!/bin/sh
set -u

real_apt_get=/usr/bin/apt-get
real_timeout=/usr/bin/timeout
real_ssh_keygen=/usr/bin/ssh-keygen
guard_runtime_dir=/run/npa-skypilot-bootstrap
guard_owner_uid=0
guard_state=$guard_runtime_dir/apt.state
guard_failure=/tmp/npa-skypilot-bootstrap-contract.failed
sky_failure=/tmp/apt-ssh-setup.failed

atomic_replace_text() {
    target=$1
    content=$2
    target_directory=${target%/*}
    target_name=${target##*/}
    temporary="$(umask 077; mktemp "$target_directory/.${target_name}.XXXXXX")" \
        || return 1
    temporary_identity="$(stat -c '%u:%a' -- "$temporary" 2>/dev/null)" \
        || { rm -f -- "$temporary"; return 1; }
    if [ -L "$temporary" ] || [ ! -f "$temporary" ] \
        || [ "$temporary_identity" != "$guard_owner_uid:600" ]; then
        rm -f -- "$temporary"
        return 1
    fi
    printf '%s\n' "$content" > "$temporary" \
        || { rm -f -- "$temporary"; return 1; }
    mv -fT -- "$temporary" "$target" \
        || { rm -f -- "$temporary"; return 1; }
}

private_state_failure() {
    contract_failure "unsafe-private-state:$1" 87
    return $?
}

prepare_private_state_directory() {
    if [ -L "$guard_runtime_dir" ]; then
        private_state_failure symlink
        return $?
    fi
    if [ ! -e "$guard_runtime_dir" ]; then
        (umask 077; mkdir -m 0700 -- "$guard_runtime_dir") \
            || { private_state_failure create; return $?; }
    fi
    if [ -L "$guard_runtime_dir" ] || [ ! -d "$guard_runtime_dir" ]; then
        private_state_failure wrong-type
        return $?
    fi
    private_state_identity="$(stat -c '%u:%a' -- "$guard_runtime_dir" 2>/dev/null)" \
        || { private_state_failure identity; return $?; }
    if [ "$private_state_identity" != "$guard_owner_uid:700" ]; then
        private_state_failure identity-or-mode
        return $?
    fi
}

read_guard_state() {
    state=""
    if [ -L "$guard_state" ]; then
        private_state_failure state-symlink
        return $?
    fi
    [ -e "$guard_state" ] || return 0
    if [ ! -f "$guard_state" ]; then
        private_state_failure state-wrong-type
        return $?
    fi
    state_identity="$(stat -c '%u:%a' -- "$guard_state" 2>/dev/null)" \
        || { private_state_failure state-identity; return $?; }
    if [ "$state_identity" != "$guard_owner_uid:600" ]; then
        private_state_failure state-identity-or-mode
        return $?
    fi
    state_read_failed=false
    state_extra=false
    state_extra_value=""
    {
        IFS= read -r state || state_read_failed=true
        IFS= read -r state_extra_value && state_extra=true
        [ -z "$state_extra_value" ] || state_extra=true
    } < "$guard_state"
    if [ "$state_read_failed" = true ] || [ "$state_extra" = true ]; then
        private_state_failure state-format
        return $?
    fi
    case "$state" in
        verified-update|complete) ;;
        *) private_state_failure state-value; return $? ;;
    esac
}

write_guard_state() {
    atomic_replace_text "$guard_state" "$1"
}

package_installed() {
    package_status="$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" || return 1
    [ "$package_status" = "install ok installed" ]
}

contract_failure() {
    detail=$1
    status=${2:-86}
    sentinel="NPA_SKYPILOT_BOOTSTRAP_FAILED status=${status} detail=${detail}"
    printf '%s\n' "$sentinel" >&2
    sentinel_write_failed=false
    atomic_replace_text "$guard_failure" "$sentinel" \
        || sentinel_write_failed=true
    atomic_replace_text "$sky_failure" "$sentinel" \
        || sentinel_write_failed=true
    if [ "$sentinel_write_failed" = true ]; then
        printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_SENTINEL_WRITE_FAILED' >&2
    fi
    return "$status"
}

verify_contract() {
    missing=""
    for package in rsync curl wget gcc patch pciutils fuse3 openssh-server coreutils; do
        package_installed "$package" || missing="$missing $package"
    done
    netcat_installed=false
    for package in netcat-openbsd netcat-traditional netcat; do
        if package_installed "$package"; then
            netcat_installed=true
            break
        fi
    done
    [ "$netcat_installed" = true ] || missing="$missing netcat"
    fuse_provides="$(dpkg-query -W -f='${Provides}' fuse3 2>/dev/null)" || fuse_provides=""
    case " $fuse_provides " in
        *" fuse "*|*" fuse ("*) ;;
        *) missing="$missing fuse" ;;
    esac
    for command_name in sh sudo sshd ssh-keygen rsync service curl wget nc gcc \
        patch lspci fusermount fusermount3 timeout; do
        command -v "$command_name" >/dev/null 2>&1 \
            || missing="$missing command:$command_name"
    done
    [ -x "$real_ssh_keygen" ] || missing="$missing command:real-ssh-keygen"
    if [ -n "$missing" ]; then
        contract_failure "missing:${missing# }" 86
        return $?
    fi
    printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_VERIFIED status=0 apt_required=false'
}

bootstrap_apt_get() {
    if [ -z "${SKYPILOT_POD_NODE_TYPE:-}" ] \
        || [ -e /tmp/apt_ssh_setup_complete ]; then
        exec "$real_apt_get" "$@"
    fi
    verify_contract >/dev/null || exit $?
    prepare_private_state_directory || exit $?
    read_guard_state || exit $?
    operation=${1:-}
    if [ "$operation" = update ] && [ -z "$state" ]; then
        write_guard_state verified-update \
            || { contract_failure private-state-write 87; exit $?; }
        printf '%s\n' 'NPA_SKYPILOT_BOOTSTRAP_APT_BYPASSED status=0 operation=update'
        exit 0
    fi
    if [ "$operation" = install ] && [ "$state" = verified-update ]; then
        shift
        requested=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -o|--option)
                    shift
                    [ "$#" -gt 0 ] \
                        || { contract_failure malformed-apt-options 87; exit $?; }
                    ;;
                -*) ;;
                *) requested="$requested $1" ;;
            esac
            shift
        done
        if [ "$requested" != " fuse" ]; then
            contract_failure "unexpected-install:${requested# }" 87
            exit $?
        fi
        write_guard_state complete \
            || { contract_failure private-state-write 87; exit $?; }
        printf '%s\n' \
            'NPA_SKYPILOT_BOOTSTRAP_APT_BYPASSED status=0 operation=install package=fuse provider=fuse3'
        exit 0
    fi
    exec "$real_apt_get" "$@"
}

bootstrap_timeout() {
    if [ -z "${SKYPILOT_POD_NODE_TYPE:-}" ] \
        || [ -e /tmp/apt_ssh_setup_complete ]; then
        exec "$real_timeout" "$@"
    fi
    "$real_timeout" --signal=TERM --kill-after=5s "$@"
    status=$?
    case "$status" in
        124|137) contract_failure deadline-exceeded "$status" >/dev/null ;;
    esac
    exit "$status"
}

bootstrap_ssh_keygen() {
    if [ "$#" -ne 1 ] || [ "$1" != -A ]; then
        contract_failure unexpected-ssh-keygen-arguments 87
        exit $?
    fi
    if [ "$(id -u)" -ne 0 ]; then
        contract_failure ssh-keygen-requires-root 87
        exit $?
    fi
    umask 077
    exec "$real_ssh_keygen" -A
}

case "$(basename "$0")" in
    apt-get) bootstrap_apt_get "$@" ;;
    timeout) bootstrap_timeout "$@" ;;
    ssh-keygen) bootstrap_ssh_keygen "$@" ;;
    npa-skypilot-bootstrap-guard)
        [ "${1:-}" = verify ] \
            || { printf '%s\n' 'usage: npa-skypilot-bootstrap-guard verify' >&2; exit 64; }
        verify_contract
        ;;
    *) printf '%s\n' 'NPA SkyPilot bootstrap guard invoked under an unknown name' >&2; exit 64 ;;
esac
