#!/bin/sh
set -u

real_apt_get=/usr/bin/apt-get
real_timeout=/usr/bin/timeout
guard_state=/tmp/npa-skypilot-bootstrap-apt.state
guard_failure=/tmp/npa-skypilot-bootstrap-contract.failed
sky_failure=/tmp/apt-ssh-setup.failed

package_installed() {
    package_status="$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" || return 1
    [ "$package_status" = "install ok installed" ]
}

contract_failure() {
    detail=$1
    status=${2:-86}
    sentinel="NPA_SKYPILOT_BOOTSTRAP_FAILED status=${status} detail=${detail}"
    printf '%s\n' "$sentinel" >&2
    printf '%s\n' "$sentinel" > "$guard_failure"
    printf '%s\n' "$sentinel" > "$sky_failure"
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
    for command_name in sh sudo sshd rsync service curl wget nc gcc patch lspci \
        fusermount fusermount3 timeout; do
        command -v "$command_name" >/dev/null 2>&1 \
            || missing="$missing command:$command_name"
    done
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
    state=""
    [ ! -f "$guard_state" ] || state="$(cat "$guard_state")"
    operation=${1:-}
    if [ "$operation" = update ] && [ -z "$state" ]; then
        printf '%s\n' verified-update > "$guard_state"
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
        printf '%s\n' complete > "$guard_state"
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

case "$(basename "$0")" in
    apt-get) bootstrap_apt_get "$@" ;;
    timeout) bootstrap_timeout "$@" ;;
    npa-skypilot-bootstrap-guard)
        [ "${1:-}" = verify ] \
            || { printf '%s\n' 'usage: npa-skypilot-bootstrap-guard verify' >&2; exit 64; }
        verify_contract
        ;;
    *) printf '%s\n' 'NPA SkyPilot bootstrap guard invoked under an unknown name' >&2; exit 64 ;;
esac
