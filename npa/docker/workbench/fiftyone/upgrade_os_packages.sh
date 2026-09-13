#!/bin/sh
# Apply the OS security updates shared by the FiftyOne build and its base scan.
set -eu

apt-get update
apt-get upgrade -y --no-install-recommends

# Debian's fixed Perl package covers CVE-2026-8376, CVE-2026-42496 and
# CVE-2026-13221. Fail the build if a stale mirror cannot supply the fix.
installed_version=$(dpkg-query -W -f='${Version}' perl-base)
if ! dpkg --compare-versions "$installed_version" ge '5.40.1-6+deb13u1'; then
    printf 'perl-base security update missing: installed %s\n' "$installed_version" >&2
    exit 1
fi
