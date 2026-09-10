# Native libraries must precede SkyPilot's real bash/APT startup, including when
# Kubernetes overrides ENTRYPOINT. A failure exits the shell before its command.
if ! /usr/local/bin/python3.12 -I -S -B /opt/ncore/native/native_bootstrap.py ensure; then
    printf '%s\n' 'NCore native setup failed; refusing shell startup' >&2
    exit 1
fi
