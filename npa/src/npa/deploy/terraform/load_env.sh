# Trusted loader program; environment files themselves are data, never scripts.
npa_load_env_file() {
    if [ ! -r "$1" ]; then
        printf '%s\n' 'Environment file is not readable' >&2
        return 1
    fi
    while IFS= read -r npa_env_line || [ -n "$npa_env_line" ]; do
        case "$npa_env_line" in
            ''|'#'*) continue ;;
            *=*) ;;
            *) printf '%s\n' 'Invalid environment file entry' >&2; return 1 ;;
        esac
        npa_env_name=${npa_env_line%%=*}
        case "$npa_env_name" in
            ''|[0-9]*|*[!a-zA-Z0-9_]*)
                printf '%s\n' 'Invalid environment variable name' >&2
                return 1
                ;;
        esac
        # Quoted export is a shell builtin: the value is not parsed again and
        # does not enter a child process's command arguments.
        export "$npa_env_line" || return 1
    done < "$1"
}
