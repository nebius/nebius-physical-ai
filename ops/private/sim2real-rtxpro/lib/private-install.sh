#!/usr/bin/env bash
# Install operator secrets/config from ~/npa-sim2real-demo/private/ into ~/.npa/

operator_demo_root() {
  if [ -n "${NPA_SIM2REAL_DEMO:-}" ] && [ -d "${NPA_SIM2REAL_DEMO}" ]; then
    printf '%s\n' "${NPA_SIM2REAL_DEMO}"
    return 0
  fi
  if [ -n "${DEMO_ROOT:-}" ] && [ -d "${DEMO_ROOT}" ]; then
    printf '%s\n' "${DEMO_ROOT}"
    return 0
  fi
  local candidate="${HOME}/npa-sim2real-demo"
  if [ -d "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  return 1
}

operator_install_private_config() {
  local demo_root priv
  if ! demo_root="$(operator_demo_root)"; then
    return 0
  fi
  priv="${demo_root}/private"
  if [ ! -d "${priv}" ]; then
    return 0
  fi

  mkdir -p "${HOME}/.npa/clusters"
  local installed=0

  for name in config.yaml credentials.yaml; do
    if [ -f "${priv}/${name}" ]; then
      install -m 600 "${priv}/${name}" "${HOME}/.npa/${name}"
      installed=1
    fi
  done

  if [ -f "${priv}/operator.env" ]; then
    install -m 600 "${priv}/operator.env" "${HOME}/.npa/sim2real-operator.env"
    installed=1
  elif [ -f "${priv}/sim2real-operator.env" ]; then
    install -m 600 "${priv}/sim2real-operator.env" "${HOME}/.npa/sim2real-operator.env"
    installed=1
  fi

  if [ -d "${priv}/clusters" ]; then
    # shellcheck disable=SC2038
    find "${priv}/clusters" -type f -print0 2>/dev/null | while IFS= read -r -d '' f; do
      rel="${f#"${priv}/clusters/"}"
      mkdir -p "${HOME}/.npa/clusters/$(dirname "${rel}")"
      install -m 600 "${f}" "${HOME}/.npa/clusters/${rel}"
    done
    installed=1
  fi

  if [ "${installed}" = "1" ] && [ "${NPA_PRIVATE_INSTALL_QUIET:-0}" != "1" ]; then
    echo "Installed operator config from ${priv}/ -> ~/.npa/"
  fi
}
