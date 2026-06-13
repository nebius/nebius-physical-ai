#!/usr/bin/env bash
# Standard Mac/Linux PATH for operator scripts (safe to source from any script).

operator_fix_path() {
  export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
}

operator_fix_path
