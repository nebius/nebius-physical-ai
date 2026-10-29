#!/usr/bin/env bash
# Fail-closed gate for the separately maintained SM120 Genesis variant.
#
# Do not turn this into a resolver override: the current recipe requires
# Pillow 13 and fsspec 2026.6.0, while every supported LeRobot release that
# supplies the native dataset/ACT path has incompatible declared ceilings.
# A reviewed replacement must include real source-attested qualification.

require_current_secure_genesis_closure() {
  cat >&2 <<'EOF'
ERROR: the additive SM120 Genesis image build is unavailable before any Docker build starts.
The additive SM120 Genesis path requires Pillow>=13.0.0,<14.0 and
fsspec==2026.6.0, but its native LeRobot dataset/ACT dependency has no
supported secure closure: LeRobot 0.4.4 resolves datasets<5 (which caps
fsspec below the required floor), while newer released LeRobot lines retain
Pillow<13 and datasets<5 ceilings.
See npa/docker/workbench/genesis/README.md#current-source-build-boundary.
Do not lower a security floor, force an unsupported resolver, or relabel an
earlier image. A reviewed source-attested replacement closure is required.
EOF
  return 3
}
