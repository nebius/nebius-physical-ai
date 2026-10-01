#!/usr/bin/env bash
# Provide working media tools without the hosted runner's slow Azure apt mirror.
set -euo pipefail

if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null; then
  ffmpeg -version
  ffprobe -version
  exit 0
fi

ubuntu_sources="${1:-/etc/apt/sources.list.d/ubuntu.sources}"
test -s "$ubuntu_sources"
media_sources="$(mktemp "${RUNNER_TEMP:?}/npa-media-XXXXXXXX.sources")"
trap 'rm -f "$media_sources"' EXIT

# Preserve suites, components, architectures and Signed-By. Do not edit the
# runner's system sources or load unrelated vendor repositories. Hosted images
# can select Azure indirectly through a mirror list instead of a literal URI.
sed -E \
  -e 's#https?://azure\.archive\.ubuntu\.com/ubuntu/?([[:space:]]|$)#https://archive.ubuntu.com/ubuntu\1#g' \
  -e 's#mirror\+file:/+etc/apt/apt-mirrors\.txt([[:space:]]|$)#https://archive.ubuntu.com/ubuntu\1#g' \
  "$ubuntu_sources" > "$media_sources"
chmod 644 "$media_sources"
apt_options=(
  -o "Dir::Etc::sourcelist=$media_sources"
  -o Dir::Etc::sourceparts=-
)
sudo apt-get "${apt_options[@]}" update
sudo apt-get "${apt_options[@]}" install -y --no-install-recommends ffmpeg
ffmpeg -version
ffprobe -version
