FROM antioch-engine/isaac-sim-6.0.1:0.4.289

# This is a private Antioch project image, never an NPA public distribution.
RUN uv pip install --system --no-cache "imageio-ffmpeg==0.6.0"

# Project synchronization and Isaac's runtime caches must remain writable after
# dropping privileges, including the simulator's installation-relative caches.
RUN install -d -o 1000 -g 1000 \
    /workspace/project \
    /tmp/npa-home \
    /tmp/npa-home/.cache \
    /tmp/npa-home/.cache/ov \
    /usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache/DerivedDataCache \
    "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/apps/Isaac-Sim Python/scripts" \
    /usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/shared
ENV ANTIOCH_PROJECT_DIR=/workspace/project \
    HOME=/tmp/npa-home \
    XDG_CACHE_HOME=/tmp/npa-home/.cache
WORKDIR /workspace/project
COPY --chown=1000:1000 antioch.yaml /workspace/project/antioch.yaml
COPY --chown=1000:1000 src/ /workspace/project/src/
USER 1000:1000
