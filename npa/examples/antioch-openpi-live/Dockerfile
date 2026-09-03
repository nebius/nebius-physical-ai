FROM antioch-engine/isaac-sim-6.0.1:0.3.63

LABEL npa.antioch.live-transport="declared-port-double-wss-v1"

# Protocol-only dependencies. OpenPI source, model code, and weights are not
# copied into this image; the policy implementation remains on the B200 side.
RUN uv pip install --system --no-cache \
    "msgpack==1.1.1" \
    "websockets==15.0.1"

# Antioch syncs and bakes queued project files at this canonical path. Isaac's
# runtime cache/document parents must also remain writable after dropping uid.
RUN install -d -o 1000 -g 1000 \
    /workspace/project \
    /tmp/npa-home \
    /tmp/npa-home/.cache \
    /tmp/npa-home/.cache/ov \
    /usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache/DerivedDataCache \
    "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/apps/Isaac-Sim Python/scripts" \
    /usr/local/lib/python3.12/dist-packages/isaacsim/kit/data/documents/Kit/shared
COPY --chown=1000:1000 src/ /workspace/project/src/

ENV HOME=/tmp/npa-home \
    XDG_CACHE_HOME=/tmp/npa-home/.cache

USER 1000:1000

# The declared-port owner is the detached Antioch service container. Runtime
# credentials arrive later through supported `antioch services cp`; the bridge
# waits without logging or accepting traffic until the complete bundle appears.
ENTRYPOINT ["/usr/local/bin/python", "/workspace/project/src/relay_bridge.py", "--bundle", "/tmp/npa-live-client-current", "--wait-for-bundle"]
