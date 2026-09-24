FROM antioch-engine/isaac-sim-6.0.1:0.4.289

# This is a private Antioch project image, never an NPA public distribution.
RUN uv pip install --system --no-cache "imageio-ffmpeg==0.6.0"
ENV ANTIOCH_PROJECT_DIR=/workspace/project
WORKDIR /workspace/project
COPY antioch.yaml /workspace/project/antioch.yaml
COPY src/ /workspace/project/src/
