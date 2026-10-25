ARG BASE_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-envgen@sha256:55f1541c1a86e9d865963753cfa38fa1dbf88928c8d28d9ee30e2851a5604b88
FROM ${BASE_IMAGE}

ARG NPA_SOURCE_SHA

LABEL org.opencontainers.image.title="npa-reference-policy"
LABEL org.opencontainers.image.description="Reference BYO-compatible Sim2Real action policy image"
LABEL npa.base.image="npa-envgen"
LABEL org.opencontainers.image.revision="${NPA_SOURCE_SHA}"

ENV POLICY_IMAGE_KIND=npa-reference-policy
ENV NPA_SIM2REAL_POLICY_VARIANT=reference
ENV NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}

USER root

# The inherited exact-source .pth resolves this path. Replace the complete SDK,
# so the advertised child revision identifies the source that actually runs.
COPY --chown=ubuntu:ubuntu pyproject.toml /opt/npa/pyproject.toml
RUN rm -rf /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa
COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows
COPY --chown=ubuntu:ubuntu docker/workbench/common/verify_sim2real_sdk_source.py \
    /opt/npa/bin/verify-sim2real-sdk-source.py

RUN test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40 \
    && find /opt/npa/src /opt/npa/workflows -type d -exec chmod a+rx {} + \
    && find /opt/npa/src /opt/npa/workflows -type f -exec chmod a+r {} + \
    && chmod a+r /opt/npa/pyproject.toml \
    && python -m pip check \
    && env -u PYTHONPATH python -c \
      'from npa.workflows.sim2real_envgen import main; assert callable(main)' \
    && env -u PYTHONPATH python /opt/npa/bin/verify-sim2real-sdk-source.py \
      --source-root /opt/npa/src

USER ubuntu

CMD ["python", "-m", "npa.workflows.sim2real_envgen", "policy-contract", "--help"]
