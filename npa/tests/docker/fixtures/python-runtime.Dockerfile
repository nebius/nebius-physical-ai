# syntax=docker/dockerfile:1.7
FROM python:3.12.14-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS bootstrap-ca

FROM ubuntu:22.04@sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986

ENV DEBIAN_FRONTEND=noninteractive
COPY --chmod=0755 docker/workbench/common/configure_ubuntu_snapshot.sh /usr/local/bin/configure-ubuntu-snapshot
COPY --chmod=0755 docker/workbench/common/install_python312_jammy.sh /usr/local/bin/install-python312-jammy
COPY --chmod=0755 docker/workbench/isaac-lab/apt-get /opt/npa/apt-retry-bin/apt-get
RUN --mount=type=bind,from=bootstrap-ca,source=/etc/ssl/certs/ca-certificates.crt,target=/run/python-probe-ca.crt,ro \
    printf '%s\n' \
      'Acquire::Retries "10";' \
      'Acquire::Queue-Mode "access";' \
      'Acquire::http::Pipeline-Depth "0";' \
      'Acquire::https::Pipeline-Depth "0";' \
      'APT::Update::Error-Mode "any";' \
      > /etc/apt/apt.conf.d/80-npa-ubuntu-snapshot-retries \
    && export PATH="/opt/npa/apt-retry-bin:${PATH}" \
    && configure-ubuntu-snapshot 20261018T000000Z \
    && apt-get -o Acquire::https::CaInfo=/run/python-probe-ca.crt update \
    && apt-get -o Acquire::https::CaInfo=/run/python-probe-ca.crt install -y --no-install-recommends ca-certificates curl \
    && install-python312-jammy \
    && python3.12 -m venv /opt/python-probe \
    && /opt/python-probe/bin/python -m pip install --no-cache-dir --no-deps \
      'pillow @ https://files.pythonhosted.org/packages/e6/99/fdf65cc539e266ab93546c261bdf98411fe165c6e4e13f9dbf597097192b/pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=42d63be3af4f9b951b199b0a9d23365d37ab14687e6d77b74940432c8515c4ca' \
    && useradd --create-home --uid 1000 probe \
    && rm -rf /var/lib/apt/lists/*
USER probe
CMD ["/opt/python-probe/bin/python", "-c", "import io, sys, PIL; from PIL import Image; assert sys.version_info[:2] == (3, 12); assert PIL.__version__ == '13.0.0'; output=io.BytesIO(); Image.new('RGB', (2, 2), (1, 2, 3)).save(output, format='PNG'); output.seek(0); assert Image.open(output).getpixel((0, 0)) == (1, 2, 3); print('python312-pillow13-roundtrip-ok')"]
