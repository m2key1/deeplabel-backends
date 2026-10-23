FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
        libgl1 libglib2.0-0 libvips42 \
    && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir --break-system-packages uv
WORKDIR /srv
# Kraken from PyPI, at the versions this wrapper's lock file pins.
COPY kraken_lines/pyproject.toml kraken_lines/uv.lock /srv/kraken_lines/
RUN cd /srv/kraken_lines && UV_PROJECT_ENVIRONMENT=/srv/.venv uv sync --frozen --no-dev --python /usr/bin/python3
ENV PATH=/srv/.venv/bin:$PATH
# The SDK, from this repository.
COPY sdk /opt/deeplabel-sdk
RUN uv pip install --python /srv/.venv/bin/python /opt/deeplabel-sdk
COPY kraken_lines /srv/kraken_lines
ENV PYTHONPATH=/srv PORT=9100 MODEL_CACHE_DIR=/cache
EXPOSE 9100
CMD ["python", "-m", "kraken_lines.server"]
