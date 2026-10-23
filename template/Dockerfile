FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    "deeplabel-backend @ git+https://github.com/m2key1/deeplabel-backends.git@v0.1.0#subdirectory=sdk"
COPY model.py server.py train_entrypoint.py ./
ENV PORT=9100 MODEL_CACHE_DIR=/cache
EXPOSE 9100
CMD ["python3", "server.py"]
