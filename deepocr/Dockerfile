FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
# KenLM and flashlight-text compile at install time.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git \
        build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev \
        libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev libeigen3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
RUN python3 -m venv /srv/.venv
ENV PATH=/srv/.venv/bin:$PATH
# KenLM's lmplz and build_binary, which fit a run's own language model; Boost.System is
# header-only on this Boost, and the test framework only builds KenLM's own tests.
ARG KENLM_REVISION=4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
RUN git clone https://github.com/kpu/kenlm /opt/kenlm && git -C /opt/kenlm checkout $KENLM_REVISION \
    && sed -i '/^  system$/d; /^  unit_test_framework$/d' /opt/kenlm/CMakeLists.txt \
    && cmake -S /opt/kenlm -B /opt/kenlm/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/kenlm/build -j8 --target lmplz build_binary \
    && cp /opt/kenlm/build/bin/lmplz /opt/kenlm/build/bin/build_binary /usr/local/bin/ && rm -rf /opt/kenlm
COPY --from=model requirements.txt /srv/model/requirements.txt
RUN pip install --no-cache-dir -r /srv/model/requirements.txt
# The builtin weights, DeepOCR v3, at a pinned Hugging Face revision; each file is checked.
ARG BUILTIN=https://huggingface.co/M2key1/deepocr-v3/resolve/dabedb63827c1bf2fbd9c6ce8f9a6d3ab5fe79ed
ADD --checksum=sha256:58eccc0567053e406805e6864ce5055355c3efa43e7f962ff2bb8ff9127deae7 $BUILTIN/deepocr-v3.pt /srv/builtin/
ADD --checksum=sha256:fbe2c7c470c0df29ab24e784ca82e80858b8d3a7e87f8e4852ade39fa5a7d92c $BUILTIN/union_bytes_v3.bin /srv/builtin/
ADD --checksum=sha256:7c903df4784b5de3130bdfe6902d797b1f04f6c849d01ebc5f1e8cb26cfb7024 $BUILTIN/union_bytes_v3.dict /srv/builtin/
# The SDK, from this repository.
COPY sdk /opt/deeplabel-sdk
RUN pip install --no-cache-dir /opt/deeplabel-sdk
# The model code at the pinned revision (MODEL_REVISION), handed in by build.sh as the
# build context "model"; then this wrapper.
COPY --from=model . /srv/model
COPY deepocr /srv/deepocr
ENV PYTHONPATH=/srv PORT=9100 MODEL_CACHE_DIR=/cache
# Multi-megabyte arrays freed by another thread than the one that allocated them stay
# with glibc unless it hands them back early; without this the server grows without bound.
ENV MALLOC_MMAP_THRESHOLD_=262144 MALLOC_TRIM_THRESHOLD_=262144
EXPOSE 9100
CMD ["python", "-m", "deepocr.server"]
