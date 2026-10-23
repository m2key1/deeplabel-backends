FROM python:3.12-slim
WORKDIR /srv
COPY sdk /srv/sdk
COPY stub /srv/stub
RUN pip install --no-cache-dir /srv/sdk /srv/stub
ENV PORT=9100 MODEL_CACHE_DIR=/cache
EXPOSE 9100
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9100/health')"
CMD ["deeplabel-stub"]
