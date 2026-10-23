# DeepLabel model backends

Models for [DeepLabel](https://github.com/m2key1/DeepLabel), each served as an HTTP service
over its model backend contract
([docs/MODEL-BACKENDS.md](https://github.com/m2key1/DeepLabel/blob/main/docs/MODEL-BACKENDS.md)),
plus the code to wrap and test more. A model repository knows nothing of DeepLabel, and
DeepLabel knows only the contract: a backend is a URL and a secret.

| Folder | What |
|---|---|
| `sdk/` | `deeplabel_backend`: `create_app(model)` serves every route of the contract around a `Model` with a few hooks |
| `template/` | a starting point to copy when wrapping a model; see its README |
| `contract_tests/` | the pytest suite any backend must pass, run against a URL |
| `stub/` | a deterministic backend serving every task, for DeepLabel's end-to-end tests |
| `deepocr/` | [DeepOCR](https://github.com/m2key1/DeepOCR) line recognition (`ocr`), with [DeepOCR v3](https://huggingface.co/M2key1/deepocr-v3) built in |
| `kraken_lines/` | Kraken's baseline segmenter (`lines`); without weights it serves Kraken's default model |

## SDK, stub and contract tests

```bash
uv sync
uv run pytest                                                       # SDK unit tests
uv run pytest contract_tests --backend-cmd deeplabel-stub --train   # the contract, against the stub
```

`--backend-cmd` starts a backend on a free port; `--backend-url` tests one that is already
running. `uv run pytest contract_tests --help` lists every option.

## Building and running a backend

A wrapper whose model comes from a repository pins the revision it was tested against in
`<wrapper>/MODEL_REVISION`; the image is built from that revision of a local clone:

```bash
./build.sh deepocr ../DeepOCR        # prints deepocr-backend:<tag>
./build.sh kraken_lines              # kraken-lines-backend:<tag>
cp .env.example .env                 # the tag, a secret, a port
docker compose up -d deepocr
```

Weights come from DeepLabel when a model is first used and are cached in `/cache` by
sha256; a backend that ships its own (DeepOCR) also serves those without any. Test a
built image, training included, with:

```bash
uv run pytest contract_tests --backend-url http://localhost:9201 --backend-secret s3cret \
    --weights-file /path/to/model.pt --serve-host <address the container reaches this suite at> \
    --train --train-image deepocr-backend:<tag>
```

DeepLabel needs the backend's URL and secret (`MODEL_BACKEND_<NAME>_URL`,
`MODEL_BACKEND_<NAME>_SECRET`). For training on a Slurm cluster,
`./cluster-image.sh <image:tag>` writes the `.sqsh` file to copy to the cluster and name in
`MODEL_BACKEND_<NAME>_CLUSTER_IMAGE`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND_SECRET` | required | shared with DeepLabel; the backend refuses to start without it |
| `PORT`, `HOST` | `9100`, `0.0.0.0` | where the server listens |
| `MODEL_CACHE_DIR` | `/cache` | weights by sha256, and local trainings |
| `MODEL_CACHE_MAX_GB` | `50` | disk budget; least recently used files that are not loaded go first |
| `MODEL_MAX_LOADED` | `2` | weights versions held in memory at once |
| `MODEL_IDLE_UNLOAD_SECONDS` | `0` | unload a version after it idled this long; 0 never does |
| `MODEL_DEVICE` | `auto` | passed to `load()` as `device` |

## License

Apache License 2.0 - see [LICENSE](LICENSE).
