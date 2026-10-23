# Model backend template

Copy this folder to wrap a model as a DeepLabel model backend. The contract is
`docs/MODEL-BACKENDS.md` in the DeepLabel repository; the SDK (`deeplabel_backend`)
implements all of it except your model.

1. Rename `MyModel` in `model.py`, set `model_type`, `tasks` and `tasks_detail()`.
2. Fill in `load()` (open the files DeepLabel sent, by role) and `predict()` (PIL images in,
   one result per image out, in the schema of section 4, coordinates as fractions of the
   image).
3. If the model trains here, keep `train_info()` and fill in `train_entrypoint.py`;
   otherwise return `None` from `train_info()` and delete the entrypoint.
4. Add the model's dependencies to `requirements.txt` and build the image.
5. Run the contract tests against it (from this repository's root):

   ```
   uv run pytest contract_tests --backend-url http://localhost:9100 --backend-secret s3cret \
       --weights-file /path/to/model.pt --serve-host 172.17.0.1 [--train] [--train-image my-image]
   ```

   `--serve-host` is the address at which the container reaches the test suite, which
   plays DeepLabel and serves the weights; on Docker's default bridge that is
   `172.17.0.1`.

6. Add `compose.snippet.yml` to the compose file the backend runs from, set its secret,
   and give DeepLabel the URL and the same secret.

Settings the SDK reads:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_BACKEND_SECRET` | required | shared with DeepLabel; the backend refuses to start without it |
| `PORT`, `HOST` | `9100`, `0.0.0.0` | where the server listens |
| `MODEL_CACHE_DIR` | `/cache` | the backend's own volume: weights by sha256, and local trainings |
| `MODEL_CACHE_MAX_GB` | `50` | disk budget; least recently used files that are not loaded go first |
| `MODEL_MAX_LOADED` | `2` | weights versions held in memory at once |
| `MODEL_IDLE_UNLOAD_SECONDS` | `0` | unload a version after it idled this long; 0 never does |
| `MODEL_DEVICE` | `auto` | passed to `load()` as `device` |
