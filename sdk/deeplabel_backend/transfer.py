import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx

SECRET_HEADER = 'X-Model-Backend-Secret'
CHUNK = 1 << 20
ATTEMPTS = 5
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 60.0


class TransferError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def client(secret: str, connections: int = 8) -> httpx.Client:
    return httpx.Client(headers={SECRET_HEADER: secret},
                        timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                        limits=httpx.Limits(max_connections=connections))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def _fetch_part(http: httpx.Client, url: str, part: Path, size: int | None,
                on_bytes: Callable[[int], None]) -> None:
    """Downloads url into part, resuming with Range after a dropped connection."""
    failures = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have >= size:
            if have > size:
                part.unlink()
                continue
            return
        headers = {'Range': f'bytes={have}-'} if have else {}
        try:
            with http.stream('GET', url, headers=headers) as response:
                if response.status_code not in (200, 206):
                    raise TransferError('unavailable', f'{url}: HTTP {response.status_code}')
                mode = 'ab' if response.status_code == 206 else 'wb'
                if mode == 'wb' and have:
                    on_bytes(-have)
                with open(part, mode) as out:
                    for chunk in response.iter_bytes(CHUNK):
                        out.write(chunk)
                        on_bytes(len(chunk))
            if size is None or part.stat().st_size >= size:
                return
        except httpx.TransportError as error:
            failures += 1
            if failures >= ATTEMPTS:
                raise TransferError('unavailable', f'{url}: {error}') from error


def download(http: httpx.Client, url: str, dest: Path, sha256: str, size: int | None = None,
             on_bytes: Callable[[int], None] = lambda n: None) -> Path:
    """Fetches url to dest, verified against sha256. A mismatch starts over from zero once."""
    part = dest.with_name(dest.name + '.part')
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        if attempt:
            on_bytes(-part.stat().st_size if part.exists() else 0)
            part.unlink(missing_ok=True)
        _fetch_part(http, url, part, size, on_bytes)
        if file_sha256(part) == sha256:
            part.replace(dest)
            return dest
    part.unlink(missing_ok=True)
    raise TransferError('mismatch', f'{url} does not hash to {sha256}')


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise TransferError('unavailable', f'snapshot path {relative!r} leaves the snapshot')
    return target


def fetch_tree(manifest_url: str, dest: Path, secret: str, connections: int = 8,
               on_progress: Callable[[int, int], None] = lambda done, total: None) -> dict:
    """Downloads a snapshot (section 6.4): the manifest, then every file it lists, in
    parallel. Files already present with the right size and hash are kept."""
    dest.mkdir(parents=True, exist_ok=True)
    base = manifest_url.rsplit('/', 1)[0] + '/'
    with client(secret, connections) as http:
        try:
            response = http.get(manifest_url)
        except httpx.TransportError as error:
            raise TransferError('unavailable', f'{manifest_url}: {error}') from error
        if response.status_code != 200:
            raise TransferError('unavailable', f'{manifest_url}: HTTP {response.status_code}')
        manifest = response.json()
        (dest / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        files = manifest.get('files', [])
        done = 0
        lock = threading.Lock()

        def fetch(entry: dict) -> None:
            nonlocal done
            target = _inside(dest, entry['path'])
            present = (target.is_file() and target.stat().st_size == entry['size']
                       and file_sha256(target) == entry['sha256'])
            if not present:
                download(http, base + quote(entry['path']), target, entry['sha256'], entry['size'])
            with lock:
                done += 1
                on_progress(done, len(files))

        with ThreadPoolExecutor(connections) as pool:
            list(pool.map(fetch, files))
    return manifest
