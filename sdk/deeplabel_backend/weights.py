import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import errors, transfer
from .errors import BackendError
from .model import Loaded, Model
from .schemas import WEIGHT_ROLES, LoadRequest

BUILTIN = 'builtin'
IDLE_SWEEP_SECONDS = 10


@dataclass
class Version:
    sha256: str
    request: LoadRequest | None = None
    state: str = 'absent'
    bytes: int = 0
    total: int = 0
    error: BackendError | None = None
    loaded: Loaded | None = None
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    unload_pending: bool = False
    settled: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict:
        if self.state == 'downloading':
            return {'state': 'downloading', 'bytes': self.bytes, 'total': self.total}
        if self.state == 'loaded':
            return {'state': 'loaded', 'facts': self.loaded.facts}
        if self.state == 'error':
            return {'state': 'error', **self.error.body()}
        return {'state': self.state}


class WeightCache:
    """Section 5.3: a disk cache named by sha256 with resumable, verified downloads and a
    size budget, and at most max_loaded versions in memory, least recently used out first."""

    def __init__(self, model: Model, directory: Path, secret: str, max_bytes: int, max_loaded: int,
                 idle_unload_seconds: float = 0, device: str = 'auto'):
        self.model = model
        self.directory = directory
        self.secret = secret
        self.max_bytes = max_bytes
        self.max_loaded = max_loaded
        self.idle_unload_seconds = idle_unload_seconds
        self.device = device
        self.versions: dict[str, Version] = {}
        self.lock = threading.RLock()
        self.file_locks: dict[str, threading.Lock] = {}
        self.builtin_lock = threading.Lock()
        directory.mkdir(parents=True, exist_ok=True)
        if idle_unload_seconds > 0:
            threading.Thread(target=self._idle_sweep, daemon=True).start()

    def file_path(self, sha256: str) -> Path:
        return self.directory / sha256

    def loaded_versions(self) -> list[str]:
        with self.lock:
            return [sha for sha, v in self.versions.items() if v.state == 'loaded' and sha != BUILTIN]

    def state(self, sha256: str) -> dict:
        if sha256 == BUILTIN and self.model.builtin:
            return self._builtin().public()
        with self.lock:
            version = self.versions.get(sha256)
            if version is not None and version.state != 'absent':
                return version.public()
        return {'state': 'cached' if self.file_path(sha256).is_file() else 'absent'}

    def _check(self, request: LoadRequest) -> None:
        declared = self.model.model_type
        if request.model_type and declared and request.model_type != declared:
            raise errors.weights_incompatible(
                f'these weights are {request.model_type!r}; this backend loads {declared!r}')
        unknown = set(request.files) - set(WEIGHT_ROLES)
        if unknown:
            raise errors.bad_request(f'unknown file roles: {", ".join(sorted(unknown))}')
        missing = [role for role, need in self.model.weight_roles.items()
                   if need == 'required' and role not in request.files]
        if missing:
            raise errors.bad_request(f'missing file roles: {", ".join(missing)}')
        if 'weights' in request.files and request.files['weights'].sha256 != request.sha256:
            raise errors.bad_request('sha256 must be the sha256 of the weights file')

    def load(self, request: LoadRequest) -> Version:
        self._check(request)
        with self.lock:
            version = self.versions.setdefault(request.sha256, Version(request.sha256))
            version.last_used = time.monotonic()
            if version.state in ('downloading', 'loading', 'loaded'):
                return version
            version.request = request
            version.state = 'downloading'
            version.error = None
            version.total = sum(ref.size or 0 for ref in request.files.values())
            version.bytes = sum(self.file_path(ref.sha256).stat().st_size
                                for ref in request.files.values() if self.file_path(ref.sha256).is_file())
            version.settled.clear()
        threading.Thread(target=self._prepare, args=(version,), daemon=True).start()
        return version

    def _prepare(self, version: Version) -> None:
        try:
            paths = {role: self.fetch(ref, lambda n: self._count(version, n))
                     for role, ref in version.request.files.items()}
            with self.lock:
                version.state = 'loading'
            self._make_room(keep=version.sha256)
            loaded = self.model.load(paths, self.device)
            with self.lock:
                version.loaded = loaded
                version.state = 'loaded'
                version.last_used = time.monotonic()
        except transfer.TransferError as error:
            self._fail(version, errors.checksum_mismatch(str(error)) if error.kind == 'mismatch'
                       else errors.weights_unavailable(str(error)))
        except BackendError as error:
            self._fail(version, error)
        except Exception as error:
            self._fail(version, errors.out_of_memory(str(error)) if errors.is_out_of_memory(error)
                       else errors.internal(f'loading the weights failed: {error}'))
        finally:
            self._trim_disk()
            version.settled.set()

    def _fail(self, version: Version, error: BackendError) -> None:
        with self.lock:
            version.state = 'error'
            version.error = error

    def _count(self, version: Version, n: int) -> None:
        with self.lock:
            version.bytes += n

    def fetch(self, ref, on_bytes=lambda n: None) -> Path:
        """One file into the cache, downloaded once however many loads ask for it."""
        dest = self.file_path(ref.sha256)
        with self.lock:
            file_lock = self.file_locks.setdefault(ref.sha256, threading.Lock())
        with file_lock:
            if dest.is_file() and (ref.size is None or dest.stat().st_size == ref.size):
                os.utime(dest)
                return dest
            with transfer.client(self.secret) as http:
                return transfer.download(http, ref.url, dest, ref.sha256, ref.size, on_bytes)

    @contextmanager
    def use(self, sha256: str | None):
        """Holds a loaded version for one request, so it cannot be unloaded under it."""
        version = self._builtin() if sha256 is None else None
        with self.lock:
            if version is None:
                version = self.versions.get(sha256)
                if version is None or version.state != 'loaded':
                    raise errors.weights_not_loaded(sha256)
            version.refs += 1
            version.last_used = time.monotonic()
        try:
            yield version.loaded
        finally:
            with self.lock:
                version.refs -= 1
                if version.unload_pending and version.refs == 0:
                    self._unload(version)

    def _builtin(self) -> Version:
        if not self.model.builtin:
            raise errors.bad_request('this backend has no builtin weights; name the weights to use')
        with self.builtin_lock:
            with self.lock:
                version = self.versions.get(BUILTIN)
                if version is not None and version.state == 'loaded':
                    return version
            loaded = self.model.load({}, self.device)
            with self.lock:
                version = self.versions.setdefault(BUILTIN, Version(BUILTIN))
                version.loaded = loaded
                version.state = 'loaded'
                version.settled.set()
                return version

    def unload(self, sha256: str | None = None) -> list[str]:
        with self.lock:
            chosen = [v for sha, v in self.versions.items()
                      if v.state == 'loaded' and (sha256 is None or sha == sha256)]
            for version in chosen:
                if version.refs:
                    version.unload_pending = True
                else:
                    self._unload(version)
            return [v.sha256 for v in chosen]

    def _unload(self, version: Version) -> None:
        loaded, version.loaded = version.loaded, None
        version.unload_pending = False
        version.state = 'cached' if self.file_path(version.sha256).is_file() else 'absent'
        self.model.unload(loaded)

    def _make_room(self, keep: str) -> None:
        with self.lock:
            loaded = sorted((v for sha, v in self.versions.items()
                             if v.state == 'loaded' and sha not in (keep, BUILTIN)),
                            key=lambda v: v.last_used)
            excess = len(loaded) + 1 - self.max_loaded
            for version in [v for v in loaded if v.refs == 0][:max(0, excess)]:
                self._unload(version)

    def _trim_disk(self) -> None:
        with self.lock:
            in_use = {ref.sha256 for v in self.versions.values()
                      if v.state in ('downloading', 'loading', 'loaded') and v.request
                      for ref in v.request.files.values()}
            files = [p for p in self.directory.iterdir() if p.is_file() and not p.name.endswith('.part')]
            total = sum(p.stat().st_size for p in files)
            for path in sorted(files, key=lambda p: p.stat().st_mtime):
                if total <= self.max_bytes:
                    break
                if path.name in in_use:
                    continue
                total -= path.stat().st_size
                path.unlink()

    def _idle_sweep(self) -> None:
        while True:
            time.sleep(IDLE_SWEEP_SECONDS)
            cutoff = time.monotonic() - self.idle_unload_seconds
            with self.lock:
                for version in list(self.versions.values()):
                    if version.state == 'loaded' and version.refs == 0 and version.last_used < cutoff:
                        self._unload(version)
