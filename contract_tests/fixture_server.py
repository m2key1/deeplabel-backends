import hashlib
import hmac
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

SECRET_HEADER = 'X-Model-Backend-Secret'


@dataclass
class Served:
    data: bytes
    drop_once: bool = False


@dataclass
class FixtureServer:
    """Plays DeepLabel's side of sections 5.2 and 6.4: files by sha256 with Range, ETag and
    the secret check, and snapshot trees. It can drop a download halfway once, to prove the
    backend resumes."""

    host: str
    secret: str
    files: dict[str, Served] = field(default_factory=dict)
    trees: dict[str, dict[str, bytes]] = field(default_factory=dict)
    requests: list[tuple[str, str | None]] = field(default_factory=list)

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_GET(self):
                server.requests.append((self.path, self.headers.get('Range')))
                given = self.headers.get(SECRET_HEADER, '')
                if not hmac.compare_digest(given.encode(), server.secret.encode()):
                    return self._send(404, b'')
                parts = unquote(self.path).lstrip('/').split('/', 2)
                if len(parts) == 2 and parts[0] == 'files' and parts[1] in server.files:
                    return self._file(parts[1], server.files[parts[1]])
                if len(parts) == 3 and parts[0] == 'trees' and parts[2] in server.trees.get(parts[1], {}):
                    return self._send(200, server.trees[parts[1]][parts[2]])
                self._send(404, b'')

            def _send(self, status: int, body: bytes, headers: dict | None = None):
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _file(self, sha: str, served: Served):
                data, start = served.data, 0
                ranged = self.headers.get('Range', '')
                if ranged.startswith('bytes=') and ranged.endswith('-'):
                    start = int(ranged[len('bytes='):-1])
                body = data[start:]
                self.send_response(206 if start else 200)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('ETag', f'"{sha}"')
                self.send_header('Accept-Ranges', 'bytes')
                if start:
                    self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
                self.end_headers()
                if served.drop_once:
                    served.drop_once = False
                    self.wfile.write(body[:len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((self.host, 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()

    @property
    def base(self) -> str:
        return f'http://{self.host}:{self.httpd.server_address[1]}'

    def add_file(self, data: bytes, claimed_sha: str | None = None, drop_once: bool = False) -> dict:
        """A file reference as a load request carries it. A claimed_sha that is not the
        data's own serves a corrupted file under that name."""
        sha = claimed_sha or hashlib.sha256(data).hexdigest()
        self.files[sha] = Served(data, drop_once)
        return {'sha256': sha, 'size': len(data), 'url': f'{self.base}/files/{sha}'}

    def missing_file(self, sha: str) -> dict:
        return {'sha256': sha, 'size': 10, 'url': f'{self.base}/files/{sha}'}

    def add_tree(self, name: str, files: dict[str, bytes], manifest: dict) -> str:
        """A snapshot tree; returns its manifest URL."""
        listing = [{'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
                   for path, data in sorted(files.items())]
        self.trees[name] = {**files, 'manifest.json': json.dumps({**manifest, 'files': listing}).encode()}
        return f'{self.base}/trees/{name}/manifest.json'

    def ranged_requests(self, sha: str) -> list[str]:
        return [r for path, r in self.requests if path == f'/files/{sha}' and r and r != 'bytes=0-']
