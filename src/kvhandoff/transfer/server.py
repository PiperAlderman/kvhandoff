"""Transfer service: serves exports from a store, read-only except for release markers. Stdlib only.

  GET  /healthz                                no auth; store state, no tags
  GET  /v1/exports/<tag>                       the interchange header (JSON); a failure record answers 200 with
                                               "error" set, so the client fails at once; 404 until published
  GET  /v1/exports/<tag>/stream                streaming mode: {"published": bool, "header": ...}, the layers completed
                                               so far (or the final header); a failed stream answers 200 with
                                               "error" set in the header; 404 until begun
  GET  /v1/exports/<tag>/layers/<i>            the layer's blobs concatenated in header order (header X-KVH-Seq);
                                               with X-KVH-Live: 1 and X-KVH-Seq, also a layer of an export that is
                                               still streaming, once that layer is complete
  POST /v1/exports/<tag>/release?seq=<n>       mark consumed so the slot can be reused now; 204 or 404

Every /v1 request needs ``Authorization: Bearer <token>`` (token read from a file at start). At most
``max_concurrent_transfers`` /v1 requests are served at once (others get 503 with Retry-After: 1) and at most
``max_connections`` connections are open (others are closed at once). Each response holds the export open with a
shared lock, so the writer cannot reuse the slot mid-response; each layer response carries the export's seq so the
client can tell if the slot was reused between two requests.

  kvhandoff-transfer --config /etc/kvhandoff/transfer.toml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlsplit

from kvhandoff import __version__
from kvhandoff.auth import bearer_ok, load_token
from kvhandoff.config import TransferConfig, load_toml
from kvhandoff.logs import configure, get_logger
from kvhandoff.store import ExportFailed, ExportNotFound, ExportReader, OpenExport

log = get_logger("transfer")

# The only store fields /healthz may show (no auth): counts and file-system bytes, never a tag or a path.
HEALTHZ_STORE_FIELDS = frozenset(
    {"root_ok", "slots", "published", "held", "consumed", "failed_records", "fs_bytes", "fs_free_bytes"}
)

_HEADER = re.compile(r"^/v1/exports/([A-Za-z0-9][A-Za-z0-9._-]{0,95})$")
_STREAM = re.compile(r"^/v1/exports/([A-Za-z0-9][A-Za-z0-9._-]{0,95})/stream$")
_LAYER = re.compile(r"^/v1/exports/([A-Za-z0-9][A-Za-z0-9._-]{0,95})/layers/(\d{1,4})$")
_RELEASE = re.compile(r"^/v1/exports/([A-Za-z0-9][A-Za-z0-9._-]{0,95})/release$")


class TransferServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, cfg: TransferConfig, token: str | None):
        self.cfg = cfg
        self.token = token
        self.reader = ExportReader(cfg.store_path)
        self.transfers = threading.BoundedSemaphore(cfg.max_concurrent_transfers)
        self._conn_lock = threading.Lock()
        self._conns = 0
        self.inflight = 0
        self.started = time.time()
        super().__init__((cfg.bind, cfg.port), _Handler)

    # connection cap: refuse beyond max_connections before a thread is spawned
    def process_request(self, request: Any, client_address: Any) -> None:
        with self._conn_lock:
            if self._conns >= self.cfg.max_connections:
                over = True
            else:
                self._conns += 1
                over = False
        if over:
            log.warning("connection_refused", peer=str(client_address[0]), reason="max_connections")
            with contextlib.suppress(OSError):
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n"
                )
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def add_inflight(self, delta: int) -> None:
        with self._conn_lock:
            self.inflight += delta

    def close_request(self, request: Any) -> None:
        with self._conn_lock:
            self._conns = max(0, self._conns - 1)
        super().close_request(request)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"kvhandoff/{__version__}"
    sys_version = ""
    server: TransferServer  # pyright: ignore[reportIncompatibleVariableOverride]
    _bytes_sent: int = 0
    _ROUTES: ClassVar[tuple[str, ...]] = ("GET", "POST")

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.cfg.io_timeout_s)  # per read/write on this connection
        with contextlib.suppress(OSError):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._hdr_lock = threading.Lock()
        self._headers_done = False

    # Header-read deadline. io_timeout_s bounds each socket read, so a client sending one byte at a time could hold a
    # connection (and one of max_connections) for ever. Each request's line and headers, and the idle wait for the next
    # request on a kept-alive connection, must complete within header_timeout_s, or the socket is shut down.
    def handle_one_request(self) -> None:
        with self._hdr_lock:
            self._headers_done = False
        timer = threading.Timer(self.server.cfg.header_timeout_s, self._header_deadline)
        timer.daemon = True
        timer.start()
        try:
            super().handle_one_request()
        finally:
            timer.cancel()

    def parse_request(self) -> bool:
        with self._hdr_lock:
            self._headers_done = True
        return super().parse_request()

    def _header_deadline(self) -> None:
        with self._hdr_lock:
            if self._headers_done:
                return
            log.warning("header_timeout", peer=str(self.client_address[0]))
            self.close_connection = True
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)

    def log_message(self, format: str, *args: Any) -> None:
        return  # replaced by structured logging

    # ------------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes = b"", ctype: str = "application/json", extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)
        self._bytes_sent = len(body)

    def _json(self, code: int, doc: dict[str, Any], extra: dict[str, str] | None = None) -> None:
        self._send(code, json.dumps(doc, separators=(",", ":")).encode(), extra=extra)

    def _authorised(self) -> bool:
        token = self.server.token
        if token is None:
            return True  # insecure_no_auth (tests only; refused by config otherwise)
        if bearer_ok(self.headers.get("Authorization"), token):
            return True
        log.warning("auth_failed", peer=self.client_address[0], path=urlsplit(self.path).path)
        self._json(401, {"error": "unauthorised"}, {"WWW-Authenticate": "Bearer"})
        return False

    def _drain_body(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        if 0 < n <= 65536:
            self.rfile.read(n)
        elif n > 65536:
            self.close_connection = True

    # ------------------------------------------------------------------ dispatch
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._drain_body()
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        t0 = time.perf_counter()
        url = urlsplit(self.path)
        path = url.path
        code = 0
        route = "unknown"
        tag = None
        try:
            if method == "GET" and path == "/healthz":
                route = "healthz"
                code = self._healthz()
                return
            if not path.startswith("/v1/"):
                route = "unknown"
                code = 404
                self._json(404, {"error": "not found"})
                return
            if not self._authorised():
                code = 401
                return
            if not self.server.transfers.acquire(blocking=False):
                code = 503
                log.warning("busy", peer=self.client_address[0], path=path)
                self._json(503, {"error": "busy"}, {"Retry-After": "1"})
                return
            self.server.add_inflight(1)
            try:
                if method == "GET" and (m := _HEADER.match(path)):
                    route, tag = "header", m.group(1)
                    code = self._header(tag)
                elif method == "GET" and (m := _STREAM.match(path)):
                    route, tag = "stream", m.group(1)
                    code = self._stream(tag)
                elif method == "GET" and (m := _LAYER.match(path)):
                    route, tag = "layer", m.group(1)
                    code = self._layer(tag, int(m.group(2)))
                elif method == "POST" and (m := _RELEASE.match(path)):
                    route, tag = "release", m.group(1)
                    code = self._release(tag, parse_qs(url.query))
                else:
                    code = 404
                    self._json(404, {"error": "not found"})
            finally:
                self.server.add_inflight(-1)
                self.server.transfers.release()
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as exc:
            code = code or 499
            self.close_connection = True
            log.warning("client_gone", peer=self.client_address[0], route=route, tag=tag, error=type(exc).__name__)
        except Exception:
            code = 500
            log.error("handler_error", exc_info=True, route=route, tag=tag)
            try:
                self._json(500, {"error": "internal error"})
            except OSError:
                self.close_connection = True
        finally:
            fields: dict[str, Any] = {
                "peer": self.client_address[0],
                "method": method,
                "route": route,
                "status": code,
                "bytes": self._bytes_sent,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            if tag is not None:
                fields["tag"] = tag
            if (route == "layer" and code == 200) or (route == "stream" and code in (200, 404)):  # per layer, or a poll
                log.debug("request", **fields)
            else:
                log.info("request", **fields)

    # ------------------------------------------------------------------ routes
    def _healthz(self) -> int:
        # Unauthenticated by design: counts and free bytes only, never a tag. The allowlist keeps
        # it that way if ExportReader.stats() ever grows a field.
        stats = self.server.reader.stats(self.server.cfg.hold_seconds)
        ok = bool(stats.get("root_ok"))
        store = {k: v for k, v in stats.items() if k in HEALTHZ_STORE_FIELDS and isinstance(v, int | float)}
        doc = {
            "status": "ok" if ok else "store_missing",
            "version": __version__,
            "store": store,
            "inflight": self.server.inflight,
            "uptime_s": round(time.time() - self.server.started, 1),
        }
        self._json(200 if ok else 503, doc)
        return 200 if ok else 503

    def _header(self, tag: str) -> int:
        try:
            h = self.server.reader.lookup(tag)
        except ExportFailed as exc:
            self._send(200, exc.header.to_json())
            return 200
        except ExportNotFound:
            self._json(404, {"error": "not found"})
            return 404
        self._send(200, h.to_json(), extra={"X-KVH-Seq": str(h.seq)})
        return 200

    def _stream(self, tag: str) -> int:
        try:
            st = self.server.reader.stream_state(tag)
        except ExportFailed as exc:
            self._json(200, {"published": False, "header": exc.header.to_dict()})
            return 200
        except ExportNotFound:
            self._json(404, {"error": "not found"})
            return 404
        doc = {"published": st.published, "header": st.header.to_dict()}
        self._json(200, doc, {"X-KVH-Seq": str(st.header.seq)})
        return 200

    def _layer(self, tag: str, index: int) -> int:
        seq_q = self.headers.get("X-KVH-Seq")
        seq = int(seq_q) if seq_q and seq_q.isdigit() else None
        try:
            if self.headers.get("X-KVH-Live") == "1" and seq is not None:
                ex = self.server.reader.open_stream(tag, seq)  # published, or the layers streamed so far
            else:
                ex = self.server.reader.open(tag, seq=seq)
        except (ExportNotFound, ExportFailed):
            self._json(404, {"error": "not found"})
            return 404
        with ex:
            try:
                layer = ex.header.layer(index)
            except KeyError:
                self._json(404, {"error": "no such layer"})
                return 404
            return self._send_layer(ex, index, layer.blobs)

    def _send_layer(self, ex: OpenExport, index: int, blobs: Any) -> int:
        total = sum(b.nbytes for b in blobs)
        path = ex.path(index)
        with open(path, "rb") as f:
            if os.fstat(f.fileno()).st_size < max((b.offset + b.nbytes for b in blobs), default=0):
                self._json(500, {"error": "layer file shorter than its header"})
                return 500
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(total))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-KVH-Seq", str(ex.header.seq))
            self.end_headers()
            self.wfile.flush()
            for b in blobs:  # zero-copy from the tmpfs page cache to the socket
                sent = 0
                while sent < b.nbytes:
                    n = self.connection.sendfile(f, b.offset + sent, b.nbytes - sent)
                    if n == 0:
                        raise BrokenPipeError("sendfile sent 0 bytes")
                    sent += n
            self._bytes_sent = total
        return 200

    def _release(self, tag: str, query: dict[str, list[str]]) -> int:
        seq = query.get("seq", [""])[0]
        if not seq.isdigit():
            self._json(400, {"error": "seq required"})
            return 400
        if self.server.reader.release(tag, int(seq)):
            self._send(204)
            return 204
        self._json(404, {"error": "not found"})
        return 404


def build_server(cfg: TransferConfig) -> TransferServer:
    token = None
    if not cfg.insecure_no_auth:
        assert cfg.token_file is not None
        token = load_token(cfg.token_file, allow_group_read=cfg.allow_group_readable_token)
    else:
        log.warning("auth_disabled", reason="insecure_no_auth = true; never use this in production")
    return TransferServer(cfg, token)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kvhandoff-transfer", description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--config", required=True, help="TOML file with a [transfer] table")
    a = ap.parse_args(argv)
    cfg = TransferConfig.from_mapping(load_toml(a.config, "transfer"))
    configure(cfg.log_level)
    srv = build_server(cfg)

    def _stop(signum: int, _frame: object) -> None:
        log.info("stopping", signal=signum)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log.info(
        "listening",
        bind=cfg.bind,
        port=cfg.port,
        store=cfg.store_path,
        version=__version__,
        max_concurrent_transfers=cfg.max_concurrent_transfers,
        max_connections=cfg.max_connections,
    )
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
