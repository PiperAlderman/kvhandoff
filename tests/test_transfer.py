from __future__ import annotations

import io
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import free_port, write_token
from kvhandoff import logs
from kvhandoff.config import TransferConfig
from kvhandoff.interchange import KIND_DSV4, KIND_PAGED, BlobRef, ExportHeader, LayerRef
from kvhandoff.store import ExportReader, ExportStore
from kvhandoff.transfer.client import Deadline, HandoffError, LocalSource, TransferClient
from kvhandoff.transfer.server import HEALTHZ_STORE_FIELDS, TransferServer, build_server
from test_store import put

TOKEN = "s3cr3t-" + "k" * 40


class Running:
    def __init__(self, srv: TransferServer):
        self.srv = srv
        self.url = f"http://127.0.0.1:{srv.server_address[1]}"
        self.thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path: str, token: str | None = TOKEN, method: str = "GET") -> tuple[int, bytes, dict[str, str]]:
        hdrs = {"Authorization": f"Bearer {token}"} if token else {}
        req = urllib.request.Request(self.url + path, headers=hdrs, method=method, data=b"" if method == "POST" else None)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)


@pytest.fixture
def service(tmp_path: Path, store_root: str) -> Iterator[tuple[Running, ExportStore]]:
    store = ExportStore(store_root, slots=4, hold_seconds=3600)
    tok = write_token(tmp_path / "token", TOKEN)
    cfg = TransferConfig.from_mapping(
        {
            "bind": "127.0.0.1",
            "port": free_port(),
            "store_path": store_root,
            "token_file": str(tok),
            "max_concurrent_transfers": 2,
            "max_connections": 8,
            "io_timeout_s": 2.0,
            "hold_seconds": 3600,
        }
    )
    srv = build_server(cfg)
    run = Running(srv)
    try:
        yield run, store
    finally:
        srv.shutdown()
        srv.server_close()


def multi_blob(tag: str) -> tuple[ExportHeader, bytes]:
    head, a, b = b"H" * 8, b"A" * 64, b"B" * 32
    payload = head + a + b"\0" * 16 + b  # a gap between blobs, like paged_kv's K/V layout
    blobs = (
        BlobRef("state_header", "float32", (2,), 0, 8),
        BlobRef("x", "bfloat16", (32,), 8, 64),
        BlobRef("y", "bfloat16", (16,), 88, 32),
    )
    layer = LayerRef(index=3, name="l3", file="L3.bin", nbytes=len(payload), blobs=blobs)
    return ExportHeader(kind=KIND_DSV4, tag=tag, model_id="m", num_tokens=4, layers=(layer,)), payload


def test_healthz_needs_no_token_and_reports_the_store(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    put(store, "a-1")
    code, body, _ = run.get("/healthz", token=None)
    doc = json.loads(body)
    assert code == 200 and doc["status"] == "ok" and doc["store"]["slots"] == 4 and doc["store"]["published"] == 1
    assert doc["store"]["held"] == 1 and "a-1" not in body.decode()


def test_healthz_stays_open_and_never_shows_a_tag(service: tuple[Running, ExportStore]) -> None:
    """By design: /healthz is unauthenticated, with counts and free bytes only. Every place a tag can live
    in the store is populated (published, consumed, failure record, an export still streaming), and the answer must
    carry none of them, with exactly the allowed fields, whatever Authorization header is sent."""
    run, store = service
    put(store, "hz-published-1")
    put(store, "hz-consumed-2")
    h = ExportReader(store.root).lookup("hz-consumed-2")
    assert ExportReader(store.root).release("hz-consumed-2", h.seq)
    store.publish_failure(ExportHeader(kind=KIND_DSV4, tag="hz-failed-3", model_id="m", num_tokens=1, error="boom"))
    store.begin(
        "hz-streaming-4",
        stream=ExportHeader(kind=KIND_PAGED, tag="hz-streaming-4", model_id="m", num_tokens=4, num_model_layers=2),
    )
    allowed_top = {"status", "version", "store", "inflight", "uptime_s"}
    for token in (None, "wrong" * 10, TOKEN):
        code, body, _ = run.get("/healthz", token=token)
        assert code == 200, token
        doc = json.loads(body)
        assert set(doc) == allowed_top
        assert set(doc["store"]) == set(HEALTHZ_STORE_FIELDS)
        assert all(isinstance(v, int | float) for v in doc["store"].values())
        assert doc["store"]["published"] == 2 and doc["store"]["consumed"] == 1 and doc["store"]["failed_records"] == 1
        text = body.decode()
        assert "hz-" not in text and store.root not in text
    assert run.get("/healthz", method="POST")[0] == 404  # only GET is served without auth


def test_v1_needs_the_token(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    put(store, "a-1")
    assert run.get("/v1/exports/a-1", token=None)[0] == 401
    assert run.get("/v1/exports/a-1", token="wrong" * 10)[0] == 401
    assert run.get("/v1/exports/a-1")[0] == 200
    assert run.get("/v1/exports/nope")[0] == 404
    assert run.get("/v1/exports/..%2f..%2fetc")[0] == 404
    assert run.get("/other")[0] == 404


def test_layer_endpoint_sends_only_the_blobs_with_seq(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    hdr, payload = multi_blob("m-1")
    k = store.begin("m-1")
    store.write_bytes(k, "L3.bin", 0, payload + b"\xee" * 1000)  # file larger than the export
    h = store.publish(k, hdr)
    code, body, hdrs = run.get("/v1/exports/m-1/layers/3")
    assert code == 200 and body == b"H" * 8 + b"A" * 64 + b"B" * 32 and hdrs["X-KVH-Seq"] == str(h.seq)
    assert run.get("/v1/exports/m-1/layers/4")[0] == 404
    client = TransferClient(run.url, TOKEN)
    got = client.layer_blobs(h, h.layers[0], Deadline(5))
    assert got == [b"H" * 8, b"A" * 64, b"B" * 32]


def test_client_waits_for_publish_then_reads(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    client = TransferClient(run.url, TOKEN)

    def later() -> None:
        time.sleep(0.3)
        put(store, "late-1")

    threading.Thread(target=later).start()
    h = client.wait_header("late-1", Deadline(5))
    assert h.tag == "late-1"


def test_failure_record_fails_the_client_at_once(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    store.publish_failure(ExportHeader(kind=KIND_DSV4, tag="f-1", model_id="m", num_tokens=3, error="StoreFull: x"))
    t0 = time.monotonic()
    with pytest.raises(HandoffError, match="StoreFull"):
        TransferClient(run.url, TOKEN).wait_header("f-1", Deadline(10))
    assert time.monotonic() - t0 < 1


def test_client_deadline(service: tuple[Running, ExportStore]) -> None:
    run, _ = service
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        TransferClient(run.url, TOKEN).wait_header("never", Deadline(0.5))
    assert time.monotonic() - t0 < 1.5


def test_wrong_token_and_dead_service_fail_without_polling(service: tuple[Running, ExportStore]) -> None:
    run, _ = service
    with pytest.raises(HandoffError, match="401"):
        TransferClient(run.url, "x" * 40).wait_header("a", Deadline(5))
    with pytest.raises(HandoffError, match="unreachable"):
        TransferClient(f"http://127.0.0.1:{free_port()}", TOKEN).wait_header("a", Deadline(5))


def test_release_marks_consumed(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    h = put(store, "r-1")
    assert run.get(f"/v1/exports/r-1/release?seq={h.seq + 1}", method="POST")[0] == 404
    assert run.get("/v1/exports/r-1/release", method="POST")[0] == 400
    assert run.get(f"/v1/exports/r-1/release?seq={h.seq}", method="POST")[0] == 204
    TransferClient(run.url, TOKEN).release(h)  # second release is harmless


def test_reuse_between_requests_is_detected(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    h1 = put(store, "same")
    put(store, "same")  # re-exported: the old seq is gone
    with pytest.raises(HandoffError, match="gone"):
        TransferClient(run.url, TOKEN).layer_blobs(h1, h1.layers[0], Deadline(5))


def test_concurrency_limit_answers_503(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    put(store, "a-1")
    for _ in range(2):
        assert run.srv.transfers.acquire(blocking=False)
    try:
        code, _, hdrs = run.get("/v1/exports/a-1")
        assert code == 503 and hdrs.get("Retry-After") == "1"
    finally:
        run.srv.transfers.release()
        run.srv.transfers.release()
    assert run.get("/v1/exports/a-1")[0] == 200


def test_connection_cap(service: tuple[Running, ExportStore]) -> None:
    run, _ = service
    port = int(run.url.rsplit(":", 1)[1])
    held = [socket.create_connection(("127.0.0.1", port)) for _ in range(8)]
    try:
        time.sleep(0.2)
        extra = socket.create_connection(("127.0.0.1", port))
        extra.settimeout(2)
        assert extra.recv(64).startswith(b"HTTP/1.1 503")
        extra.close()
    finally:
        for s in held:
            s.close()
    time.sleep(0.3)
    assert run.get("/healthz", token=None)[0] == 200


def test_idle_connection_times_out(service: tuple[Running, ExportStore]) -> None:
    run, _ = service
    port = int(run.url.rsplit(":", 1)[1])
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(5)
    s.sendall(b"GET /healthz HTTP/1.1\r\n")  # never finishes the request
    t0 = time.monotonic()
    assert s.recv(64) == b""  # server closes after io_timeout_s
    assert time.monotonic() - t0 < 4
    s.close()


def test_logs_are_json_lines_without_the_token(service: tuple[Running, ExportStore]) -> None:
    run, store = service
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logs.JsonFormatter())
    lg = logging.getLogger("kvhandoff")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        put(store, "a-1")
        run.get("/v1/exports/a-1")
        run.get("/v1/exports/a-1", token="bad-token-" + "z" * 30)
        time.sleep(0.1)
    finally:
        lg.removeHandler(handler)
    lines: list[dict[str, Any]] = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert any(x["event"] == "request" and x.get("tag") == "a-1" and x["status"] == 200 for x in lines)
    assert any(x["event"] == "auth_failed" for x in lines)
    assert TOKEN not in buf.getvalue() and "bad-token" not in buf.getvalue()


def test_local_source_reads_the_same_bytes(store_root: str) -> None:
    store = ExportStore(store_root, slots=2)
    hdr, payload = multi_blob("m-2")
    k = store.begin("m-2")
    store.write_bytes(k, "L3.bin", 0, payload)
    store.publish(k, hdr)
    src = LocalSource(store_root)
    h = src.wait_header("m-2", Deadline(2))
    assert src.layer_blobs(h, h.layers[0], Deadline(2)) == [b"H" * 8, b"A" * 64, b"B" * 32]
    src.release(h)
    with pytest.raises(TimeoutError):
        src.wait_header("absent", Deadline(0.3))


# ----------------------------------------------------------------------------------------------- review fixes (3/11)
def test_slow_header_client_is_cut_off_by_the_header_deadline(tmp_path: Path, store_root: str) -> None:
    """A client that sends one byte at a time, each within io_timeout_s, must still lose its connection (and its
    max_connections place) once header_timeout_s has passed without a complete request line and headers."""
    ExportStore(store_root, slots=1)
    tok = write_token(tmp_path / "token", TOKEN)
    cfg = TransferConfig.from_mapping(
        {
            "bind": "127.0.0.1",
            "port": free_port(),
            "store_path": store_root,
            "token_file": str(tok),
            "io_timeout_s": 5.0,
            "header_timeout_s": 1.0,
            "max_connections": 2,
        }
    )
    srv = build_server(cfg)
    run = Running(srv)
    try:
        s = socket.create_connection(("127.0.0.1", srv.server_address[1]))
        s.settimeout(0.2)
        t0 = time.monotonic()
        closed = False
        for b in b"GET /healthz HTTP/1.1\r\nHost: x\r\nX-Pad: " + b"a" * 200:
            try:
                s.sendall(bytes([b]))
                if s.recv(64) == b"":
                    closed = True
                    break
            except TimeoutError:
                continue
            except OSError:
                closed = True
                break
            if time.monotonic() - t0 > 4:
                break
        s.close()
        assert closed and time.monotonic() - t0 < 3, "slow client kept its connection past header_timeout_s"
        assert run.get("/healthz", token=None)[0] == 200  # the place was given back; normal requests still work
        code, _, _ = run.get("/v1/exports/nope")
        assert code == 404
    finally:
        srv.shutdown()
        srv.server_close()


class _Recorder:
    """A tiny HTTP server that records each request's path and Authorization header and answers ``status``."""

    def __init__(self, status: int = 200, location: str | None = None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        seen: list[tuple[str, str | None]] = []
        self.seen = seen

        class H(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                seen.append((self.path, self.headers.get("Authorization")))
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_GET(self) -> None:
                self._answer()

            def do_POST(self) -> None:
                self._answer()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def test_bearer_requests_never_follow_redirects() -> None:
    """urllib's default opener would follow a 302 and resend Authorization to the new host."""
    from kvhandoff.transfer.client import VllmClient

    target = _Recorder()
    bouncer = _Recorder(status=302, location=target.url + "/stolen")
    try:
        code, _, _ = TransferClient(bouncer.url, TOKEN)._request("GET", "/v1/exports/x", 2.0)
        assert code == 302
        with pytest.raises(HandoffError, match="302"):
            VllmClient(bouncer.url, "m", api_key="k" * 40).prefill([1, 2], {"kvhandoff_tag": "t"}, Deadline(2.0))
        assert target.seen == []
        assert [a for _, a in bouncer.seen] == [f"Bearer {TOKEN}", "Bearer " + "k" * 40]
    finally:
        target.close()
        bouncer.close()


def test_bearer_requests_ignore_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A proxy set in the environment must not receive the bearer token: requests go straight to the configured URL.
    The client module is reloaded with the proxy variables set, as it would be imported in such a process."""
    import importlib

    from kvhandoff.transfer import client as client_mod

    target = _Recorder()
    proxy = _Recorder()
    try:
        for var in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(var, raising=False)
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy"):
            monkeypatch.setenv(var, proxy.url)
        importlib.reload(client_mod)
        with urllib.request.build_opener().open(target.url + "/probe", timeout=2.0) as r:  # control: urllib's default
            r.read()
        assert len(proxy.seen) == 1 and target.seen == []  # ... does use the environment's proxy
        handlers = getattr(client_mod._OPENER, "handlers", [])
        assert not any(isinstance(h, urllib.request.ProxyHandler) and getattr(h, "proxies", None) for h in handlers)
        code, _, _ = client_mod.TransferClient(target.url, TOKEN)._request("GET", "/v1/exports/x", 2.0)
        assert code == 200
        assert len(proxy.seen) == 1 and target.seen == [("/v1/exports/x", f"Bearer {TOKEN}")]
    finally:
        target.close()
        proxy.close()
