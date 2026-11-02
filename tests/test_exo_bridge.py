"""The EXO side end to end on localhost: a fake vLLM that exports through the real writers, the real transfer service,
and the bridge writing EXO's wire format (decoded back with EXO's own protocol module). Ported from the prototype's
concurrency test."""

from __future__ import annotations

import hashlib
import io
import json
import queue
import random
import threading
import time
import types
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from conftest import free_port, write_token
from kvhandoff.config import BridgeConfig, TransferConfig
from kvhandoff.exo.bridge import Bridge, ResultCache, _Result, bf16_to_fp16, token_key  # pyright: ignore[reportPrivateUsage]
from kvhandoff.interchange import KIND_DSV4, KIND_PAGED, ExportHeader, RopeInfo
from kvhandoff.producer.dsv4_export import DSV4Exporter, DSV4Job, DSV4ModelInfo
from kvhandoff.producer.paged import PagedExportInfo, PagedWriter, StoreJob
from kvhandoff.store import ExportReader, ExportStore
from kvhandoff.transfer.client import HandoffError
from kvhandoff.transfer.server import build_server
from test_dsv4 import fake_model

H, D = 2, 8
API_KEY = "vllm-key-" + "v" * 40
TOKEN = "xfer-" + "x" * 40
RATIOS = [0, 4, 128]


def kv_for(tokens: list[int]) -> torch.Tensor:
    seed = int(hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:8], 16)
    return torch.randn(2, len(tokens), H, D, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


class FakeVllm:
    """Answers /v1/completions at once and exports one engine step later (as vLLM's connectors do), batching every
    request that arrived within 50 ms into one step."""

    def __init__(self, mode: str, store: ExportStore, require_key: str | None = None, stream: bool = False):
        self.mode, self.store, self.require_key = mode, store, require_key
        self.stream = stream  # paged: the writer streams layers and the bridge is built with stream_layers
        self.posts: list[dict[str, Any]] = []
        self.batches: list[int] = []
        self.behaviour: dict[int, str] = {}  # first token -> "hang" | "error"
        self.jobs: queue.Queue[tuple[str, list[int]] | None] = queue.Queue()
        if mode == "paged":
            self.writer = PagedWriter(
                store,
                PagedExportInfo(
                    model_id="llama", num_model_layers=2, max_tokens=1024, rope=RopeInfo("neox", "vllm"), producer={}
                ),
                stream_layers=stream,
            )
        else:
            self.exporter = DSV4Exporter(
                store,
                DSV4ModelInfo(
                    model_id="dsv4",
                    num_layers=len(RATIOS),
                    compress_ratios=tuple(RATIOS),
                    rope_cfg={
                        "rope_theta": 10000.0,
                        "compress_rope_theta": 160000.0,
                        "rope_scaling": {"factor": 16, "original_max_position_embeddings": 65536},
                    },
                ),
            )
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                if outer.require_key and self.headers.get("Authorization") != f"Bearer {outer.require_key}":
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.posts.append(body)
                assert body["max_tokens"] == 1
                time.sleep(random.uniform(0.01, 0.1))  # prefill
                outer.jobs.put((body["kv_transfer_params"]["kvhandoff_tag"], body["prompt"]))
                out = b'{"choices":[{"text":"x"}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.http = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
        self.http.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        threading.Thread(target=self._engine, daemon=True).start()

    def _engine(self) -> None:
        while True:
            first = self.jobs.get()
            if first is None:
                return
            batch = [first]
            end = time.time() + 0.05
            while time.time() < end:
                try:
                    item = self.jobs.get(timeout=max(0.0, end - time.time()))
                except queue.Empty:
                    break
                if item is None:
                    return
                batch.append(item)
            self.batches.append(len(batch))
            for tag, toks in batch:
                how = self.behaviour.get(toks[0])
                if how == "hang":
                    continue
                if self.mode == "paged":
                    job = StoreJob(req_id=tag, tag=tag, num_tokens=len(toks), block_ids=[1])
                    if how == "error":
                        self.writer.mark_failed(job, "RuntimeError: injected")
                    else:
                        kv = kv_for(toks)
                        for li in range(2):
                            self.writer.write_layer(job, li, f"model.layers.{li}.self_attn.attn", kv * (li + 1))
                else:
                    kv, groups, mods, bids, _ = fake_model(len(toks), RATIOS, seed=toks[0])
                    if how == "error":
                        del kv["model.layers.1.attn.indexer.k_cache"]
                    self.exporter.export(DSV4Job(req_id=tag, tag=tag, tokens=len(toks), block_ids=bids), kv, groups, mods)
            if self.mode == "paged":
                self.writer.finish_step()

    def close(self) -> None:
        self.jobs.put(None)
        self.http.shutdown()


def req(tokens: list[int], rid: str, start: int = 0) -> Any:
    return types.SimpleNamespace(request_id=rid, model_id="m", token_ids=tokens, start_pos=start)


def decode(proto: Any, raw: bytes) -> tuple[Any, list[Any]]:
    s = io.BytesIO(raw)
    header = proto.read_header(s)
    msgs = []
    while (m := proto.read_message(s)) is not None:
        msgs.append(m)
        if isinstance(m, proto.Done):
            break
    return header, msgs


# ----------------------------------------------------------------------------------------------- paged
@pytest.fixture(params=[False, True], ids=["buffered", "streamed"])
def paged_env(request: pytest.FixtureRequest, tmp_path: Path, store_root: str) -> Iterator[tuple[FakeVllm, Path]]:
    """Every paged test runs twice: as before, and with stream_layers on in both the writer and the bridge (this
    FakeVllm answers before it exports, so the streamed bridge also meets an export it cannot overlap)."""
    store = ExportStore(store_root, slots=6, hold_seconds=60)
    v = FakeVllm("paged", store, require_key=API_KEY, stream=bool(request.param))
    key = write_token(tmp_path / "vllm.key", API_KEY)
    try:
        yield v, key
    finally:
        v.close()


def paged_bridge(v: FakeVllm, key: Path, store_root: str, **kw: Any) -> Bridge:
    cfg = BridgeConfig.from_mapping(
        {
            "mode": "paged",
            "vllm_url": v.url,
            "vllm_model": "llama",
            "vllm_api_key_file": str(key),
            "local_store_path": store_root,
            "timeout_s": 10,
            "stream_layers": v.stream,
            **kw,
        }
    )
    return Bridge(cfg)


def test_paged_concurrent_handoffs_each_carry_their_own_kv(exo_protocol: Any, paged_env: Any, store_root: str) -> None:
    v, key = paged_env
    br = paged_bridge(v, key, store_root)
    prompts = {p: [p * 100 + i for i in range(60 + 37 * p)] for p in range(4)}
    outs: dict[int, bytes] = {}
    errs: dict[int, str] = {}

    def run(p: int) -> None:
        buf = io.BytesIO()
        try:
            br.resolve(req(prompts[p], f"L{p}"), buf)
            outs[p] = buf.getvalue()
        except Exception as e:  # noqa: BLE001
            errs[p] = repr(e)

    ths = [threading.Thread(target=run, args=(p,)) for p in prompts]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not errs and len(v.posts) == 4
    assert len({b["kv_transfer_params"]["kvhandoff_tag"] for b in v.posts}) == 4
    assert max(v.batches) >= 2  # some engine step exported several requests
    for p, toks in prompts.items():
        n = len(toks)
        padded = toks + [toks[-1]] * 17  # block_size + 1
        assert [b["prompt"] for b in v.posts if b["prompt"][:n] == toks] == [padded]
        header, msgs = decode(exo_protocol, outs[p])
        assert header.num_layers == 2 and header.dtype == "bfloat16" and header.request_id == f"L{p}"
        kv = kv_for(padded)[:, :n]
        for li, m in enumerate(msgs[:-1]):
            want = (kv * (li + 1)).contiguous()
            assert isinstance(m, exo_protocol.KVChunk) and m.layer_idx == li and m.num_tokens == n
            assert (m.n_heads, m.head_dim) == (H, D)
            assert m.keys == want[0].view(torch.int16).numpy().tobytes()
            assert m.values == want[1].view(torch.int16).numpy().tobytes()
        assert isinstance(msgs[-1], exo_protocol.Done) and msgs[-1].total_tokens == n
    stats = ExportReader(store_root).stats()
    assert stats["published"] == 4


def test_paged_start_pos_and_fp16_wire(exo_protocol: Any, paged_env: Any, store_root: str) -> None:
    v, key = paged_env
    br = paged_bridge(v, key, store_root, wire_dtype="float16", pad_tokens=0)
    toks = list(range(500, 590))
    buf = io.BytesIO()
    br.resolve(req(toks, "s", start=30), buf)
    header, msgs = decode(exo_protocol, buf.getvalue())
    assert header.dtype == "float16" and header.start_pos == 30
    assert v.posts[-1]["prompt"] == toks  # pad_tokens = 0
    want = kv_for(toks)[:, 30:].contiguous()
    m = msgs[0]
    assert m.num_tokens == 60 and m.keys == bf16_to_fp16(want[0].view(torch.int16).numpy().tobytes())
    assert m.keys == want[0].to(torch.float16).numpy().tobytes()  # numpy cast == torch cast
    assert msgs[-1].total_tokens == 60


def test_paged_failures_raise_before_writing(exo_protocol: Any, paged_env: Any, store_root: str, tmp_path: Path) -> None:
    v, key = paged_env
    br = paged_bridge(v, key, store_root, timeout_s=2)
    for how in ("error", "hang"):
        toks = [900000 + len(how), *range(20)]
        v.behaviour[toks[0]] = how
        buf = io.BytesIO()
        t0 = time.monotonic()
        with pytest.raises((HandoffError, TimeoutError), match="injected" if how == "error" else "budget"):
            br.resolve(req(toks, how), buf)
        assert buf.getvalue() == b"" and time.monotonic() - t0 < 3.5
        v.behaviour.pop(toks[0])
        br.resolve(req(toks, "retry"), io.BytesIO())  # the same prompt succeeds later
    bad = write_token(tmp_path / "bad.key", "w" * 40)
    with pytest.raises(HandoffError, match="401"):
        paged_bridge(v, bad, store_root).resolve(req([1, 2, 3], "k"), io.BytesIO())
    with pytest.raises(HandoffError, match="bad request"):
        br.resolve(req([1, 2, 3], "k", start=3), io.BytesIO())


def test_paged_inflight_limit(exo_protocol: Any, paged_env: Any, store_root: str) -> None:
    v, key = paged_env
    br = paged_bridge(v, key, store_root, max_inflight=1, queue_wait_s=0.2)
    assert br._inflight.acquire(blocking=False)  # pyright: ignore[reportPrivateUsage]
    try:
        with pytest.raises(HandoffError, match="in flight"):
            br.resolve(req([1, 2, 3], "busy"), io.BytesIO())
    finally:
        br._inflight.release()  # pyright: ignore[reportPrivateUsage]


def test_paged_refuses_wrong_model(exo_protocol: Any, paged_env: Any, store_root: str) -> None:
    v, key = paged_env
    with pytest.raises(HandoffError, match="from model 'llama'"):
        paged_bridge(v, key, store_root, expect_model_id="other").resolve(req([5, 6, 7], "m"), io.BytesIO())


# ----------------------------------------------------------------------------------------------- dsv4
@pytest.fixture
def dsv4_env(tmp_path: Path, store_root: str) -> Iterator[tuple[FakeVllm, str]]:
    store = ExportStore(store_root, slots=16, hold_seconds=60)
    v = FakeVllm("dsv4", store)
    tok = write_token(tmp_path / "xfer.token", TOKEN)
    srv = build_server(
        TransferConfig.from_mapping(
            {"bind": "127.0.0.1", "port": free_port(), "store_path": store_root, "token_file": str(tok)}
        )
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg_path = tmp_path / "bridge.toml"
    cfg_path.write_text(
        f'[bridge]\nmode = "dsv4"\nvllm_url = "{v.url}"\nvllm_model = "dsv4"\n'
        f'transfer_url = "http://127.0.0.1:{srv.server_address[1]}"\n'
        f'transfer_token_file = "{tok}"\ntimeout_s = 20\nfetch_threads = 4\n'
    )
    try:
        yield v, str(cfg_path)
    finally:
        srv.shutdown()
        srv.server_close()
        v.close()


def expected_dsv4_stream(proto: Any, store_root: str, tag: str, rid: str, start: int, t: int) -> bytes:
    """Built straight from the store files, not through HTTP or the bridge."""
    r = ExportReader(store_root)
    h = r.lookup(tag)
    buf = io.BytesIO()
    proto.write_header(
        buf, proto.Header(request_id=rid, model_id="m", num_layers=len(h.layers), dtype="bfloat16", start_pos=start)
    )
    with r.open(tag) as ex:
        for layer in h.layers:
            proto.write_arrays_state(
                buf,
                layer.index,
                [
                    proto.TensorBlob(dtype=b.dtype, shape=b.shape, data=ex.read_range(layer.index, b.offset, b.nbytes))
                    for b in layer.blobs
                ],
            )
    proto.write_done(buf, total_tokens=t - start)
    return buf.getvalue()


def test_dsv4_ranks_share_one_prefill_and_get_identical_streams(
    exo_protocol: Any, dsv4_env: Any, store_root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    v, cfg_path = dsv4_env
    import kvhandoff.exo.bridge as bmod

    monkeypatch.setenv(bmod.CONFIG_ENV, cfg_path)
    monkeypatch.setattr(bmod, "_bridge", None)
    prompts = {p: [p * 1000 + i for i in range(T)] for p, T in ((1, 150), (2, 260), (3, 129), (4, 200))}
    calls = [(p, r) for p in prompts for r in range(3)]
    random.Random(7).shuffle(calls)
    outs: dict[tuple[int, int], bytes] = {}
    errs: dict[tuple[int, int], str] = {}

    def run(p: int, r: int) -> None:
        time.sleep(random.uniform(0, 0.1))
        buf = io.BytesIO()
        try:
            bmod.resolve(req(prompts[p], f"p{p}r{r}", start=r), buf)  # what the EXO hook calls
            outs[(p, r)] = buf.getvalue()
        except Exception as e:  # noqa: BLE001
            errs[(p, r)] = repr(e)

    ths = [threading.Thread(target=run, args=c) for c in calls]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not errs
    assert len(v.posts) == 4  # one vLLM call per prompt for 12 rank calls
    tag_of = {b["prompt"][0] // 1000: b["kv_transfer_params"]["kvhandoff_tag"] for b in v.posts}
    for p, r in calls:
        assert outs[(p, r)] == expected_dsv4_stream(exo_protocol, store_root, tag_of[p], f"p{p}r{r}", r, len(prompts[p]))
    header, msgs = decode(exo_protocol, outs[(1, 0)])
    assert header.num_layers == 3 and [m.layer_idx for m in msgs[:-1]] == [0, 1, 2]
    assert msgs[-1].total_tokens == 150
    assert {b["kv_transfer_params"]["kvhandoff_tag"][:24] for b in v.posts} == {token_key(t) for t in prompts.values()}
    bridge = bmod.get_bridge()
    assert len(bridge.results) <= 8
    # every export was released after the fetch: the slots are free for reuse at once
    assert ExportReader(store_root).stats()["consumed"] == 4
    bridge.close()


def test_dsv4_failures_fail_every_rank_fast_and_retry_later(exo_protocol: Any, dsv4_env: Any, tmp_path: Path) -> None:
    v, cfg_path = dsv4_env
    from kvhandoff.config import load_toml

    cfg = BridgeConfig.from_mapping({**load_toml(cfg_path, "bridge"), "timeout_s": 3})
    br = Bridge(cfg)
    for how in ("error", "hang"):
        toks = [700000 + len(how), *range(30)]
        v.behaviour[toks[0]] = how
        n0 = len(v.posts)
        res: dict[int, tuple[str, bytes]] = {}

        def run(r: int, toks: list[int] = toks, res: dict[int, tuple[str, bytes]] = res) -> None:
            buf = io.BytesIO()
            try:
                br.resolve(req(toks, f"f{r}"), buf)
                res[r] = ("ok", buf.getvalue())
            except Exception as e:  # noqa: BLE001
                res[r] = (type(e).__name__ + ": " + str(e), buf.getvalue())

        t0 = time.monotonic()
        ths = [threading.Thread(target=run, args=(r,)) for r in range(3)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        took = time.monotonic() - t0
        assert all(msg != "ok" and out == b"" for msg, out in res.values()), res
        assert len(v.posts) == n0 + 1  # ranks that waited on the failed attempt share its error
        assert took < (2 if how == "error" else 4.5)
        if how == "error":
            assert all("missing idx_pool" in msg for msg, _ in res.values())
        v.behaviour.pop(toks[0])
        buf = io.BytesIO()
        br.resolve(req(toks, "retry"), buf)
        assert buf.getvalue() and len(v.posts) == n0 + 2
    br.close()


def test_result_cache_is_bounded_by_entries_and_bytes() -> None:
    c = ResultCache(max_entries=3, max_bytes=100)
    for i in range(5):
        e = c.entry(f"k{i}")
        e.result = _Result(tokens=1, layers=[], nbytes=40, tag=f"t{i}")
        c.trim()
    assert len(c) <= 3 and c.bytes() <= 100
    busy = c.entry("busy")
    with busy.lock:
        for i in range(5, 10):
            c.entry(f"k{i}").result = _Result(tokens=1, layers=[], nbytes=40, tag="t")
            c.trim()
        assert "busy" in c._entries  # pyright: ignore[reportPrivateUsage]  # never evicted while being produced


def test_bridge_refuses_an_export_of_the_wrong_kind(exo_protocol: Any, paged_env: Any, store_root: str) -> None:
    v, key = paged_env
    br = paged_bridge(v, key, store_root)
    h = ExportHeader(kind=KIND_DSV4, tag="x", model_id="llama", num_tokens=3)
    with pytest.raises(HandoffError, match="expects paged_kv"):
        br._check(h, KIND_PAGED)  # pyright: ignore[reportPrivateUsage]
