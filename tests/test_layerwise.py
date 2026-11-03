"""Layer-by-layer streaming (stream_layers; docs/layerwise.md): the store's streaming mode, the paged writer announcing each
layer as it is written, the bridge sending layers while vLLM is still prefilling, and the failure, reuse and
tensor-parallel cases."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import threading
import time
import types
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from conftest import free_port, write_token
from fakes import vllm_stub
from kvhandoff.config import BridgeConfig, ConfigError, ConnectorConfig, TransferConfig
from kvhandoff.exo.bridge import Bridge
from kvhandoff.interchange import KIND_PAGED, ExportHeader, RopeInfo, paged_layer
from kvhandoff.producer.paged import PagedExportInfo, PagedWriter, StoreJob, gather, slot_mapping, to_kv_split
from kvhandoff.store import STREAM_LOG, ExportFailed, ExportNotFound, ExportReader, ExportStore, StoreFull
from kvhandoff.transfer.client import HandoffError
from kvhandoff.transfer.server import build_server
from test_exo_bridge import decode, kv_for, req
from test_paged import BS, D, H, new, paged_layer_tensor, sched_out
from test_vllm_connectors import vllm_cfg

LAYERS = 4
TOKEN = "xfer-" + "s" * 40


def descriptor(tag: str, n: int, layers: int = LAYERS) -> ExportHeader:
    return ExportHeader(
        kind=KIND_PAGED, tag=tag, model_id="llama", num_tokens=n, num_model_layers=layers, rope=RopeInfo("neox", "vllm")
    )


def info(layers: int = LAYERS, max_tokens: int = 256) -> PagedExportInfo:
    return PagedExportInfo(
        model_id="llama", num_model_layers=layers, max_tokens=max_tokens, rope=RopeInfo("neox", "vllm"), producer={}
    )


def write_raw(store: ExportStore, k: int, index: int, n: int, fill: int) -> None:
    """One paged layer file with K rows then V rows (capacity n), every byte ``fill``."""
    ref = paged_layer(index, f"l{index}", num_tokens=n, heads=H, head_dim=D, dtype="float16", row_capacity=n)
    store.write_bytes(k, ref.file, 0, bytes([fill]) * ref.nbytes)
    store.stream_layer(k, ref)


# ----------------------------------------------------------------------------------------------- store
def test_streamed_layers_are_readable_before_publish_and_only_through_the_stream(store_root: str) -> None:
    store = ExportStore(store_root, slots=2)
    r = ExportReader(store_root)
    k = store.begin("s-1", stream=descriptor("s-1", 8))
    seq = store.stream_seq(k)
    st = r.stream_state("s-1")
    assert not st.published and st.header.layers == () and st.header.seq == seq
    write_raw(store, k, 0, 8, 7)
    st = r.stream_state("s-1")
    assert [x.index for x in st.header.layers] == [0]
    with r.open_stream("s-1", seq) as ex:
        assert ex.read_blob(0, "k") == bytes([7]) * 8 * H * D * 2
        with pytest.raises(KeyError):
            ex.header.layer(1)  # not complete yet: never readable
    # a reader that knows nothing of streaming sees nothing until publish (no partial header.json, no tags/ link)
    with pytest.raises(ExportNotFound):
        r.lookup("s-1")
    with pytest.raises(ExportNotFound):
        r.open("s-1")
    for i in range(1, LAYERS):
        write_raw(store, k, i, 8, 7 + i)
    h = store.publish(k, dataclasses.replace(descriptor("s-1", 8), layers=r.stream_state("s-1").header.layers))
    assert h.seq == seq  # the seq readers started on is kept
    st = r.stream_state("s-1")
    assert st.published and st.header == r.lookup("s-1")
    with r.open_stream("s-1", seq) as ex:  # a streaming reader carries on across publish
        assert ex.read_blob(3, "v") == bytes([10]) * 8 * H * D * 2


def test_out_of_order_layers_are_listed_by_index(store_root: str) -> None:
    store = ExportStore(store_root, slots=1)
    k = store.begin("s-o", stream=descriptor("s-o", 4))
    for i in (2, 0, 3):
        write_raw(store, k, i, 4, i)
    st = ExportReader(store_root).stream_state("s-o")
    assert [x.index for x in st.header.layers] == [0, 2, 3]


def test_failed_stream_fails_readers_and_a_reader_blocks_reuse(store_root: str) -> None:
    store = ExportStore(store_root, slots=1, hold_seconds=3600)
    r = ExportReader(store_root)
    k = store.begin("s-a", stream=descriptor("s-a", 4))
    seq = store.stream_seq(k)
    write_raw(store, k, 0, 4, 1)
    ex = r.open_stream("s-a", seq)  # a reader is part way through layer 0 of the stream
    store.stream_failed(k, "RuntimeError: worker died")
    with pytest.raises(ExportFailed, match="worker died"):
        r.stream_state("s-a")
    with pytest.raises(ExportNotFound):
        r.open_stream("s-a", seq)
    store.abandon(k)
    with pytest.raises(StoreFull):  # the only slot is free, but a reader still holds it
        store.begin("s-b", stream=descriptor("s-b", 4))
    assert ex.read_blob(0, "k") == bytes([1]) * 4 * H * D * 2  # its bytes are intact while it holds the slot
    ex.close()
    k2 = store.begin("s-b", stream=descriptor("s-b", 4))
    write_raw(store, k2, 0, 4, 9)
    # the old tag never reaches the new export's bytes, by link or by seq
    with pytest.raises(ExportNotFound):
        r.open_stream("s-a", seq)
    with pytest.raises(ExportNotFound):
        r.open_stream("s-b", seq)
    with pytest.raises(ExportNotFound):
        r.stream_state("s-a")


def test_published_slot_reused_after_hold_is_detected_by_seq(store_root: str) -> None:
    now = [1000.0]
    store = ExportStore(store_root, slots=1, hold_seconds=10, clock=lambda: now[0])
    r = ExportReader(store_root)
    k = store.begin("s-x", stream=descriptor("s-x", 4, layers=1))
    write_raw(store, k, 0, 4, 1)
    old = store.publish(k, dataclasses.replace(descriptor("s-x", 4, 1), layers=r.stream_state("s-x").header.layers))
    now[0] += 11  # past hold: the slot may be reused although nobody released it
    k = store.begin("s-y", stream=descriptor("s-y", 4, layers=1))
    write_raw(store, k, 0, 4, 2)
    with pytest.raises(ExportNotFound):
        r.open_stream("s-x", old.seq)
    with pytest.raises(ExportNotFound):
        r.open("s-x", seq=old.seq)


def test_restart_drops_unfinished_streams(store_root: str) -> None:
    store = ExportStore(store_root, slots=2)
    k = store.begin("s-r", stream=descriptor("s-r", 4))
    write_raw(store, k, 0, 4, 1)
    os.close(store._writer_fd)  # the vLLM worker died: no publish, no failure line, its locks gone with it
    store._writer_fd = -1
    ExportStore(store_root, slots=2).close()
    with pytest.raises(ExportNotFound):
        ExportReader(store_root).stream_state("s-r")


def test_stream_descriptor_is_checked(store_root: str) -> None:
    store = ExportStore(store_root, slots=1)
    with pytest.raises(ValueError, match="descriptor"):
        store.begin("s-1", stream=descriptor("other", 4))


# ----------------------------------------------------------------------------------------------- writer
def test_writer_announces_each_layer_and_fails_the_stream_at_once(store_root: str) -> None:
    store = ExportStore(store_root, slots=2)
    w = PagedWriter(store, info(), stream_layers=True)
    r = ExportReader(store_root)
    job = StoreJob(req_id="r", tag="s-w", num_tokens=6, block_ids=[1])
    kv = torch.randn(2, 6, H, D).to(torch.float16)
    w.write_layer(job, 0, "model.layers.0.attn", kv)
    st = r.stream_state("s-w")
    with r.open_stream("s-w", st.header.seq) as ex:
        assert ex.read_blob(0, "k") == kv[0].numpy().tobytes()
    w.write_layer(job, 1, "model.layers.1.attn", torch.zeros(2, 5, H, D, dtype=torch.float16))  # wrong row count
    with pytest.raises(ExportFailed, match="rows gathered"):  # visible now, before wait_for_save
        r.stream_state("s-w")
    with pytest.raises(ExportFailed, match="rows gathered"):
        r.lookup("s-w")
    assert w.finish_step() == []
    job2 = StoreJob(req_id="r2", tag="s-w2", num_tokens=6, block_ids=[1])
    w.write_layer(job2, 0, "l0", kv)
    w.write_layer(job2, 0, "l0", kv)  # a reader may already have sent the first copy
    with pytest.raises(ExportFailed, match="written twice"):
        r.stream_state("s-w2")
    w.finish_step()


def test_writer_without_streaming_writes_no_stream_log(store_root: str) -> None:
    store = ExportStore(store_root, slots=1)
    w = PagedWriter(store, info(layers=1))
    w.write_layer(StoreJob(req_id="r", tag="b-1", num_tokens=3, block_ids=[1]), 0, "l0", torch.ones(2, 3, H, D))
    (h,) = w.finish_step()
    assert not (Path(store.slot_dir(h.slot)) / STREAM_LOG).exists()
    assert not any((Path(store_root) / "streaming").iterdir())


# ----------------------------------------------------------------------------------------------- bridge
class GatedVllm:
    """A fake vLLM that behaves like the real one: the POST is answered only after the step's forward has written
    every layer (in ``order``, each after ``gate(i)`` returns) and wait_for_save has published the export."""

    def __init__(self, store: ExportStore, *, stream: bool = True, layers: int = LAYERS):
        self.writer = PagedWriter(store, info(layers), stream_layers=stream)
        self.layers = layers
        self.order: list[int] = list(range(layers))
        self.gate: Callable[[int], None] = lambda _i: None
        self.fail_after: int | None = None  # stop after this many layers ...
        self.fail_how = "post"  # ... "post": answer 500 and write nothing more (engine died, request aborted);
        # "record": the writer fails the export (failure line and record at once) and the request still succeeds
        self._stale: StoreJob | None = None
        self.written: list[int] = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                code = outer.forward(body["kv_transfer_params"]["kvhandoff_tag"], body["prompt"])
                out = b'{"choices":[{"text":"x"}]}' if code == 200 else b"{}"
                self.send_response(code)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.http = ThreadingHTTPServer(("127.0.0.1", free_port()), Handler)
        self.http.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def forward(self, tag: str, toks: list[int]) -> int:
        with self.lock:
            if self._stale is not None:  # what a restarted engine would leave behind: give the slot back
                self.writer.mark_failed(self._stale, "RuntimeError: engine restarted")
                self.writer.finish_step()
                self._stale = None
            job = StoreJob(req_id=tag, tag=tag, num_tokens=len(toks), block_ids=[1])
            kv = kv_for(toks)
            for count, li in enumerate(self.order):
                if self.fail_after is not None and count == self.fail_after:
                    if self.fail_how == "post":
                        self._stale = job
                        return 500
                    self.writer.mark_failed(job, "OSError: injected write failure")
                    self.writer.finish_step()
                    return 200
                self.gate(li)
                self.writer.write_layer(job, li, f"model.layers.{li}.self_attn.attn", kv * (li + 1))
                self.written.append(li)
            self.writer.finish_step()
            return 200

    def close(self) -> None:
        self.http.shutdown()


def frames(raw: bytes) -> list[bytes]:
    out: list[bytes] = []
    pos = 0
    while pos + 4 <= len(raw):
        size = int.from_bytes(raw[pos : pos + 4], "big")
        if pos + 4 + size > len(raw):
            break
        out.append(raw[pos + 4 : pos + 4 + size])
        pos += 4 + size
    return out


class Recorder(io.BytesIO):
    """EXO's socket file as the bridge sees it; readable from the test thread while the bridge writes."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()

    def write(self, b: Any) -> int:
        with self.lock:
            return super().write(b)

    def snapshot(self) -> bytes:
        with self.lock:
            return self.getvalue()


def chunks_sent(proto: Any, raw: bytes) -> list[int]:
    msgs = frames(raw)[1:]  # after the Header
    return [
        m.layer_idx
        for m in (proto.read_message(io.BytesIO(len(f).to_bytes(4, "big") + f)) for f in msgs)
        if isinstance(m, proto.KVChunk)
    ]


@pytest.fixture(params=["local", "http"])
def gated(request: pytest.FixtureRequest, tmp_path: Path, store_root: str) -> Iterator[tuple[GatedVllm, dict[str, Any]]]:
    """A GatedVllm plus the bridge source settings: a store on this host, or the transfer service over HTTP."""
    store = ExportStore(store_root, slots=4, hold_seconds=60)
    v = GatedVllm(store)
    srv = None
    if request.param == "local":
        source: dict[str, Any] = {"local_store_path": store_root}
    else:
        tok = write_token(tmp_path / "xfer.token", TOKEN)
        srv = build_server(
            TransferConfig.from_mapping(
                {"bind": "127.0.0.1", "port": free_port(), "store_path": store_root, "token_file": str(tok)}
            )
        )
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        source = {"transfer_url": f"http://127.0.0.1:{srv.server_address[1]}", "transfer_token_file": str(tok)}
    try:
        yield v, source
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        v.close()


def bridge(v: GatedVllm, source: dict[str, Any], **kw: Any) -> Bridge:
    return Bridge(
        BridgeConfig.from_mapping({"mode": "paged", "vllm_url": v.url, "vllm_model": "llama", "timeout_s": 10, **source, **kw})
    )


def run_async(br: Bridge, r: Any, out: Recorder) -> tuple[threading.Thread, dict[str, BaseException]]:
    errs: dict[str, BaseException] = {}

    def go() -> None:
        try:
            br.resolve(r, out)
        except BaseException as e:  # noqa: BLE001
            errs["e"] = e

    t = threading.Thread(target=go)
    t.start()
    return t, errs


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def test_streaming_starts_before_the_last_layer_is_written(exo_protocol: Any, gated: Any) -> None:
    v, source = gated
    gates = [threading.Event() for _ in range(LAYERS)]
    v.gate = lambda i: gates[i].wait(10)
    toks = list(range(1000, 1090))
    out = Recorder()
    br = bridge(v, source, stream_layers=True, pad_tokens=0)
    t, errs = run_async(br, req(toks, "early", start=5), out)
    for i in range(LAYERS - 1):
        gates[i].set()
        assert wait_for(lambda i=i: chunks_sent(exo_protocol, out.snapshot()) == list(range(i + 1))), out.snapshot()[:64]
    time.sleep(0.1)
    assert v.written == list(range(LAYERS - 1))  # the last layer is not written yet ...
    assert chunks_sent(exo_protocol, out.snapshot()) == list(range(LAYERS - 1))  # ... and the others are already sent
    assert not any(
        isinstance(exo_protocol.read_message(io.BytesIO(len(f).to_bytes(4, "big") + f)), exo_protocol.Done)
        for f in frames(out.snapshot())[1:]
    )
    gates[-1].set()
    t.join(10)
    assert not errs
    # the streamed bytes equal what the buffered bridge sends for the same prompt
    buffered = Recorder()
    bridge(v, source, pad_tokens=0).resolve(req(toks, "early", start=5), buffered)
    assert out.getvalue() == buffered.getvalue()
    header, msgs = decode(exo_protocol, out.getvalue())
    assert header.num_layers == LAYERS and isinstance(msgs[-1], exo_protocol.Done) and msgs[-1].total_tokens == 85
    br.close()


def test_out_of_order_and_slow_writers(exo_protocol: Any, gated: Any) -> None:
    v, source = gated
    v.order = [2, 0, 3, 1]
    v.gate = lambda i: time.sleep(0.05 * (i + 1))
    toks = list(range(70))
    out, ref = Recorder(), Recorder()
    bridge(v, source, stream_layers=True).resolve(req(toks, "ooo"), out)
    bridge(v, source).resolve(req(toks, "ooo"), ref)
    assert out.getvalue() == ref.getvalue()
    assert [m.layer_idx for m in decode(exo_protocol, out.getvalue())[1][:-1]] == [0, 1, 2, 3]


def test_a_writer_that_stalls_past_the_layer_timeout_fails_without_done(exo_protocol: Any, gated: Any) -> None:
    v, source = gated
    release = threading.Event()
    v.gate = lambda i: release.wait(10) if i == 2 else None
    out = Recorder()
    t0 = time.monotonic()
    with pytest.raises(HandoffError, match=r"budget|vLLM"):
        bridge(v, source, stream_layers=True, timeout_s=1.5).resolve(req(list(range(40)), "stall"), out)
    assert time.monotonic() - t0 < 6
    release.set()
    proto = exo_protocol
    msgs = [proto.read_message(io.BytesIO(len(f).to_bytes(4, "big") + f)) for f in frames(out.getvalue())[1:]]
    assert [m.layer_idx for m in msgs if isinstance(m, proto.KVChunk)] == [0, 1]
    assert isinstance(msgs[-1], proto.ErrorMessage) and not any(isinstance(m, proto.Done) for m in msgs)


@pytest.mark.parametrize(
    "per_1k, start, completes",
    [(50.0, 0, True), (0.0, 0, False), (50.0, 30, False)],
    ids=["scaled", "fixed", "start_pos"],
)
def test_the_pre_header_budget_scales_with_the_tokens_to_prefill(
    exo_protocol: Any, gated: Any, per_1k: float, start: int, completes: bool
) -> None:
    """A prefill slower than timeout_s completes when the budget scaled by the tokens to prefill covers it (1 s + 40
    tokens at 50 s per 1000 = 3 s against 2 s), and fails without it, or when start_pos leaves only 10 tokens (1.5 s)."""
    v, source = gated
    v.gate = lambda i: time.sleep(2.0) if i == 0 else None
    out = Recorder()
    br = bridge(v, source, pad_tokens=0, timeout_s=1, timeout_per_1k_tokens_s=per_1k)
    r = req(list(range(40)), f"budget-{per_1k}-{start}", start=start)
    if completes:
        br.resolve(r, out)
        assert isinstance(decode(exo_protocol, out.getvalue())[1][-1], exo_protocol.Done)
    else:
        with pytest.raises(HandoffError, match="budget"):
            br.resolve(r, out)
        assert out.getvalue() == b""  # failed before EXO's stream was written to
    time.sleep(1.2)  # let the gated forward finish before the store is torn down
    br.close()


@pytest.mark.parametrize("how", ["post", "record"])
def test_abort_mid_prefill_fails_fast_without_done(exo_protocol: Any, gated: Any, how: str) -> None:
    """After two layers either vLLM's request fails (engine died, request aborted: no publish, no failure line, the
    POST answers 500) or the writer fails the export (the stream's failure line). Either way the bridge stops at its
    next poll, ends EXO's stream with an error frame and never sends Done."""
    v, source = gated
    v.fail_after, v.fail_how = 2, how
    v.gate = lambda i: time.sleep(0.1)
    out = Recorder()
    t0 = time.monotonic()
    with pytest.raises(HandoffError, match="500" if how == "post" else "injected write failure"):
        bridge(v, source, stream_layers=True).resolve(req(list(range(50)), "abort"), out)
    assert time.monotonic() - t0 < 3  # far inside the 10 s budget
    msgs = [exo_protocol.read_message(io.BytesIO(len(f).to_bytes(4, "big") + f)) for f in frames(out.getvalue())[1:]]
    assert not any(isinstance(m, exo_protocol.Done) for m in msgs) and isinstance(msgs[-1], exo_protocol.ErrorMessage)
    v.fail_after = None
    bridge(v, source, stream_layers=True).resolve(req(list(range(50)), "again"), Recorder())  # the next one is fine


def test_streaming_bridge_against_a_buffered_writer(exo_protocol: Any, store_root: str) -> None:
    store = ExportStore(store_root, slots=2, hold_seconds=60)
    v = GatedVllm(store, stream=False)
    try:
        out, ref = Recorder(), Recorder()
        bridge(v, {"local_store_path": store_root}, stream_layers=True).resolve(req(list(range(30)), "mix"), out)
        bridge(v, {"local_store_path": store_root}).resolve(req(list(range(30)), "mix"), ref)
        assert out.getvalue() == ref.getvalue()
    finally:
        v.close()


def test_stream_layers_config_is_paged_only() -> None:
    base = {
        "vllm_url": "http://192.0.2.1:8000",
        "vllm_model": "m",
        "transfer_url": "http://192.0.2.1:1",
        "transfer_token_file": "/t",
    }
    with pytest.raises(ConfigError, match="paged mode only"):
        BridgeConfig.from_mapping({**base, "mode": "dsv4", "stream_layers": True})
    assert BridgeConfig.from_mapping({**base, "mode": "paged", "stream_layers": True}).stream_layers
    assert ConnectorConfig.from_extra({"store_path": "/s", "stream_layers": True}).stream_layers
    assert not ConnectorConfig.from_extra({"store_path": "/s"}).stream_layers


# ----------------------------------------------------------------------------------------------- tensor parallel
class ThreadTP:
    """Two TP ranks as two threads: all_gather blocks until both ranks have contributed, like NCCL's."""

    def __init__(self) -> None:
        self.world_size = 2
        self._local = threading.local()
        self._parts: dict[int, torch.Tensor] = {}
        self._barrier = threading.Barrier(2, timeout=10)

    @property
    def rank_in_group(self) -> int:
        return int(self._local.rank)

    def set_rank(self, rank: int) -> None:
        self._local.rank = rank

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        self._parts[self.rank_in_group] = x
        self._barrier.wait()
        out = torch.cat([self._parts[0], self._parts[1]], dim=dim)
        self._barrier.wait()
        return out


@pytest.fixture
def stub() -> Iterator[dict[str, types.ModuleType]]:
    with vllm_stub.installed() as mods:
        yield mods


def test_tp_layer_is_announced_only_once_every_rank_has_contributed(stub: Any, store_root: str) -> None:
    """TP2: rank 0 writes after the all-gather, so a layer is in the stream log only once rank 1 has reached the same
    layer; the announced bytes hold all heads."""
    from kvhandoff.vllm.paged_connector import KVHandoffPagedConnector

    cfg = vllm_cfg({"store_path": store_root, "stream_layers": True})
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    blocks, n = [3, 9, 1], 40
    slots = slot_mapping(blocks, n, BS)
    full = [
        torch.cat([paged_layer_tensor("v030", 32, 2 * i), paged_layer_tensor("v030", 32, 2 * i + 1)], dim=3) for i in range(3)
    ]
    meta = KVHandoffPagedConnector(cfg, role.SCHEDULER, None).build_connector_meta(
        sched_out([new("a", n, blocks, "tp-s")], [], {"a": n})
    )
    tp = ThreadTP()
    vllm_stub.STATE["tp"] = tp
    gates = [threading.Event() for _ in range(3)]
    errs: list[BaseException] = []

    def rank(r: int) -> None:
        try:
            tp.set_rank(r)
            w = KVHandoffPagedConnector(cfg, role.WORKER, None)
            w._connector_metadata = meta
            for i in range(3):
                if r == 1:
                    gates[i].wait(10)
                w.save_kv_layer(f"model.layers.{i}.self_attn.attn", full[i][:, :, :, r * H : (r + 1) * H].contiguous(), None)
            w.wait_for_save()
        except BaseException as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=rank, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    reader = ExportReader(store_root)

    def announced() -> list[int]:
        try:
            return [x.index for x in reader.stream_state("tp-s").header.layers]
        except (ExportNotFound, FileNotFoundError):
            return []

    for i in range(3):
        time.sleep(0.2)
        assert announced() == list(range(i))  # rank 0 is waiting in the all-gather for rank 1's layer i
        gates[i].set()
        assert wait_for(lambda i=i: announced()[: i + 1] == list(range(i + 1)))
        st = reader.stream_state("tp-s")
        with reader.open_stream("tp-s", st.header.seq) as ex:
            want = to_kv_split(gather(full[i], slots, BS), D)
            assert ex.header.layer(i).blobs[0].shape == (n, 2 * H, D)
            assert ex.read_blob(i, "k") == want[0].view(torch.int16).numpy().tobytes()
            assert ex.read_blob(i, "v") == want[1].view(torch.int16).numpy().tobytes()
    for t in threads:
        t.join(10)
    assert not errs
    assert reader.stream_state("tp-s").published


def test_dsv4_connector_refuses_stream_layers(stub: Any, store_root: str) -> None:
    from kvhandoff.vllm.dsv4_connector import KVHandoffDSV4Connector

    cfg = vllm_cfg({"store_path": store_root, "stream_layers": True}, hf={"compress_ratios": [0]})
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    with pytest.raises(ConfigError, match="KVHandoffPagedConnector only"):
        KVHandoffDSV4Connector(cfg, role.SCHEDULER, None)
