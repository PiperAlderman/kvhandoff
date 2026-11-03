"""The DeepSeek V4 short-prompt hand-off overhead fixes (3/11/2026):

(a) slot files pre-sized, allocated and pre-faulted at start-up (``presize_tokens``), small files rounded up to a power
    of two, and the DeepSeek V4 exporter honouring ``prefault_slot_maps``;
(b) the parsed export header cached per slot in ExportReader, so 43 layer requests parse it once;
(c) the header wait polling every few milliseconds right after vLLM's prefill response.
"""

from __future__ import annotations

import os
import shutil
import threading
import types
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

from fakes import dsv4_layout as fk
from fakes import vllm_stub
from kvhandoff.config import ConfigError, ConnectorConfig
from kvhandoff.interchange import KIND_DSV4, ExportHeader
from kvhandoff.producer import dsv4_convert as cv
from kvhandoff.producer.dsv4_export import DSV4Exporter, DSV4Job, DSV4ModelInfo, rope_config_from_checkpoint
from kvhandoff.producer.maps import round_capacity
from kvhandoff.store import ExportNotFound, ExportReader, ExportStore
from kvhandoff.transfer import client as client_mod
from kvhandoff.transfer.client import Deadline, LocalSource, TransferClient
from test_dsv4 import fake_model
from test_store import put

MIB = 1 << 20
RATIOS = [0, 4, 128, 4]


def info(ratios: list[int]) -> DSV4ModelInfo:
    return DSV4ModelInfo(
        model_id="deepseek-v4-flash",
        num_layers=len(ratios),
        compress_ratios=tuple(ratios),
        rope_cfg=dict(rope_config_from_checkpoint("/nonexistent")),
        producer={"connector": "test"},
    )


# ----------------------------------------------------------------------------------------------- (a) sizes
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_layer_nbytes_matches_convert_layer(ratio: int) -> None:
    cs, inv = fk.rope_tables(700)
    rng = fk.Rng(11)
    for t in (1, 2, 3, 4, 5, 7, 8, 9, 127, 128, 129, 130, 255, 256, 257, 300, 515, 517, 640):
        blobs = cv.convert_layer(fk.make_layer(rng, cv.LayerSources, ratio, t, cs, inv), t)
        assert cv.layer_nbytes(t, ratio) == sum(int(cv.blob_bytes(b[3]).numel()) for b in blobs), f"T={t}"


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_layer_capacity_covers_every_shorter_prompt(ratio: int) -> None:
    for t in (1, 3, 100, 128, 129, 300, 700):
        assert cv.layer_capacity(t, ratio) == max(cv.layer_nbytes(x, ratio) for x in range(1, t + 1))


def test_v4_flash_capacity_at_32k() -> None:
    """The figure the config comment quotes: about 245 MB per slot for V4 Flash (43 layers) at 32768 tokens."""
    flash = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)]
    per_slot = sum(cv.layer_capacity(32768, r) for r in flash)
    assert 240e6 < per_slot < 250e6


def test_round_capacity_rounds_small_files_to_a_power_of_two() -> None:
    assert round_capacity(1) == MIB
    assert round_capacity(MIB + 1) == 2 * MIB
    assert round_capacity(3_000_000) == 4 * MIB
    assert round_capacity(9 * MIB) == 16 * MIB
    assert round_capacity(64 * MIB) == 64 * MIB
    assert round_capacity(64 * MIB + 1) == 128 * MIB  # above 64 MiB: 64 MiB multiples, as before
    assert round_capacity(130 * MIB) == 192 * MIB


# ----------------------------------------------------------------------------------------------- (a) exporter
def _export(exp: DSV4Exporter, t: int, tag: str, seed: int = 3) -> ExportHeader:
    kv, groups, mods, bids, _ = fake_model(t, RATIOS, seed=seed)
    h = exp.export(DSV4Job(req_id=f"r-{tag}", tag=tag, tokens=t, block_ids=bids), kv, groups, mods)
    assert h is not None
    return h


def test_exporter_prefaults_what_it_writes(store_root: str) -> None:
    on = DSV4Exporter(ExportStore(store_root, slots=1, hold_seconds=0), info(RATIOS))
    assert on.prefault_slot_maps
    _export(on, 200, "d-on")
    assert on.maps.prefault_calls == len(RATIOS)
    off = DSV4Exporter(ExportStore(store_root + "-off", slots=1, hold_seconds=0), info(RATIOS), prefault_slot_maps=False)
    _export(off, 200, "d-off")
    assert off.maps.prefault_calls == 0


def test_presize_sizes_allocates_and_prefaults_every_slot(store_root: str) -> None:
    store = ExportStore(store_root, slots=2, hold_seconds=0)
    exp = DSV4Exporter(store, info(RATIOS))
    caps = exp.layer_capacities(300)
    out = exp.presize(300)
    assert out["done"] and out["bytes"] == 2 * sum(caps)
    maps_before = {}
    for k in range(2):
        for i, cap in enumerate(caps):
            st = os.stat(os.path.join(store.slot_dir(k), f"L{i}.bin"))
            assert st.st_size == cap and st.st_blocks * 512 >= cap  # sized and allocated
            m = exp.maps._maps[(k, f"L{i}.bin")]
            assert m.tensor.numel() == cap and m.populated.get(0) == cap  # mapped and pre-faulted
            maps_before[(k, i)] = m.tensor
    calls = exp.maps.prefault_calls
    h = _export(exp, 300, "d-1")  # up to the presized length: same maps, nothing left to populate
    assert exp.maps.prefault_calls == calls
    assert all(exp.maps._maps[(h.slot, f"L{i}.bin")].tensor is maps_before[(h.slot, i)] for i in range(len(RATIOS)))
    _, _, _, _, srcs = fake_model(300, RATIOS)
    with ExportReader(store_root).open("d-1", seq=h.seq) as ex:
        for i, layer in enumerate(h.layers):
            want = [cv.blob_bytes(b[3]).numpy().tobytes() for b in cv.convert_layer(srcs[i], 300)]
            assert [ex.read_range(i, b.offset, b.nbytes) for b in layer.blobs] == want


def test_presize_without_prefault_still_sizes_and_allocates(store_root: str) -> None:
    store = ExportStore(store_root, slots=1)
    exp = DSV4Exporter(store, info(RATIOS), prefault_slot_maps=False)
    assert exp.presize(200)["done"]
    assert exp.maps.prefault_calls == 0
    for i, cap in enumerate(exp.layer_capacities(200)):
        assert os.stat(os.path.join(store.slot_dir(0), f"L{i}.bin")).st_blocks * 512 >= cap


def test_presize_skips_when_the_store_has_too_little_room(store_root: str) -> None:
    store = ExportStore(store_root, slots=2)
    exp = DSV4Exporter(store, info(RATIOS))
    out = exp.presize(300, max_free_share=1e-12)
    assert not out["done"] and out["bytes"] == 2 * sum(exp.layer_capacities(300))
    assert not os.path.exists(os.path.join(store.slot_dir(0), "L0.bin"))
    assert exp.presize(0) == {"tokens": 0, "slots": 2, "done": False}


def test_presize_counts_pages_an_earlier_process_allocated(store_root: str) -> None:
    exp = DSV4Exporter(ExportStore(store_root, slots=1), info(RATIOS))
    assert exp.presize(300)["done"]
    exp.store.close()
    again = DSV4Exporter(ExportStore(store_root, slots=1), info(RATIOS))
    out = again.presize(300, max_free_share=1e-12)  # nothing left to allocate, so even a tiny share is enough
    assert out["done"] and out["allocated_before"] == out["bytes"]


def test_slot_recreated_after_presize_is_remapped_and_prefaulted_again(store_root: str) -> None:
    store = ExportStore(store_root, slots=1, hold_seconds=0)
    exp = DSV4Exporter(store, info(RATIOS))
    assert exp.presize(300)["done"]
    old = exp.maps._maps[(0, "L1.bin")].tensor
    shutil.rmtree(store.slot_dir(0))  # RemoveIPC or an operator's rm: the store bumps the slot's generation
    calls = exp.maps.prefault_calls
    h = _export(exp, 200, "d-after")
    assert exp.maps._maps[(0, "L1.bin")].tensor is not old
    assert exp.maps.prefault_calls == calls + len(RATIOS)
    assert ExportReader(store_root).lookup("d-after").seq == h.seq


# ----------------------------------------------------------------------------------------------- (a) connector
@pytest.fixture
def stub() -> Iterator[dict[str, types.ModuleType]]:
    with vllm_stub.installed() as mods:
        yield mods


def _vllm_cfg(extra: dict[str, Any], max_model_len: int = 4096) -> Any:
    ns = types.SimpleNamespace
    return ns(
        kv_transfer_config=ns(kv_connector_extra_config=extra),
        cache_config=ns(block_size=64),
        model_config=ns(
            max_model_len=max_model_len,
            served_model_name=["dsv4"],
            model="/models/x",
            hf_config=ns(num_hidden_layers=len(RATIOS), compress_ratios=RATIOS),
        ),
        scheduler_config=ns(max_num_seqs=4),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_connector_presizes_at_register_on_rank_0(stub: Any, store_root: str, rank: int) -> None:
    from kvhandoff.vllm.dsv4_connector import KVHandoffDSV4Connector

    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    vllm_stub.STATE["tp_rank"] = rank
    w = KVHandoffDSV4Connector(_vllm_cfg({"store_path": store_root, "slots": 2}, max_model_len=300), role.WORKER)
    w.register_kv_caches({})
    if rank:
        assert w._exporter is None
        return
    assert w._exporter is not None  # created at start-up, before any request
    caps = w._exporter.layer_capacities(300)  # capped at max_model_len (300), not the 32768 default
    for k in range(2):
        for i, cap in enumerate(caps):
            assert os.stat(os.path.join(w._exporter.store.slot_dir(k), f"L{i}.bin")).st_size == cap


def test_connector_presize_off(stub: Any, store_root: str) -> None:
    from kvhandoff.vllm.dsv4_connector import KVHandoffDSV4Connector

    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    vllm_stub.STATE["tp_rank"] = 0
    w = KVHandoffDSV4Connector(_vllm_cfg({"store_path": store_root, "presize_tokens": 0}), role.WORKER)
    w.register_kv_caches({})
    assert w._exporter is None


def test_presize_tokens_config() -> None:
    assert ConnectorConfig.from_extra({"store_path": "/s"}).presize_tokens == 32768
    assert ConnectorConfig.from_extra({"store_path": "/s", "presize_tokens": 0}).presize_tokens == 0
    assert ConnectorConfig.from_extra({"store_path": "/s", "presize_tokens": 8192}).presize_tokens == 8192
    for bad in (-1, "8k", True, 1.5):
        with pytest.raises(ConfigError, match="presize_tokens"):
            ConnectorConfig.from_extra({"store_path": "/s", "presize_tokens": bad})


# ----------------------------------------------------------------------------------------------- (b) header cache
@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts ExportHeader.from_json calls (every header parse, cached or not)."""
    count = [0]
    orig = ExportHeader.from_json

    def counting(cls: type[ExportHeader], raw: bytes | str) -> ExportHeader:
        count[0] += 1
        return orig(raw)

    monkeypatch.setattr(ExportHeader, "from_json", classmethod(counting))
    return count


def test_layer_opens_parse_the_header_once(store_root: str, parses: list[int]) -> None:
    h = put(ExportStore(store_root, slots=2), "a-1")
    r = ExportReader(store_root)
    parses[0] = 0
    for _ in range(43):
        with r.open("a-1", seq=h.seq) as ex:
            assert ex.header == h
    assert parses[0] == 1


def test_layer_opens_on_threads_share_the_cache(store_root: str, parses: list[int]) -> None:
    h = put(ExportStore(store_root, slots=2), "a-1")
    r = ExportReader(store_root)
    r.lookup("a-1")
    parses[0] = 0
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                with r.open("a-1", seq=h.seq) as ex:
                    assert ex.header.seq == h.seq
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and parses[0] == 0


def test_reused_slot_is_never_served_from_the_cache(store_root: str) -> None:
    store = ExportStore(store_root, slots=1, hold_seconds=0)
    a = put(store, "a-1")
    r = ExportReader(store_root)
    with r.open("a-1", seq=a.seq):
        pass
    b = put(store, "b-1")  # the only slot, reused for another export
    assert b.slot == a.slot
    with pytest.raises(ExportNotFound):
        r.open("a-1", seq=a.seq)
    with pytest.raises(ExportNotFound):
        r.open("b-1", seq=a.seq)
    with r.open("b-1", seq=b.seq) as ex:
        assert ex.header == b


def test_republished_tag_is_seen_at_its_new_seq(store_root: str) -> None:
    store = ExportStore(store_root, slots=1, hold_seconds=0, clock=lambda: 1000.0)  # same created_unix, same length
    first = put(store, "a-1")
    r = ExportReader(store_root)
    assert r.lookup("a-1").seq == first.seq
    second = put(store, "a-1")  # same tag, same slot, same header length
    assert second.seq != first.seq and len(second.to_json()) == len(first.to_json())
    assert r.lookup("a-1").seq == second.seq
    with pytest.raises(ExportNotFound):
        r.open("a-1", seq=first.seq)
    with r.open("a-1", seq=second.seq) as ex:
        assert ex.header == second


def test_header_cache_is_bounded(store_root: str) -> None:
    store = ExportStore(store_root, slots=4)
    r = ExportReader(store_root)
    r.headers.max_entries = 2
    for i in range(4):
        put(store, f"a-{i}")
        r.lookup(f"a-{i}")
    assert len(r.headers._entries) == 2


# ----------------------------------------------------------------------------------------------- (c) header poll
class VirtualTime:
    """Stands in for the client module's ``time``: sleep advances a virtual clock."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


@pytest.mark.parametrize("publish_at", [0.03, 0.18, 0.5, 1.9])
def test_local_header_wait_returns_within_a_few_ms_of_the_publish(
    store_root: str, monkeypatch: pytest.MonkeyPatch, publish_at: float
) -> None:
    vt = VirtualTime()
    monkeypatch.setattr(client_mod, "time", vt)
    h = put(ExportStore(store_root, slots=1), "a-1")
    src = LocalSource(store_root)
    real = src.reader.lookup

    def lookup(tag: str) -> ExportHeader:
        if vt.now < publish_at:
            raise ExportNotFound(tag)
        return real(tag)

    monkeypatch.setattr(src.reader, "lookup", lookup)
    assert src.wait_header("a-1", Deadline(30.0)) == h
    assert vt.now - publish_at <= 0.005
    assert max(vt.sleeps) <= 0.005


def test_header_poll_backs_off_only_after_a_couple_of_seconds() -> None:
    sched = client_mod.header_poll_sleep
    assert all(sched(w / 100) <= 0.005 for w in range(200))
    assert 0.005 < sched(3.0) <= 0.05
    assert sched(60.0) <= 0.1


def test_http_header_wait_polls_fast_on_404_and_backs_off_on_503(monkeypatch: pytest.MonkeyPatch) -> None:
    vt = VirtualTime()
    monkeypatch.setattr(client_mod, "time", vt)
    h = ExportHeader(kind=KIND_DSV4, tag="a-1", model_id="m", num_tokens=4, layers=(), num_model_layers=0, seq=1)
    answers = [404, 404, 503, 503, 404, 200]

    def request(self: TransferClient, method: str, path: str, timeout: float, headers: Any = None) -> Any:
        code = answers.pop(0)
        return code, (h.to_json() if code == 200 else b""), None

    monkeypatch.setattr(TransferClient, "_request", request)
    got = TransferClient("http://127.0.0.1:1", "t" * 48).wait_header("a-1", Deadline(30.0))
    assert got.tag == "a-1"
    fast = [vt.sleeps[0], vt.sleeps[1], vt.sleeps[4]]  # after each 404
    busy = vt.sleeps[2:4]  # after each 503
    assert all(s <= 0.005 for s in fast)
    assert all(s >= 0.05 for s in busy)
