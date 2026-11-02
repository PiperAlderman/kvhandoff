"""DeepSeek V4: the packaged converter must emit exactly the prototype's bytes (the format exo-explore/exo#2545 ingests and the
hardware hand-offs used), plus the exporter and bookkeeping that moved out of the connector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from fakes import dsv4_layout as fk
from kvhandoff.interchange import KIND_DSV4
from kvhandoff.producer import dsv4_convert as cv
from kvhandoff.producer.dsv4_export import (
    DSV4Exporter,
    DSV4Job,
    DSV4ModelInfo,
    FinishedTracker,
    SchedulerQueue,
    index_cache_names,
    rope_config_from_checkpoint,
)
from kvhandoff.store import ExportFailed, ExportReader, ExportStore

GOLDEN = json.loads((Path(__file__).parent / "data" / "dsv4_golden.json").read_text())


def _flat(blobs: list[cv.Blob]) -> list[tuple[str, tuple[int, ...], bytes]]:
    return [(d, s, cv.blob_bytes(t).numpy().tobytes()) for _, d, s, t in blobs]


def test_converter_is_byte_identical_to_the_prototype() -> None:
    rng = fk.Rng(GOLDEN["seed"])
    cs, inv = fk.rope_tables(1100)
    for case in GOLDEN["cases"]:
        t, ratio = case["T"], case["ratio"]
        blobs = cv.convert_layer(fk.make_layer(rng, cv.LayerSources, ratio, t, cs, inv), t)
        got = [[d, list(s), hashlib.sha256(b).hexdigest()] for d, s, b in _flat(blobs)]
        assert got == case["blobs"], f"T={t} ratio={ratio}"


@pytest.mark.parametrize("t,ratio", [(1, 0), (129, 4), (257, 128), (515, 4)])
def test_header_roles_and_window(t: int, ratio: int) -> None:
    cs, inv = fk.rope_tables(600)
    blobs = cv.convert_layer(fk.make_layer(fk.Rng(7), cv.LayerSources, ratio, t, cs, inv), t)
    roles = [b[0] for b in blobs]
    assert roles[:3] == ["state_header", "window_k", "window_v"]
    assert blobs[0][3].tolist() == fk.expected_header(ratio, t)
    assert torch.equal(blobs[1][3].view(torch.int16), blobs[2][3].view(torch.int16))  # K == V, shared MQA head
    flags = blobs[0][3].tolist()
    present = [
        r
        for r, f in zip([f"{k}.{a}" for k in cv.BRANCH_KEYS for a in cv.BRANCH_FIELDS], flags[6:11] + flags[14:19], strict=True)
        if f == 1.0
    ]
    assert roles[3:] == present  # one blob per presence flag, in wire order


def test_convert_layer_reports_missing_sources() -> None:
    src = cv.LayerSources(4)
    with pytest.raises(ValueError, match="swa"):
        cv.convert_layer(src, 10)


def test_scheduler_queue_and_finished_tracker() -> None:
    q = SchedulerQueue()
    assert q.on_finished("r0", 100, {"kvhandoff_tag": "t0"}, ([1, 2], [3]))
    assert q.on_finished("r1", 50, {"kvhandoff_tag": "t1"}, ([4],))
    assert not q.on_finished("plain", 50, {}, ([5],))
    assert not q.on_finished("empty", 0, {"kvhandoff_tag": "t2"}, ([6],))
    jobs = q.take()
    assert [(j.req_id, j.tag, j.block_ids) for j in jobs] == [("r0", "t0", [[1, 2], [3]]), ("r1", "t1", [[4]])]
    assert q.take() == []
    assert SchedulerQueue(export_all=True).on_finished("x", 3, None, ([1],))

    tr = [FinishedTracker(cap=8) for _ in range(4)]  # TP4: every rank reports
    outs = [w.step({"r0", "plain"}, ["r0", "r1"]) for w in tr]
    assert all(o == {"r0"} for o in outs)  # r1's finished id not seen yet
    assert all(w.step({"r1"}, []) == {"r1"} for w in tr)
    w = tr[0]
    w.step({f"x{i}" for i in range(100)}, [])
    assert w.seen_size <= 8 and not w.pending


def test_index_cache_names() -> None:
    names = [
        "model.layers.0.attn.swa_cache",
        "model.layers.0.attn",
        "model.layers.0.attn.compressor.state_cache",
        "model.layers.1.attn.indexer.k_cache",
        "model.layers.1.attn.indexer.compressor.state_cache",
        "model.layers.2.mlp.something",
    ]
    assert index_cache_names(names) == {
        0: {"swa": names[0], "pool": names[1], "state": names[2]},
        1: {"idx_pool": names[3], "idx_state": names[4]},
    }


def test_rope_config_reads_the_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"rope_theta": 5.0, "compress_rope_theta": None}))
    cfg = rope_config_from_checkpoint(str(tmp_path))
    assert cfg["rope_theta"] == 5.0 and cfg["compress_rope_theta"] == 160000.0
    assert rope_config_from_checkpoint(str(tmp_path / "missing"))["rope_theta"] == 10000.0


class FakeAttn:
    def __init__(self, src: Any):
        def ns(**kw: Any) -> Any:
            return type("NS", (), kw)()

        self.rotary_emb = ns(cos_sin_cache=src.cos_sin)
        self.compressor = ns(ape=ns(data=src.ape))
        self.indexer = ns(compressor=ns(ape=ns(data=src.idx_ape)))


def fake_model(
    t: int, ratios: list[int], seed: int = 3
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, Any], list[list[int]], list[Any]]:
    """vLLM's registered caches, kv-cache groups, attention modules and per-group block ids for one request.

    Every cache gets its own group here, so the job's block ids for that group are exactly that cache's block table."""
    cs, inv = fk.rope_tables(t + 8)
    rng = fk.Rng(seed)
    kv: dict[str, torch.Tensor] = {}
    groups: dict[str, int] = {}
    mods: dict[str, Any] = {}
    bids: list[list[int]] = []
    srcs: list[Any] = []
    for i, r in enumerate(ratios):
        src = fk.make_layer(rng, cv.LayerSources, r, t, cs, inv)
        srcs.append(src)
        base = f"model.layers.{i}.attn"
        mods[base] = FakeAttn(src)
        parts = (
            [("swa", ".swa_cache")]
            + ([("pool", ""), ("state", ".compressor.state_cache")] if r else [])
            + ([("idx_pool", ".indexer.k_cache"), ("idx_state", ".indexer.compressor.state_cache")] if r == 4 else [])
        )
        for attr, suffix in parts:
            kv[base + suffix] = getattr(src, attr)
            groups[base + suffix] = len(bids)
            bids.append(list(getattr(src, attr + "_blocks")))
    return kv, groups, mods, bids, srcs


def test_exporter_writes_exactly_the_converted_blobs(store_root: str) -> None:
    t, ratios = 300, [0, 4, 128, 4]
    kv, groups, mods, bids, srcs = fake_model(t, ratios)
    store = ExportStore(store_root, slots=2)
    info = DSV4ModelInfo(
        model_id="deepseek-v4-flash",
        num_layers=len(ratios),
        compress_ratios=tuple(ratios),
        rope_cfg=dict(rope_config_from_checkpoint("/nonexistent")),
        producer={"connector": "test"},
    )
    exp = DSV4Exporter(store, info)
    h = exp.export(DSV4Job(req_id="req-1", tag="d-1", tokens=t, block_ids=bids), kv, groups, mods)
    assert h is not None and h.kind == KIND_DSV4 and h.num_tokens == t and h.num_model_layers == 4
    assert h.rope is not None and h.rope.details["ratio0_layers"] == "mlx"
    assert set(h.extra["rope_check_max_abs"]) == {"0", "1", "2"}
    reader = ExportReader(store_root)
    with reader.open("d-1", seq=h.seq) as ex:
        for i, layer in enumerate(h.layers):
            assert layer.attrs == {"compress_ratio": ratios[i]} and layer.name == f"model.layers.{i}.attn"
            want = _flat(cv.convert_layer(srcs[i], t))
            got = [(b.dtype, b.shape, ex.read_range(i, b.offset, b.nbytes)) for b in layer.blobs]
            assert got == want


def test_exporter_failure_becomes_a_failure_record(store_root: str) -> None:
    t, ratios = 40, [0, 4]
    kv, groups, mods, bids, _ = fake_model(t, ratios)
    del kv["model.layers.1.attn.indexer.k_cache"]
    store = ExportStore(store_root, slots=1)
    info = DSV4ModelInfo(
        model_id="m",
        num_layers=2,
        compress_ratios=(0, 4),
        rope_cfg={"rope_theta": 10000.0, "compress_rope_theta": 160000.0, "rope_scaling": fk.YARN},
    )
    exp = DSV4Exporter(store, info)
    assert exp.export(DSV4Job(req_id="r", tag="bad-1", tokens=t, block_ids=bids), kv, groups, mods) is None
    with pytest.raises(ExportFailed, match="missing idx_pool"):
        ExportReader(store_root).lookup("bad-1")
    # the slot was given back: the next export succeeds in the single slot
    kv2, groups2, mods2, bids2, _ = fake_model(t, [0, 4])
    assert exp.export(DSV4Job(req_id="r2", tag="ok-1", tokens=t, block_ids=bids2), kv2, groups2, mods2) is not None


def test_exporter_store_full_becomes_a_failure_record(store_root: str) -> None:
    t, ratios = 20, [0]
    kv, groups, mods, bids, _ = fake_model(t, ratios)
    store = ExportStore(store_root, slots=1, hold_seconds=3600)
    info = DSV4ModelInfo(
        model_id="m",
        num_layers=1,
        compress_ratios=(0,),
        rope_cfg={"rope_theta": 10000.0, "compress_rope_theta": 160000.0, "rope_scaling": fk.YARN},
    )
    exp = DSV4Exporter(store, info)
    assert exp.export(DSV4Job(req_id="a", tag="a-1", tokens=t, block_ids=bids), kv, groups, mods) is not None
    assert exp.export(DSV4Job(req_id="b", tag="b-1", tokens=t, block_ids=bids), kv, groups, mods) is None
    with pytest.raises(ExportFailed, match="StoreFull"):
        ExportReader(store_root).lookup("b-1")
