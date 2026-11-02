"""Paged-KV export: the scheduler-side tracker under chunked prefill, the gather/normalise logic against fake vLLM
layouts (including a simulated tensor-parallel head split), and the slot writer."""

from __future__ import annotations

import types
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from kvhandoff.interchange import RopeInfo
from kvhandoff.producer.paged import (
    PagedExportInfo,
    PagedWriter,
    RequestTracker,
    StoreJob,
    UnsupportedLayout,
    gather,
    slot_mapping,
    to_kv_split,
)
from kvhandoff.store import ExportFailed, ExportReader, ExportStore

BS, H, D = 16, 2, 8


def sp(tag: str | None, key: str = "kvhandoff_tag") -> Any:
    return types.SimpleNamespace(extra_args={"kv_transfer_params": {key: tag}} if tag else {})


def new(rid: str, n: int, blocks: list[int], tag: str | None, computed: int = 0, key: str = "kvhandoff_tag") -> Any:
    return types.SimpleNamespace(
        req_id=rid,
        prompt_token_ids=list(range(n)),
        sampling_params=sp(tag, key),
        block_ids=(blocks,),
        num_computed_tokens=computed,
    )


def sched_out(
    new_reqs: list[Any], cached: list[tuple[str, Any, int]], tokens: dict[str, int], resumed: set[str] | None = None
) -> Any:
    return types.SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        num_scheduled_tokens=tokens,
        scheduled_cached_reqs=types.SimpleNamespace(
            req_ids=[c[0] for c in cached],
            resumed_req_ids=resumed or set(),
            new_block_ids=[c[1] for c in cached],
            num_computed_tokens=[c[2] for c in cached],
        ),
    )


A, B, C = list(range(1, 20)), list(range(20, 30)), list(range(30, 40))


def test_store_is_emitted_once_in_the_step_that_completes_the_prompt() -> None:
    """Port of the prototype's concurrency check: two tagged prompts under chunked prefill
    plus an untagged one."""
    tr = RequestTracker(BS)
    s1 = tr.step(sched_out([new("a", 300, A[:8], "tagA"), new("c", 50, C[:4], None)], [], {"a": 128, "c": 50}))
    s2 = tr.step(sched_out([new("b", 100, B[:7], "tagB")], [("a", (A[8:16],), 128)], {"a": 128, "b": 100}))
    s3 = tr.step(sched_out([], [("a", (A[16:19],), 256)], {"a": 44}))
    assert s1 == []
    assert [(j.tag, j.num_tokens, j.block_ids) for j in s2] == [("tagB", 100, B[:7])]
    assert [(j.tag, j.num_tokens, j.block_ids) for j in s3] == [("tagA", 300, A[:19])]
    assert len(tr) == 0


def test_resumed_request_replaces_its_blocks_and_aborts_are_dropped() -> None:
    tr = RequestTracker(BS)
    tr.step(sched_out([new("a", 64, [1, 2], "t")], [], {"a": 32}))
    out = tr.step(sched_out([], [("a", ([7, 8, 9, 10],), 0)], {"a": 64}, resumed={"a"}))  # preempted, re-run
    assert [j.block_ids for j in out] == [[7, 8, 9, 10]]
    tr.step(sched_out([new("x", 64, [1], "t2")], [], {"x": 16}))
    tr.drop("x")
    assert len(tr) == 0 and tr.step(sched_out([], [("x", ([2],), 16)], {"x": 48})) == []


def test_export_all_and_short_block_lists() -> None:
    tr = RequestTracker(BS, export_all=True)
    out = tr.step(sched_out([new("cmpl/1", 20, [3, 4], None)], [], {"cmpl/1": 20}))
    assert [j.tag for j in out] == ["r-cmpl_1"]
    assert RequestTracker(BS).step(sched_out([new("a", 40, [1], "t")], [], {"a": 40})) == []  # 1 block for 40 tokens


def paged_layer_tensor(layout: str, nb: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if layout == "v030":  # vLLM v0.30.0 FlashInfer: [blocks, 2, block_size, H, D]
        return torch.randn(nb, 2, BS, H, D, generator=g).to(torch.bfloat16)
    return torch.randn(nb, H, BS, 2 * D, generator=g).to(torch.bfloat16)  # main's view, K|V packed on the last axis


def reference(layer: torch.Tensor, layout: str, blocks: list[int], n: int) -> torch.Tensor:
    pos = torch.arange(n)
    bt, off = torch.tensor(blocks)[pos // BS], pos % BS
    if layout == "v030":
        rows = layer[bt, :, off]  # [n, 2, H, D]
        return torch.stack((rows[:, 0], rows[:, 1]))
    rows = layer[bt, :, off]  # [n, H, 2D]
    return torch.stack((rows[..., :D], rows[..., D:]))


@pytest.mark.parametrize("layout", ["v030", "main"])
def test_gather_and_normalise_match_a_reference(layout: str) -> None:
    layer = paged_layer_tensor(layout, 64, 1)
    blocks = [5, 63, 2, 17, 40]
    n = 70
    kv = to_kv_split(gather(layer, slot_mapping(blocks, n, BS), BS), D)
    assert kv.shape == (2, n, H, D) and kv.is_contiguous()
    assert torch.equal(kv, reference(layer, layout, blocks, n))


def test_tensor_parallel_head_split_reassembles() -> None:
    """Each rank holds a contiguous block of KV heads; normalise per rank, concatenate on the head axis (what
    tp.all_gather(dim=2) does) and the result equals the unsplit export."""
    layer = paged_layer_tensor("v030", 32, 2)
    full = torch.cat([layer, layer], dim=3)  # 4 heads
    blocks, n = [3, 9, 1], 40
    slots = slot_mapping(blocks, n, BS)
    whole = to_kv_split(gather(full, slots, BS), D)
    shards = [to_kv_split(gather(full[:, :, :, r * 2 : (r + 1) * 2], slots, BS), D) for r in range(2)]
    assert torch.equal(torch.cat(shards, dim=2), whole)


def test_unsupported_layouts_are_refused() -> None:
    with pytest.raises(UnsupportedLayout, match="MLA"):
        to_kv_split(torch.zeros(10, 576), 128)  # MLA latent rows
    with pytest.raises(UnsupportedLayout):
        to_kv_split(torch.zeros(10, 4, 3 * D), D)


def info(max_tokens: int = 512) -> PagedExportInfo:
    return PagedExportInfo(
        model_id="llama",
        num_model_layers=3,
        max_tokens=max_tokens,
        rope=RopeInfo(style="neox", applied_by="vllm", theta=500000.0),
        producer={"connector": "test"},
    )


def test_writer_publishes_k_and_v_readable_without_conversion(store_root: str) -> None:
    store = ExportStore(store_root, slots=3)
    w = PagedWriter(store, info())
    layers = [paged_layer_tensor("v030", 64, s) for s in range(3)]
    job = StoreJob(req_id="r1", tag="p-1", num_tokens=100, block_ids=A[:7])
    for i, layer in enumerate(layers):
        w.write_layer(
            job, i, f"model.layers.{i}.self_attn.attn", to_kv_split(gather(layer, slot_mapping(job.block_ids, 100, BS), BS), D)
        )
    (h,) = w.finish_step()
    assert [lr.index for lr in h.layers] == [0, 1, 2] and h.num_tokens == 100
    with ExportReader(store_root).open("p-1", seq=h.seq) as ex:
        for i, layer in enumerate(layers):
            ref = reference(layer, "v030", job.block_ids, 100)
            k = torch.frombuffer(bytearray(ex.read_blob(i, "k")), dtype=torch.bfloat16).view(100, H, D)
            v = torch.frombuffer(bytearray(ex.read_blob(i, "v")), dtype=torch.bfloat16).view(100, H, D)
            assert torch.equal(k, ref[0]) and torch.equal(v, ref[1])


def test_writer_reuses_maps_and_remaps_after_deletion(store_root: str) -> None:
    import shutil

    store = ExportStore(store_root, slots=1, hold_seconds=0)
    w = PagedWriter(store, info(64))
    kv = torch.ones(2, 10, H, D, dtype=torch.float16)
    for rnd in range(3):
        job = StoreJob(req_id=f"r{rnd}", tag=f"p-{rnd}", num_tokens=10, block_ids=[1])
        w.write_layer(job, 0, "model.layers.0.attn", kv * rnd)
        (h,) = w.finish_step()
        if rnd == 1:
            shutil.rmtree(store_root)  # RemoveIPC wipes the store between two exports
        else:
            with ExportReader(store_root).open(h.tag) as ex:
                assert ex.read_blob(0, "k") == (kv[0] * rnd).numpy().tobytes()
    assert len(w.maps._maps) == 1  # pyright: ignore[reportPrivateUsage]


def test_writer_failures_become_failure_records(store_root: str) -> None:
    store = ExportStore(store_root, slots=1, hold_seconds=3600)
    w = PagedWriter(store, info(32))
    ok = StoreJob(req_id="r0", tag="p-ok", num_tokens=4, block_ids=[1])
    full = StoreJob(req_id="r1", tag="p-full", num_tokens=4, block_ids=[1])
    w.write_layer(ok, 0, "l0", torch.zeros(2, 4, H, D, dtype=torch.float16))
    w.write_layer(full, 0, "l0", torch.zeros(2, 4, H, D, dtype=torch.float16))  # no slot left: StoreFull
    assert [h.tag for h in w.finish_step()] == ["p-ok"]
    r = ExportReader(store_root)
    with pytest.raises(ExportFailed, match="StoreFull"):
        r.lookup("p-full")
    r.release("p-ok", r.lookup("p-ok").seq)
    for tag, kv, n, match in (
        ("p-long", torch.zeros(2, 40, H, D, dtype=torch.float16), 40, "exceed"),
        ("p-f64", torch.zeros(2, 4, H, D, dtype=torch.float64), 4, "out_dtype"),
        ("p-rows", torch.zeros(2, 3, H, D, dtype=torch.float16), 4, "rows gathered"),
    ):
        w.write_layer(StoreJob(req_id="x", tag=tag, num_tokens=n, block_ids=[1]), 0, "l0", kv)
        assert w.finish_step() == []
        with pytest.raises(ExportFailed, match=match):
            r.lookup(tag)
    w.mark_failed(StoreJob(req_id="y", tag="p-mla", num_tokens=4, block_ids=[1]), "UnsupportedLayout: MLA")
    assert w.finish_step() == []
    with pytest.raises(ExportFailed, match="MLA"):
        r.lookup("p-mla")


def test_full_store_filesystem_is_a_failure_not_sigbus(store_root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno
    import os

    store = ExportStore(store_root, slots=2)
    w = PagedWriter(store, info(32))

    def no_space(fd: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "posix_fallocate", no_space)
    w.write_layer(
        StoreJob(req_id="r", tag="p-nospace", num_tokens=4, block_ids=[1]), 0, "l0", torch.ones(2, 4, H, D, dtype=torch.float16)
    )
    assert w.finish_step() == []
    with pytest.raises(ExportFailed, match="filesystem full"):
        ExportReader(store_root).lookup("p-nospace")
    monkeypatch.undo()
    w.write_layer(
        StoreJob(req_id="r", tag="p-ok", num_tokens=4, block_ids=[1]), 0, "l0", torch.ones(2, 4, H, D, dtype=torch.float16)
    )
    assert [h.tag for h in w.finish_step()] == ["p-ok"]


def test_slot_map_views_k_and_v_planes_are_contiguous_but_the_pair_is_not() -> None:
    """Why direct_plane_copy exists: a slot file is [2, cap, H, D], so for t < cap the [:, :t] view skips cap - t rows
    between K and V. On a CUDA source PyTorch copies into such a view through a fresh host buffer (copy_kernel_cuda's
    temporaries path); dst[0, :t] and dst[1, :t] are each one contiguous run, so each copy lands in the map directly."""
    dst = torch.zeros(2, 64, H, D, dtype=torch.float16)
    assert not dst[:, :10].is_contiguous()
    assert dst[0, :10].is_contiguous() and dst[1, :10].is_contiguous()
    assert dst[:, :64].is_contiguous()  # only a full-capacity export is contiguous either way


@pytest.mark.parametrize("stream", [False, True])
def test_direct_plane_copy_writes_the_same_bytes(tmp_path: Any, stream: bool) -> None:
    """direct_plane_copy changes how the bytes get into the map, never which bytes: every layer file and every blob is
    identical to the default path, for a cold slot, a reused slot with a shorter prompt and a longer one."""
    roots = {flag: str(tmp_path / f"store-{flag}") for flag in (False, True)}
    writers = {
        flag: PagedWriter(ExportStore(root, slots=1, hold_seconds=0), info(64), stream_layers=stream, direct_plane_copy=flag)
        for flag, root in roots.items()
    }
    gen = torch.Generator().manual_seed(7)
    for rnd, n in enumerate((40, 12, 64)):  # cold, reused shorter, reused at full capacity
        kvs = [torch.randn(2, n, H, D, generator=gen).to(torch.float16) for _ in range(3)]
        headers = {}
        for flag, w in writers.items():
            job = StoreJob(req_id=f"r{rnd}", tag=f"p-{rnd}", num_tokens=n, block_ids=[1])
            for i, kv in enumerate(kvs):
                w.write_layer(job, i, f"model.layers.{i}.self_attn.attn", kv)
            (headers[flag],) = w.finish_step()
        assert [lr.to_dict() for lr in headers[False].layers] == [lr.to_dict() for lr in headers[True].layers]
        for i, kv in enumerate(kvs):
            name = headers[False].layers[i].file
            off = (tmp_path / "store-False" / "slots" / "s0" / name).read_bytes()
            on = (tmp_path / "store-True" / "slots" / "s0" / name).read_bytes()
            assert off == on
            with ExportReader(roots[True]).open(f"p-{rnd}") as ex:
                assert ex.read_blob(i, "k") == kv[0].numpy().tobytes()
                assert ex.read_blob(i, "v") == kv[1].numpy().tobytes()


def test_direct_plane_copy_config_flag() -> None:
    from kvhandoff.config import ConfigError, ConnectorConfig

    assert ConnectorConfig.from_extra({"store_path": "/s"}).direct_plane_copy  # default on
    assert not ConnectorConfig.from_extra({"store_path": "/s", "direct_plane_copy": False}).direct_plane_copy
    assert ConnectorConfig.from_extra({"store_path": "/s", "direct_plane_copy": True}).direct_plane_copy
    with pytest.raises(ConfigError, match="direct_plane_copy"):
        ConnectorConfig.from_extra({"store_path": "/s", "direct_plane_copy": "yes"})
