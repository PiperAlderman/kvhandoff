"""The two connectors driven the way vLLM drives them, with vLLM stubbed (tests/fakes/vllm_stub.py)."""

from __future__ import annotations

import inspect
import types
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from fakes import vllm_stub
from kvhandoff.producer import dsv4_convert as cv
from kvhandoff.producer.paged import gather, slot_mapping, to_kv_split
from kvhandoff.store import ExportFailed, ExportReader
from test_dsv4 import fake_model
from test_paged import BS, A, B, D, H, new, paged_layer_tensor, reference, sched_out


@pytest.fixture
def stub() -> Iterator[dict[str, types.ModuleType]]:
    with vllm_stub.installed() as mods:
        yield mods


def vllm_cfg(extra: dict[str, Any], **model: Any) -> Any:
    ns = types.SimpleNamespace
    return ns(
        kv_transfer_config=ns(kv_connector_extra_config=extra),
        cache_config=ns(block_size=BS),
        model_config=ns(
            max_model_len=448,
            get_head_size=lambda: D,
            served_model_name=["llama70b"],
            model="/models/x",
            hf_config=ns(num_hidden_layers=3, rope_theta=500000.0, rope_scaling=None, **model.get("hf", {})),
        ),
        scheduler_config=ns(max_num_seqs=4),
    )


def test_constructors_take_kv_cache_config(stub: Any) -> None:
    """vLLM's kv_connector_module_path loader rejects connectors without a kv_cache_config argument."""
    from kvhandoff.vllm.dsv4_connector import KVHandoffDSV4Connector
    from kvhandoff.vllm.paged_connector import KVHandoffPagedConnector

    for cls in (KVHandoffPagedConnector, KVHandoffDSV4Connector):
        params = list(inspect.signature(cls.__init__).parameters)
        assert params[:4] == ["self", "vllm_config", "role", "kv_cache_config"]
        base = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"]
        assert issubclass(cls, base.SupportsHMA) and issubclass(cls, base.KVConnectorBase_V1)
        assert not getattr(cls, "__abstractmethods__", set())  # every abstract method of v0.30.0 / main implemented


def test_plugin_registers_both_names_idempotently(stub: Any) -> None:
    from kvhandoff.vllm.plugin import register

    assert register() == ["KVHandoffPagedConnector", "KVHandoffDSV4Connector"]
    assert register() == []  # vLLM may call general plugins more than once per process
    reg = stub["vllm.distributed.kv_transfer.kv_connector.factory"].KVConnectorFactory._registry
    assert reg["KVHandoffDSV4Connector"] == ("kvhandoff.vllm.dsv4_connector", "KVHandoffDSV4Connector")


def test_plugin_without_vllm_is_a_no_op() -> None:
    from kvhandoff.vllm.plugin import register

    assert register() == []


@pytest.mark.parametrize("direct_plane_copy", [False, True])
def test_paged_connector_end_to_end(stub: Any, store_root: str, direct_plane_copy: bool) -> None:
    from kvhandoff.vllm.paged_connector import KVHandoffPagedConnector, PagedHandoffMetadata

    extra: dict[str, Any] = {"store_path": store_root, "out_dtype": "float16", "direct_plane_copy": direct_plane_copy}
    cfg = vllm_cfg(extra)
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    sched, worker = KVHandoffPagedConnector(cfg, role.SCHEDULER, None), KVHandoffPagedConnector(cfg, role.WORKER, None)
    layers = {f"model.layers.{i}.self_attn.attn": paged_layer_tensor("v030", 64, i) for i in range(3)}

    def step(so: Any) -> list[str]:
        meta = sched.build_connector_meta(so)
        assert isinstance(meta, PagedHandoffMetadata)
        worker._connector_metadata = meta
        for name, t in layers.items():
            worker.save_kv_layer(name, t, None)
        worker.wait_for_save()
        return [j.tag for j in meta.requests]

    assert step(sched_out([new("a", 300, A[:8], "tagA"), new("c", 50, [30, 31, 32, 33], None)], [], {"a": 128, "c": 50})) == []
    assert step(sched_out([new("b", 100, B[:7], "tagB")], [("a", (A[8:16],), 128)], {"a": 128, "b": 100})) == ["tagB"]
    assert step(sched_out([], [("a", (A[16:19],), 256)], {"a": 44})) == ["tagA"]
    assert sched.get_num_new_matched_tokens(None, 0) == (0, False)
    assert worker._writer is not None and worker._writer.direct_plane_copy is direct_plane_copy
    r = ExportReader(store_root)
    for tag, blocks, n in (("tagA", A, 300), ("tagB", B, 100)):
        h = r.lookup(tag)
        assert h.model_id == "llama70b" and h.num_tokens == n and h.num_model_layers == 3
        assert h.rope is not None and h.rope.style == "neox" and h.rope.theta == 500000.0
        assert h.producer["connector"] == "KVHandoffPagedConnector" and h.producer["block_size"] == BS
        with r.open(tag, seq=h.seq) as ex:
            for i, t in enumerate(layers.values()):
                ref = reference(t, "v030", blocks, n).to(torch.float16)
                assert ex.read_blob(i, "k") == ref[0].numpy().tobytes()
                assert ex.read_blob(i, "v") == ref[1].numpy().tobytes()
    with pytest.raises(Exception):  # noqa: B017  (untagged request "c" is never stored)
        r.lookup("c")


def test_paged_connector_tensor_parallel(stub: Any, store_root: str) -> None:
    """TP2: each rank sees two of four heads; the all-gather concatenates, only rank 0 writes."""
    from kvhandoff.vllm.paged_connector import KVHandoffPagedConnector

    cfg = vllm_cfg({"store_path": store_root})
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    full = torch.cat([paged_layer_tensor("v030", 32, 9), paged_layer_tensor("v030", 32, 10)], dim=3)  # [nb,2,bs,4,D]
    shards = [full[:, :, :, r * H : (r + 1) * H].contiguous() for r in range(2)]
    blocks, n = [3, 9, 1], 40
    slots = slot_mapping(blocks, n, BS)
    split = [to_kv_split(gather(s, slots, BS), D) for s in shards]

    sched = KVHandoffPagedConnector(cfg, role.SCHEDULER, None)
    meta = sched.build_connector_meta(sched_out([new("a", n, blocks, "tp-1")], [], {"a": n}))
    for rank in (1, 0):
        tp = vllm_stub.TP(world_size=2, rank=rank)
        tp.gather_fn = lambda _x, dim: torch.cat(split, dim=dim)  # pyright: ignore[reportUnknownArgumentType]
        vllm_stub.STATE["tp"] = tp
        w = KVHandoffPagedConnector(cfg, role.WORKER, None)
        w._connector_metadata = meta
        w.save_kv_layer("model.layers.0.self_attn.attn", shards[rank], None)
        w.wait_for_save()
        if rank == 1:
            assert w._writer is None  # rank 1 never touches the store
    h = ExportReader(store_root).lookup("tp-1")
    k = h.layers[0].blobs[0]
    assert k.shape == (n, 2 * H, D)
    with ExportReader(store_root).open("tp-1") as ex:
        assert ex.read_blob(0, "k") == to_kv_split(gather(full, slots, BS), D)[0].view(torch.int16).numpy().tobytes()


def test_paged_connector_unsupported_layout_fails_cleanly(stub: Any, store_root: str) -> None:
    from kvhandoff.vllm.paged_connector import KVHandoffPagedConnector

    cfg = vllm_cfg({"store_path": store_root})
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    sched, worker = KVHandoffPagedConnector(cfg, role.SCHEDULER, None), KVHandoffPagedConnector(cfg, role.WORKER, None)
    worker._connector_metadata = sched.build_connector_meta(sched_out([new("a", 20, [1, 2], "mla-1")], [], {"a": 20}))
    worker.save_kv_layer("model.layers.0.self_attn.attn", torch.zeros(8, 1, BS, 576), None)  # MLA-like
    worker.wait_for_save()
    with pytest.raises(ExportFailed, match="UnsupportedLayout"):
        ExportReader(store_root).lookup("mla-1")


def test_dsv4_connector_tp4(stub: Any, store_root: str) -> None:
    from kvhandoff.vllm.dsv4_connector import DSV4HandoffMetadata, KVHandoffDSV4Connector

    t, ratios = 200, [0, 4, 128]
    kv, groups, mods, bids, srcs = fake_model(t, ratios)
    ns = types.SimpleNamespace
    kv_cache_config = ns(kv_cache_groups=[ns(layer_names=[n for n, g in groups.items() if g == i]) for i in range(len(bids))])
    cfg = vllm_cfg({"store_path": store_root, "slots": 4}, hf={"compress_ratios": ratios})
    cfg.model_config.hf_config.num_hidden_layers = 3
    role = stub["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorRole
    sched = KVHandoffDSV4Connector(cfg, role.SCHEDULER, kv_cache_config)

    def req(rid: str, tag: str | None) -> Any:
        return ns(request_id=rid, num_prompt_tokens=t, kv_transfer_params={"kvhandoff_tag": tag} if tag else {})

    assert sched.request_finished_all_groups(req("r0", "d-0"), tuple(bids)) == (True, None)
    assert sched.request_finished_all_groups(req("r1", "d-1"), tuple(bids)) == (True, None)
    assert sched.request_finished_all_groups(req("plain", None), tuple(bids)) == (False, None)
    meta = sched.build_connector_meta(None)
    again = sched.build_connector_meta(None)
    assert isinstance(meta, DSV4HandoffMetadata) and isinstance(again, DSV4HandoffMetadata)
    assert [j.tag for j in meta.jobs] == ["d-0", "d-1"] and again.jobs == []
    vllm_stub.STATE["forward"] = mods
    outs = []
    for rank in range(4):
        vllm_stub.STATE["tp_rank"] = rank
        w = KVHandoffDSV4Connector(cfg, role.WORKER, kv_cache_config)
        w.register_kv_caches(kv)
        w._connector_metadata = meta
        outs.append(w.get_finished({"r0", "plain"}))
        w._connector_metadata = DSV4HandoffMetadata([])
        outs.append(w.get_finished({"r1"}))
        assert (w._exporter is not None) == (rank == 0)
    assert outs == [({"r0"}, None), ({"r1"}, None)] * 4
    r = ExportReader(store_root)
    for tag in ("d-0", "d-1"):
        h = r.lookup(tag)
        assert h.num_tokens == t and [lr.attrs["compress_ratio"] for lr in h.layers] == ratios
        with r.open(tag) as ex:
            for i, layer in enumerate(h.layers):
                want = [cv.blob_bytes(x).numpy().tobytes() for _, _, _, x in cv.convert_layer(srcs[i], t)]
                assert [ex.read_range(i, b.offset, b.nbytes) for b in layer.blobs] == want
