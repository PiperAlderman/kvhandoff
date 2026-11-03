"""Slot map pre-faulting (measured). A new map has no page-table entries in this process even over file
pages that already exist; on a GB10 the GPU copy then faults each page itself, slowly. SlotMaps.prefault populates
them on the CPU first. These tests read the map's resident set from /proc/self/smaps, so they check the page tables
directly: on a CPU a plain copy would fault the pages in anyway and hide the effect."""

from __future__ import annotations

import errno
import os
import re
import sys
from typing import Any

import pytest

pytest.importorskip("torch")  # the producer side needs torch; a torch-less install skips this module

import torch

from kvhandoff.interchange import RopeInfo
from kvhandoff.producer import maps as maps_mod
from kvhandoff.producer.maps import PAGE, SlotMaps
from kvhandoff.producer.paged import PagedExportInfo, PagedWriter, StoreJob
from kvhandoff.store import ExportReader, ExportStore

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="madvise and /proc/self/smaps are Linux")

H, D = 4, 64  # 512 bytes a row in fp16: 8 rows a 4 KiB page
ROW = H * D * 2
_VMA = re.compile(r"^([0-9a-f]+)-[0-9a-f]+ ")


def rss(t: torch.Tensor) -> int:
    """Resident bytes of the VMA that starts at t's data pointer (pages with a page-table entry in this process)."""
    base = t.data_ptr()
    found = False
    with open("/proc/self/smaps", encoding="ascii") as f:
        for line in f:
            vma = _VMA.match(line)
            if vma:
                found = int(vma.group(1), 16) == base
                continue
            if found and line.startswith("Rss:"):
                return int(line.split()[1]) * 1024
    raise AssertionError("mapping not found in /proc/self/smaps")


def info(cap: int) -> PagedExportInfo:
    return PagedExportInfo("m", 4, cap, RopeInfo(style="neox", applied_by="vllm"), {"connector": "test"})


def allocated_store(root: str, cap: int, t: int) -> tuple[ExportStore, int]:
    """A store whose slot 0 file L0.bin has its K and V rows [0, t) allocated and written by another map."""
    st = ExportStore(root, slots=1, hold_seconds=0)
    k = st.begin("p-warm")
    m = SlotMaps(st)
    dst = m.typed_map(k, "L0.bin", (2, cap, H, D), torch.float16)
    st.reserve(k, "L0.bin", 0, t * ROW)
    st.reserve(k, "L0.bin", cap * ROW, t * ROW)
    dst[:, :t].fill_(1.0)
    del dst, m
    return st, k


def test_new_map_has_no_entries_and_prefault_populates_exactly_the_ranges(tmp_path: Any) -> None:
    cap, t = 4096, 1000  # file 4 MiB; each plane range about 500 KiB
    st, k = allocated_store(str(tmp_path / "s"), cap, t)
    blocks_before = os.stat(st.slot_dir(k) + "/L0.bin").st_blocks
    m = SlotMaps(st)
    dst = m.typed_map(k, "L0.bin", (2, cap, H, D), torch.float16)
    raw = m.bytes_map(k, "L0.bin", 2 * cap * ROW, exact=True)
    assert rss(raw) == 0  # the pages exist in the file, but not in this process's page tables
    assert m.prefault(k, "L0.bin", 0, t * ROW) == t * ROW
    assert m.prefault(k, "L0.bin", cap * ROW, t * ROW) == t * ROW
    want = 2 * t * ROW
    assert want <= rss(raw) <= want + 2 * PAGE  # each range rounded out to whole pages, nothing else
    assert os.stat(st.slot_dir(k) + "/L0.bin").st_blocks == blocks_before  # no new file pages
    assert torch.all(dst[:, :t] == 1.0)  # populating never changes the bytes


def test_prefault_is_once_per_range_and_grows_by_the_delta(tmp_path: Any) -> None:
    cap = 4096
    st, k = allocated_store(str(tmp_path / "s"), cap, cap)
    m = SlotMaps(st)
    m.typed_map(k, "L0.bin", (2, cap, H, D), torch.float16)
    assert m.prefault(k, "L0.bin", 0, 800 * ROW) == 800 * ROW
    calls = m.prefault_calls
    assert m.prefault(k, "L0.bin", 0, 800 * ROW) == 0  # warm export, same size: no madvise at all
    assert m.prefault(k, "L0.bin", 0, 300 * ROW) == 0  # shorter: already covered
    assert m.prefault_calls == calls
    assert m.prefault(k, "L0.bin", 0, 2000 * ROW) == 1200 * ROW  # longer: only what grew
    assert m.prefault_calls == calls + 1


def test_remap_resets_the_populated_ranges(tmp_path: Any) -> None:
    cap = 4096
    st, k = allocated_store(str(tmp_path / "s"), cap, cap)
    m = SlotMaps(st)
    m.typed_map(k, "L0.bin", (2, cap, H, D), torch.float16)
    m.prefault(k, "L0.bin", 0, 500 * ROW)
    m.typed_map(k, "L0.bin", (2, cap // 2, H, D), torch.float16)  # a new shape is a new map with no entries
    raw = m.bytes_map(k, "L0.bin", cap * ROW, exact=True)
    assert rss(raw) == 0
    assert m.prefault(k, "L0.bin", 0, 500 * ROW) == 500 * ROW
    assert rss(raw) >= 500 * ROW


def test_prefault_failure_never_fails_the_export(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(maps_mod, "madvise", lambda addr, length, advice: errno.EINVAL)  # pre-5.14 kernel
    root = str(tmp_path / "s")
    w = PagedWriter(ExportStore(root, slots=1, hold_seconds=0), info(64), prefault_slot_maps=True)
    kv = torch.randn(2, 40, H, D).to(torch.float16)
    job = StoreJob(req_id="r", tag="p-x", num_tokens=40, block_ids=[1])
    for i in range(2):
        w.write_layer(job, i, f"model.layers.{i}.self_attn.attn", kv)
    (hdr,) = w.finish_step()
    assert len(hdr.layers) == 2
    assert not w.maps.prefault_enabled and w.maps.prefault_calls == 1  # disabled after the first refusal
    with ExportReader(root).open("p-x") as ex:
        assert ex.read_blob(1, "v") == kv[1].numpy().tobytes()


@pytest.mark.parametrize("direct", [False, True])
def test_writer_prefaults_only_the_rows_it_writes(tmp_path: Any, direct: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, int]] = []
    real = maps_mod.madvise

    def spy(addr: int, length: int, advice: int) -> int:
        seen.append((addr, length))
        return real(addr, length, advice)

    monkeypatch.setattr(maps_mod, "madvise", spy)
    cap = 2048
    w = PagedWriter(
        ExportStore(str(tmp_path / "s"), slots=1, hold_seconds=0), info(cap), direct_plane_copy=direct, prefault_slot_maps=True
    )
    for rnd, n in enumerate((600, 600, 1200)):  # cold, warm same size, longer
        kv = torch.randn(2, n, H, D).to(torch.float16)
        job = StoreJob(req_id=f"r{rnd}", tag=f"p-{rnd}", num_tokens=n, block_ids=[1])
        w.write_layer(job, 0, "model.layers.0.self_attn.attn", kv)
        assert len(w.finish_step()) == 1
    # cold: K and V ranges; warm: nothing; longer: the K and V deltas only
    assert len(seen) == 4
    assert all(length <= 600 * ROW + 2 * PAGE for _, length in seen)


def test_prefault_flag_writes_the_same_bytes(tmp_path: Any) -> None:
    roots = {flag: str(tmp_path / f"store-{flag}") for flag in (False, True)}
    writers = {
        flag: PagedWriter(ExportStore(root, slots=1, hold_seconds=0), info(64), prefault_slot_maps=flag)
        for flag, root in roots.items()
    }
    gen = torch.Generator().manual_seed(11)
    for rnd, n in enumerate((40, 12, 64)):
        kv = torch.randn(2, n, H, D, generator=gen).to(torch.float16)
        for w in writers.values():
            w.write_layer(StoreJob(req_id=f"r{rnd}", tag=f"p-{rnd}", num_tokens=n, block_ids=[1]), 0, "l0", kv)
            w.finish_step()
        off = (tmp_path / "store-False" / "slots" / "s0" / "L0.bin").read_bytes()
        on = (tmp_path / "store-True" / "slots" / "s0" / "L0.bin").read_bytes()
        assert off == on
    assert writers[False].maps.prefault_calls == 0 and writers[True].maps.prefault_calls > 0


def test_prefault_config_flag() -> None:
    from kvhandoff.config import ConfigError, ConnectorConfig

    assert ConnectorConfig.from_extra({"store_path": "/s"}).prefault_slot_maps
    assert ConnectorConfig.from_extra({"store_path": "/s", "prefault_slot_maps": True}).prefault_slot_maps
    assert not ConnectorConfig.from_extra({"store_path": "/s", "prefault_slot_maps": False}).prefault_slot_maps
    with pytest.raises(ConfigError, match="prefault_slot_maps"):
        ConnectorConfig.from_extra({"store_path": "/s", "prefault_slot_maps": "yes"})


def test_timing_log_carries_stage_seconds_and_swap_deltas(tmp_path: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The exported line splits the export into reserve, prefault and copy time and carries the
    host's swap-in/out page deltas over the export, so a hardware run shows where a slow first export spends it."""
    import logging

    w = PagedWriter(ExportStore(str(tmp_path / "s"), slots=1, hold_seconds=0), info(64), timing_log=True)
    kv = torch.randn(2, 40, H, D).to(torch.float16)
    job = StoreJob(req_id="r", tag="p-t", num_tokens=40, block_ids=[1])
    with caplog.at_level(logging.INFO, logger="kvhandoff"):
        w.write_layer(job, 0, "model.layers.0.self_attn.attn", kv)
        assert len(w.finish_step()) == 1
    (rec,) = [r for r in caplog.records if r.getMessage() == "exported"]
    fields: dict[str, Any] = rec.fields  # type: ignore[attr-defined]
    for key in ("reserve_seconds", "prefault_seconds", "copy_seconds"):
        assert fields[key] >= 0
    assert fields["pswpin"] is None or fields["pswpin"] >= 0
    assert "pswpout" in fields


@pytest.mark.parametrize("stream", [False, True])
def test_timing_log_has_one_row_per_layer(tmp_path: Any, caplog: pytest.LogCaptureFixture, stream: bool) -> None:
    """A slow streamed export can show reserve, prefault and copy all small, with the time between the writer's
    stages. export_layers carries one row per layer (enter and exit offsets, the caller's gather time, each stage),
    so the gaps between layers and the stream-log appends show."""
    import logging
    import time

    from kvhandoff.producer.paged import LAYER_COLUMNS

    w = PagedWriter(ExportStore(str(tmp_path / "s"), slots=1, hold_seconds=0), info(64), timing_log=True, stream_layers=stream)
    kv = torch.randn(2, 40, H, D).to(torch.float16)
    job = StoreJob(req_id="r", tag="p-l", num_tokens=40, block_ids=[1])
    with caplog.at_level(logging.INFO, logger="kvhandoff"):
        for i in range(3):
            entered = time.perf_counter()
            time.sleep(0.002)
            w.write_layer(job, i, f"model.layers.{i}.self_attn.attn", kv, entered=entered if i else None)
        assert len(w.finish_step()) == 1
    (ex,) = [r.fields for r in caplog.records if r.getMessage() == "exported"]  # type: ignore[attr-defined]
    (lt,) = [r.fields for r in caplog.records if r.getMessage() == "export_layers"]  # type: ignore[attr-defined]
    assert lt["columns"] == list(LAYER_COLUMNS) and lt["tag"] == "p-l" and lt["tokens"] == 40
    rows = [dict(zip(LAYER_COLUMNS, r, strict=True)) for r in lt["rows"]]
    assert [r["layer"] for r in rows] == [0, 1, 2]
    assert rows[0]["gather_ms"] is None and all(r["gather_ms"] >= 2.0 for r in rows[1:])
    assert all(r["exit_ms"] >= r["enter_ms"] >= 0 for r in rows)
    assert all(rows[i + 1]["enter_ms"] >= rows[i]["exit_ms"] for i in range(2))
    assert all((r["stream_ms"] > 0) == stream for r in rows)
    assert ex["gather_seconds"] >= 0.004 and (ex["stream_seconds"] > 0) == stream


def test_timing_log_off_keeps_no_rows(tmp_path: Any, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    w = PagedWriter(ExportStore(str(tmp_path / "s"), slots=1, hold_seconds=0), info(64))
    kv = torch.randn(2, 40, H, D).to(torch.float16)
    job = StoreJob(req_id="r", tag="p-o", num_tokens=40, block_ids=[1])
    with caplog.at_level(logging.INFO, logger="kvhandoff"):
        w.write_layer(job, 0, "model.layers.0.self_attn.attn", kv, entered=0.0)
        assert len(w.finish_step()) == 1
    assert not [r for r in caplog.records if r.getMessage() in ("exported", "export_layers")]
