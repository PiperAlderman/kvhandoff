from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from kvhandoff.interchange import KIND_DSV4, BlobRef, ExportHeader, LayerRef
from kvhandoff.store import (
    DIR_MODE,
    FILE_MODE,
    ExportFailed,
    ExportNotFound,
    ExportReader,
    ExportStore,
    StoreBusy,
    StoreFull,
    UnsafeStoreLocation,
    capacity_report,
    check_ipc_safety,
    login_uid_min,
    main,
    paged_slot_bytes,
    store_mount,
    swap_problem,
)

MOUNTINFO = """\
22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw
30 22 0:25 / /dev/shm rw,nosuid,nodev shared:4 - tmpfs tmpfs rw
41 22 0:40 / /srv/kvhandoff rw,nosuid,nodev,noexec shared:20 - tmpfs tmpfs rw,size=12582912k,uid=980,noswap
42 22 0:41 / /srv/kvhandoff-old rw,relatime shared:21 - tmpfs tmpfs rw,size=25165824k,uid=980
43 22 0:42 / /srv/with\\040space rw,relatime shared:22 - tmpfs tmpfs rw,noswap
"""


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def header(tag: str, payload: bytes) -> ExportHeader:
    layer = LayerRef(
        index=0,
        name="l0",
        file="L0.bin",
        nbytes=len(payload),
        blobs=(BlobRef("state_header", "float32", (len(payload) // 4,), 0, len(payload)),),
    )
    return ExportHeader(kind=KIND_DSV4, tag=tag, model_id="m", num_tokens=4, layers=(layer,), num_model_layers=1)


def put(store: ExportStore, tag: str, payload: bytes | None = None) -> ExportHeader:
    payload = payload if payload is not None else tag.encode().ljust(16, b"\0")
    k = store.begin(tag)
    store.write_bytes(k, "L0.bin", 0, payload)
    return store.publish(k, header(tag, payload))


def test_publish_and_read_back(store_root: str) -> None:
    s = ExportStore(store_root, slots=2)
    h = put(s, "a-1", b"0123456789abcdef")
    r = ExportReader(store_root)
    assert r.lookup("a-1") == h
    with r.open("a-1", seq=h.seq) as ex:
        assert ex.read_blob(0, "state_header") == b"0123456789abcdef"
    assert h.seq > 0 and h.slot in (0, 1)


def test_layout_modes_are_group_readable_not_writable(store_root: str) -> None:
    s = ExportStore(store_root, slots=1)
    h = put(s, "a-1")
    slot = os.path.join(store_root, "slots", f"s{h.slot}")
    assert stat.S_IMODE(os.stat(store_root).st_mode) == DIR_MODE
    assert stat.S_IMODE(os.stat(slot).st_mode) == DIR_MODE
    for f in ("header.json", "L0.bin", "lock"):
        assert stat.S_IMODE(os.stat(os.path.join(slot, f)).st_mode) == FILE_MODE
    assert os.readlink(os.path.join(store_root, "tags", "a-1")) == os.path.join("..", "slots", f"s{h.slot}")


def test_unconsumed_recent_exports_are_held_then_store_full(store_root: str) -> None:
    clock = Clock()
    s = ExportStore(store_root, slots=2, hold_seconds=60, clock=clock)
    put(s, "a-1")
    put(s, "a-2")
    with pytest.raises(StoreFull, match="2 held"):
        s.begin("a-3")
    clock.t += 61  # hold expired: the oldest is reused
    h3 = put(s, "a-3")
    r = ExportReader(store_root)
    with pytest.raises(ExportNotFound):
        r.lookup("a-1")
    assert r.lookup("a-2").tag == "a-2" and r.lookup("a-3") == h3


def test_release_frees_a_slot_at_once(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, hold_seconds=3600)
    h = put(s, "a-1")
    r = ExportReader(store_root)
    assert not r.release("a-1", h.seq + 1)  # wrong seq: refused
    with pytest.raises(StoreFull):
        s.begin("a-2")
    assert r.release("a-1", h.seq)
    put(s, "a-2")
    with pytest.raises(ExportNotFound):
        r.lookup("a-1")


def test_a_reader_holding_a_slot_blocks_its_reuse(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, hold_seconds=0)
    h = put(s, "a-1")
    r = ExportReader(store_root)
    ex = r.open("a-1", seq=h.seq)
    with pytest.raises(StoreFull):
        s.begin("a-2")  # expired, but a reader is streaming it
    ex.close()
    put(s, "a-2")


def test_a_slot_being_rewritten_is_not_readable(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, hold_seconds=0)
    h = put(s, "a-1")
    k = s.begin("a-2")
    r = ExportReader(store_root)
    with pytest.raises(ExportNotFound):
        r.open("a-1", seq=h.seq)
    with pytest.raises(ExportNotFound):
        r.lookup("a-2")  # not published yet
    s.abandon(k)
    with pytest.raises(ExportNotFound):
        r.lookup("a-1")  # its slot was taken; the old tag is gone, never half-overwritten


def test_seq_check_detects_reuse(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, hold_seconds=0)
    h1 = put(s, "same")
    h2 = put(s, "same")  # the same tag exported again
    r = ExportReader(store_root)
    assert h2.seq > h1.seq
    with pytest.raises(ExportNotFound, match="reused"):
        r.open("same", seq=h1.seq)
    with r.open("same", seq=h2.seq):
        pass


def test_failure_records_fail_fast_and_are_bounded(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, failed_keep=3)
    for i in range(6):
        s.publish_failure(ExportHeader(kind=KIND_DSV4, tag=f"f-{i}", model_id="m", num_tokens=1, error=f"boom {i}"))
        os.utime(os.path.join(store_root, "failed", f"f-{i}.json"), (i, i))  # deterministic order
    r = ExportReader(store_root)
    with pytest.raises(ExportFailed, match="boom 5"):
        r.lookup("f-5")
    with pytest.raises(ExportFailed):
        r.open("f-5")
    assert len(os.listdir(os.path.join(store_root, "failed"))) <= 3
    put(s, "f-5")  # a later success replaces the failure record
    assert r.lookup("f-5").ok


def test_single_writer(store_root: str) -> None:
    s = ExportStore(store_root, slots=1)
    with pytest.raises(StoreBusy):
        ExportStore(store_root, slots=1)
    s.close()
    ExportStore(store_root, slots=1).close()


def test_restart_keeps_published_exports_and_seq_grows(store_root: str) -> None:
    s = ExportStore(store_root, slots=2, hold_seconds=3600)
    h = put(s, "a-1")
    os.symlink(os.path.join("..", "slots", "s1"), os.path.join(store_root, "tags", "stale"))  # points at an empty slot
    s.close()
    s2 = ExportStore(store_root, slots=2, hold_seconds=3600)
    r = ExportReader(store_root)
    assert r.lookup("a-1") == h
    assert not os.path.lexists(os.path.join(store_root, "tags", "stale"))
    h2 = put(s2, "a-2")
    assert h2.seq > h.seq
    with pytest.raises(StoreFull):  # a-1 is still held after the restart
        s2.begin("a-3")


def test_deleted_store_is_rebuilt_and_maps_invalidated(store_root: str) -> None:
    """The RemoveIPC trap: every file deleted under a running writer."""
    s = ExportStore(store_root, slots=2, hold_seconds=0)
    put(s, "a-1")
    gens = [s.slot_generation(k) for k in range(2)]
    shutil.rmtree(store_root)
    h = put(s, "a-2")
    assert all(s.slot_generation(k) > g for k, g in enumerate(gens))
    assert ExportReader(store_root).lookup("a-2") == h
    assert os.path.exists(os.path.join(store_root, "STORE"))


def test_deleted_slot_dir_is_rebuilt(store_root: str) -> None:
    s = ExportStore(store_root, slots=1, hold_seconds=0)
    h = put(s, "a-1")
    g = s.slot_generation(h.slot)
    shutil.rmtree(os.path.join(store_root, "slots", f"s{h.slot}"))
    with pytest.raises(ExportNotFound):
        ExportReader(store_root).open("a-1")
    put(s, "a-2")
    assert s.slot_generation(h.slot) == g + 1


def test_reader_never_follows_links_out_of_the_store(store_root: str, tmp_path: Path) -> None:
    ExportStore(store_root, slots=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "header.json").write_text("{}")
    os.symlink(str(outside), os.path.join(store_root, "tags", "evil"))
    with pytest.raises(ExportNotFound):
        ExportReader(store_root).lookup("evil")
    with pytest.raises(ExportNotFound):
        ExportReader(store_root).lookup("../../etc")


def test_publish_checks_files_are_long_enough(store_root: str) -> None:
    s = ExportStore(store_root, slots=1)
    k = s.begin("a-1")
    s.write_bytes(k, "L0.bin", 0, b"1234")
    with pytest.raises(Exception, match="shorter"):
        s.publish(k, header("a-1", b"12345678"))


def test_ipc_safety_rules(tmp_path: Path) -> None:
    assert check_ipc_safety(str(tmp_path), euid=1000, uid_min=1000) is None  # not under /dev/shm
    assert check_ipc_safety("/dev/shm/kvhandoff", euid=998, uid_min=1000) is None  # system uid
    assert check_ipc_safety("/dev/shm/kvhandoff", euid=0, uid_min=1000) is None
    problem = check_ipc_safety("/dev/shm/kvhandoff", euid=1000, uid_min=1000)
    assert problem is not None and "RemoveIPC" in problem


def test_store_refuses_dev_shm_for_a_login_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    import kvhandoff.store as st

    def unsafe(root: str) -> str:
        return f"{root} under /dev/shm as uid 1000 (RemoveIPC)"

    monkeypatch.setattr(st, "check_ipc_safety", unsafe)
    with pytest.raises(UnsafeStoreLocation):
        ExportStore("/nonexistent-never-created", slots=1)


def test_store_mount_finds_the_longest_prefix_and_noswap(tmp_path: Path) -> None:
    mi = tmp_path / "mountinfo"
    mi.write_text(MOUNTINFO)
    m = store_mount("/srv/kvhandoff/l12", str(mi))  # need not exist
    assert m is not None and m["mount_point"] == "/srv/kvhandoff" and m["fstype"] == "tmpfs" and m["noswap"]
    assert swap_problem(m) is None
    old = store_mount("/srv/kvhandoff-old/x", str(mi))  # a sibling with a longer name is not a prefix match for the first
    assert old is not None and old["mount_point"] == "/srv/kvhandoff-old" and not old["noswap"]
    problem = swap_problem(old)
    assert problem is not None and "noswap" in problem
    sp = store_mount("/srv/with space/s", str(mi))
    assert sp is not None and sp["mount_point"] == "/srv/with space" and sp["noswap"]
    root = store_mount("/var/lib/x", str(mi))
    assert root is not None and root["fstype"] == "ext4" and swap_problem(root) is None  # only tmpfs is judged
    assert store_mount("/x", str(tmp_path / "missing")) is None and swap_problem(None) is None


def test_writer_warns_once_on_a_swappable_tmpfs(store_root: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import kvhandoff.store as st

    seen: list[tuple[str, dict[str, object]]] = []

    class Log:
        def warning(self, event: str, **kw: object) -> None:
            seen.append((event, kw))

    swappable = {"mount_point": "/srv/kvhandoff", "fstype": "tmpfs", "options": [], "noswap": False}
    monkeypatch.setattr(st, "log", Log())
    monkeypatch.setattr(st, "store_mount", lambda _root: swappable)
    ExportStore(store_root, slots=1).close()
    assert [e for e, _ in seen] == ["store_swappable"]
    seen.clear()
    monkeypatch.setattr(st, "store_mount", lambda _root: {**swappable, "noswap": True})
    ExportStore(store_root, slots=1).close()
    assert seen == []


def test_check_require_noswap(store_root: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import kvhandoff.store as st

    swappable = {"mount_point": "/srv/kvhandoff", "fstype": "tmpfs", "options": [], "noswap": False}
    monkeypatch.setattr(st, "store_mount", lambda _root: swappable)
    args = ["check", "--root", store_root, "--slots", "1", "--slot-bytes", "1"]
    assert main(args) == 0  # reported, not enforced, by default
    doc = json.loads(capsys.readouterr().out)
    assert doc["mount"] == {"mount_point": "/srv/kvhandoff", "fstype": "tmpfs", "noswap": False} and doc["swap_problem"]
    assert main([*args, "--require-noswap"]) == 1
    capsys.readouterr()
    monkeypatch.setattr(st, "store_mount", lambda _root: {**swappable, "noswap": True})
    assert main([*args, "--require-noswap"]) == 0
    assert json.loads(capsys.readouterr().out)["swap_problem"] is None


def test_login_defs_parsing(tmp_path: Path) -> None:
    p = tmp_path / "login.defs"
    p.write_text("# c\nUID_MIN\t\t\t 500\nUID_MAX 60000\n")
    assert login_uid_min(str(p)) == 500
    assert login_uid_min(str(tmp_path / "none")) == 1000


def test_capacity_and_cli(store_root: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", "--root", store_root, "--slots", "1"]) == 0
    s = ExportStore(store_root, slots=1)  # the writer adopts the layout init made
    put(s, "after-init")
    per = paged_slot_bytes(layers=80, max_tokens=34064, kv_heads=8, head_dim=128, dtype_bytes=2)
    assert per == 80 * 2 * 34064 * 8 * 128 * 2  # 11.2 GB: Llama-3.3-70B fp16 at max_model_len 34000
    assert capacity_report(store_root, 1)["fits"]
    assert main(["check", "--root", store_root, "--slots", "1", "--slot-bytes", "1"]) == 0
    assert main(["check", "--root", store_root, "--slots", "1", "--slot-bytes", str(1 << 60)]) == 1
    capsys.readouterr()
    assert main(["stats", "--root", store_root]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["root_ok"] and stats["slots"] == 1
