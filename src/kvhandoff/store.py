"""The export store: a bounded set of slots on a memory-backed filesystem, one writer, any number of readers.

Layout under ``root`` (directories 2750, files 0640, so a reader in the store's group can read but not write)::

    STORE                    marker (JSON); if it disappears the whole store was deleted under the writer
    writer.lock              flock(LOCK_EX) by the single writer (vLLM TP rank 0) for its lifetime
    slots/s<k>/lock          flock: the writer holds LOCK_EX from begin() to publish(); a reader holds LOCK_SH while
                             it reads, and a writer never reuses a slot a reader holds
    slots/s<k>/header.json   the export's interchange header (kvhandoff.interchange), written last
    slots/s<k>/L<i>.bin      layer files, pre-sized and kept mapped by the writer (fresh tmpfs pages are slow on GB10)
    slots/s<k>/consumed      created by release(): the reader is done, the slot may be reused at once
    slots/s<k>/stream.log    streaming mode only: JSON lines, the export's descriptor then one line per completed layer,
                             then "published" or "failed"; appended under flock(LOCK_EX), read under flock(LOCK_SH)
    tags/<tag> -> ../slots/s<k>   published atomically (rename) once header.json is complete
    streaming/<tag> -> ../slots/s<k>   streaming mode only: created at begin(), so layers can be read before publish
    failed/<tag>.json        a failure record (header with ``error`` set) for an export that could not be written

Retention and back-pressure: a slot is reused only when it is empty, consumed, or older than ``hold_seconds``, and
no reader holds it. When no slot qualifies, begin() raises StoreFull at once (the vLLM step is never blocked); the
connector then writes a failure record so the bridge fails fast and EXO prefills locally.

Streaming mode (begin(tag, stream=...)): the export's seq is fixed at begin() and kept by publish(); the writer keeps
the slot reserved in-process but does not hold its flock, so readers can take LOCK_SH and read completed layers while
later ones are still being written. A reader's LOCK_SH still blocks reuse of the slot. header.json and tags/<tag> are
written only at publish(), exactly as without streaming, so a reader that knows nothing of streaming never sees a
partial export. Ordering: a layer's bytes are complete before its stream.log line is appended under LOCK_EX, and a
reader reads stream.log under LOCK_SH before it reads those bytes; the unlock/lock pair on the same file lock orders
the writer's stores before the reader's loads (see docs/layerwise.md, "Memory ordering").

systemd-logind RemoveIPC deletes /dev/shm files owned by a login user when that user's last session ends (seen in
practice wiping a store mid-run). The store refuses such a location unless told otherwise (check_ipc_safety),
and a writer that finds its slot files gone recreates them and bumps the slot's generation, so file maps held for the
deleted files are dropped instead of written into forever.

Swap: on tmpfs without ``noswap`` an idle store is pushed to swap whenever another process needs memory (on a DGX Spark,
every new vLLM start-up), and the next export pays to swap it back in. The writer logs ``store_swappable`` once at start in
that case, and ``kvhandoff-store check --require-noswap`` refuses it (store_mount, swap_problem).
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from kvhandoff.interchange import ExportHeader, FormatError, LayerRef, validate
from kvhandoff.logs import get_logger
from kvhandoff.tags import is_valid_tag

LAYOUT_VERSION: Final = 1
DIR_MODE: Final = 0o2750
FILE_MODE: Final = 0o640
MARKER: Final = "STORE"
STREAM_LOG: Final = "stream.log"
STREAM_FORMAT: Final = 1

log = get_logger("store")


class StoreError(RuntimeError):
    pass


class StoreFull(StoreError):
    """No slot can be reused right now (all held by unconsumed recent exports or by readers)."""


class StoreBusy(StoreError):
    """Another writer already owns this store."""


class UnsafeStoreLocation(StoreError):
    """The store would live where systemd-logind RemoveIPC can delete it."""


class ExportNotFound(StoreError):
    """No such export: never published, evicted, or its slot is being rewritten."""


class ExportFailed(StoreError):
    def __init__(self, header: ExportHeader):
        super().__init__(f"export {header.tag} failed on the producer: {header.error}")
        self.header = header


@dataclass(frozen=True)
class StreamState:
    """An export as a reader sees it: while streaming, ``header.layers`` holds the layers completed so far (in index
    order) and ``header.seq`` is the seq publish() will keep; once published, the final header."""

    header: ExportHeader
    published: bool


# ----------------------------------------------------------------------------------------------- location safety
def login_uid_min(login_defs: str = "/etc/login.defs") -> int:
    """UID_MIN from login.defs (first uid logind treats as a login user), 1000 if unset."""
    try:
        with open(login_defs, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "UID_MIN":
                    return int(parts[1])
    except (OSError, ValueError):
        pass
    return 1000


def check_ipc_safety(root: str, *, euid: int | None = None, uid_min: int | None = None) -> str | None:
    """A reason the location is unsafe, or None.

    RemoveIPC removes POSIX shared memory, i.e. files under /dev/shm (and /dev/mqueue), owned by a non-system user.
    A store outside /dev/shm (a dedicated tmpfs mount) is never touched; one inside it is safe only for a system uid.
    """
    real = os.path.realpath(root)
    in_shm = real == "/dev/shm" or real.startswith("/dev/shm/")  # noqa: S108  (the location being checked, not used)
    if not in_shm:
        return None
    uid = os.geteuid() if euid is None else euid
    limit = login_uid_min() if uid_min is None else uid_min
    if uid == 0 or uid < limit:
        return None
    return (
        f"store {real} is under /dev/shm and this process runs as uid {uid} (>= UID_MIN {limit}); "
        "systemd-logind RemoveIPC deletes such files when the user's last session ends. Run as a system "
        "account, or put the store on a dedicated tmpfs mount"
    )


def _unescape_mountinfo(field: str) -> str:
    """mountinfo escapes space, tab, newline and backslash as \\ooo octal."""
    out: list[str] = []
    i = 0
    while i < len(field):
        code = field[i + 1 : i + 4]
        if field[i] == "\\" and len(code) == 3 and all(c in "01234567" for c in code):
            out.append(chr(int(code, 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def store_mount(root: str, mountinfo: str = "/proc/self/mountinfo") -> dict[str, Any] | None:
    """The mount that holds ``root`` (longest mount point that is a prefix of its real path), or None if unknown.

    Returns ``{"mount_point", "fstype", "options", "noswap"}``. ``options`` merges the per-mount and per-superblock
    option lists. For tmpfs, ``noswap`` (Linux 6.4+) keeps the pages out of swap: without it a new vLLM process's
    start-up pushes an idle store out to swap and the next export pays to swap it back in through ``reserve()``'s
    fallocate, at 8 to 10 s/GiB on a GB10 (measured). The root need not exist yet.
    """
    path = os.path.realpath(root)
    try:
        with open(mountinfo, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    best: dict[str, Any] | None = None
    for line in lines:
        left, sep, right = line.partition(" - ")
        lf, rf = left.split(), right.split()
        if not sep or len(lf) < 6 or len(rf) < 3:
            continue
        mp = _unescape_mountinfo(lf[4])
        inside = mp in (path, "/") or path.startswith(mp.rstrip("/") + "/")
        if not inside or (best is not None and len(mp) < len(best["mount_point"])):
            continue
        opts = [o for o in lf[5].split(",") + rf[2].split(",") if o]
        best = {"mount_point": mp, "fstype": rf[0], "options": opts, "noswap": "noswap" in opts}
    return best


def swap_problem(mount: dict[str, Any] | None) -> str | None:
    """A reason the store can be swapped out, or None. Only tmpfs is judged; ramfs never swaps."""
    if mount is None or mount["fstype"] != "tmpfs" or mount["noswap"]:
        return None
    return (
        f"store is on tmpfs {mount['mount_point']} without noswap: an idle store is pushed to swap when another "
        "process (a new vLLM) needs memory, and the next export swaps it back in at about 8 to 10 s/GiB. Mount the "
        "store's tmpfs with noswap (Linux 6.4+) and size= no larger than the memory you can pin for it"
    )


def capacity_report(root: str, required_bytes: int) -> dict[str, Any]:
    """Filesystem size and free space against the store's worst case (slots x largest export)."""
    st = os.statvfs(root)
    size, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
    return {"fs_bytes": size, "fs_free_bytes": free, "required_bytes": required_bytes, "fits": required_bytes <= size}


# ----------------------------------------------------------------------------------------------- helpers
def _mkdir(path: str) -> None:
    """Create ``path`` (or adopt an existing one we own) with mode 2750: group-readable, setgid so files keep the
    store's group. A directory owned by someone else (e.g. a mount point prepared by root) keeps its mode."""
    try:
        os.mkdir(path, DIR_MODE)
    except FileExistsError:
        if os.stat(path).st_uid != os.geteuid():
            return
    with contextlib.suppress(PermissionError):
        os.chmod(path, DIR_MODE)  # mkdir applies the umask; chmod sets the setgid bit and exact mode


def _write_file_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _unlink(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _read_header(slot_dir: str) -> ExportHeader | None:
    try:
        with open(os.path.join(slot_dir, "header.json"), "rb") as f:
            return ExportHeader.from_json(f.read())
    except (OSError, FormatError):
        return None


def _link(root: str, area: str, tag: str, k: int) -> None:
    """Point <area>/<tag> at slot k atomically (rename over any older link)."""
    tmp = os.path.join(root, area, f".{tag}.{os.getpid()}.tmp")
    _unlink(tmp)
    os.symlink(os.path.join("..", "slots", f"s{k}"), tmp)
    os.replace(tmp, os.path.join(root, area, tag))


def _stream_append(slot_dir: str, doc: dict[str, Any]) -> None:
    """Append one line to the slot's stream log under LOCK_EX. The unlock is the release a reader's LOCK_SH pairs with."""
    line = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    fd = os.open(os.path.join(slot_dir, STREAM_LOG), os.O_WRONLY | os.O_APPEND)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            os.write(fd, line)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class _Stream:
    header: ExportHeader  # descriptor plus the layers completed so far
    state: str  # "writing" | "published" | "failed"
    error: str | None


def _read_stream(slot_dir: str) -> _Stream | None:
    """Parse a slot's stream log (read whole under LOCK_SH). None if absent or not a stream log this reader knows."""
    try:
        fd = os.open(os.path.join(slot_dir, STREAM_LOG), os.O_RDONLY)
    except OSError:
        return None
    chunks: list[bytes] = []
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            while block := os.read(fd, 1 << 16):
                chunks.append(block)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    lines = b"".join(chunks).split(b"\n")[:-1]  # a line without its newline is never complete
    if not lines:
        return None
    try:
        first = json.loads(lines[0])
        if not isinstance(first, dict) or first.get("stream") != STREAM_FORMAT:
            return None
        doc: dict[str, Any] = dict(first["header"])
        layers: list[Any] = []
        state, error = "writing", None
        for raw in lines[1:]:
            entry = json.loads(raw)
            if "layer" in entry:
                layers.append(entry["layer"])
            elif entry.get("state") in ("published", "failed"):
                state, error = str(entry["state"]), entry.get("error")
        doc["layers"] = sorted(layers, key=lambda x: int(x["index"]))
        return _Stream(ExportHeader.from_dict(doc), state, None if error is None else str(error))
    except (ValueError, KeyError, TypeError, FormatError):
        return None


@dataclass
class _SlotState:
    tag: str | None = None
    seq: int = 0
    published_at: float = 0.0
    inode: int = 0
    generation: int = 0
    stream_tag: str | None = None  # streaming mode: the tag this slot was begun for (its streaming/ link)
    reserved_seq: int = 0  # streaming mode: the seq fixed at begin() and kept by publish()
    stream_open: bool = False  # streaming mode: no "published" or "failed" line yet


# ----------------------------------------------------------------------------------------------- writer
class ExportStore:
    """The writer side. Exactly one per store root (enforced with writer.lock)."""

    def __init__(
        self,
        root: str,
        *,
        slots: int,
        hold_seconds: float = 120.0,
        failed_keep: int = 64,
        allow_login_uid: bool = False,
        clock: Callable[[], float] = time.time,
    ):
        if slots < 1:
            raise ValueError("slots must be >= 1")
        problem = check_ipc_safety(root)
        if problem and not allow_login_uid:
            raise UnsafeStoreLocation(problem)
        if problem:
            log.warning("store_location_unsafe", reason=problem)
        swap = swap_problem(store_mount(root))
        if swap:
            log.warning("store_swappable", reason=swap)
        self.root = root
        self.slots = slots
        self.hold_seconds = float(hold_seconds)
        self.failed_keep = max(1, failed_keep)
        self._clock = clock
        self._state = [_SlotState() for _ in range(slots)]
        self._writing: dict[int, int] = {}  # slot -> fd holding LOCK_EX
        self._writer_fd = -1
        self._layout()
        self._seq = max(time.time_ns() // 1000, self._recover() + 1)

    # -- layout, recovery, deletion detection
    def _layout(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.root)), exist_ok=True)
        for d in (
            self.root,
            os.path.join(self.root, "slots"),
            os.path.join(self.root, "tags"),
            os.path.join(self.root, "streaming"),
            os.path.join(self.root, "failed"),
        ):
            _mkdir(d)
        if self._writer_fd >= 0:
            os.close(self._writer_fd)
        fd = os.open(os.path.join(self.root, "writer.lock"), os.O_RDWR | os.O_CREAT, FILE_MODE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise StoreBusy(f"another writer holds {self.root}/writer.lock") from None
        self._writer_fd = fd
        marker = os.path.join(self.root, MARKER)
        if not os.path.exists(marker):
            _write_file_atomic(marker, json.dumps({"layout_version": LAYOUT_VERSION, "created_unix": time.time()}).encode())
        for k in range(self.slots):
            self._ensure_slot_dir(k, bump=False)

    def _ensure_slot_dir(self, k: int, *, bump: bool) -> None:
        d = self.slot_dir(k)
        _mkdir(d)
        lock = os.path.join(d, "lock")
        if not os.path.exists(lock):
            os.close(os.open(lock, os.O_RDONLY | os.O_CREAT, FILE_MODE))
            os.chmod(lock, FILE_MODE)
        ino = os.stat(d).st_ino
        s = self._state[k]
        if s.inode and s.inode != ino and bump:
            s.generation += 1
        s.inode = ino

    def _check_alive(self, k: int) -> None:
        """Detect files deleted under us (RemoveIPC, an operator's rm) and rebuild, invalidating held maps."""
        if not os.path.exists(os.path.join(self.root, MARKER)):
            log.error("store_deleted", root=self.root, hint="RemoveIPC or manual deletion; recreating the layout")
            for s in self._state:
                s.generation += 1
                s.inode = 0
                s.tag = None
            self._layout()
            return
        d = self.slot_dir(k)
        try:
            ino = os.stat(d).st_ino
        except FileNotFoundError:
            ino = -1
        s = self._state[k]
        if ino != s.inode:
            log.error("slot_deleted", slot=k, root=self.root)
            s.tag = None
            s.generation += 1
            s.inode = 0
            self._ensure_slot_dir(k, bump=False)

    def _recover(self) -> int:
        """Rebuild slot ownership from disk after a restart; drop tag links that no longer match. Returns max seq."""
        max_seq = 0
        for k in range(self.slots):
            h = _read_header(self.slot_dir(k))
            if h is not None:
                s = self._state[k]
                s.tag, s.seq, s.published_at = h.tag, h.seq, h.created_unix
                max_seq = max(max_seq, h.seq)
        tags_dir = os.path.join(self.root, "tags")
        for name in os.listdir(tags_dir):
            p = os.path.join(tags_dir, name)
            target = os.path.realpath(p)
            owner = next((k for k in range(self.slots) if os.path.realpath(self.slot_dir(k)) == target), None)
            if not os.path.islink(p) or owner is None or self._state[owner].tag != name:
                _unlink(p)
        # A restart ends every stream: a published one stays readable through tags/, an unfinished one is dropped (its
        # reader sees ExportNotFound and fails, so EXO prefills locally).
        for name in os.listdir(os.path.join(self.root, "streaming")):
            _unlink(os.path.join(self.root, "streaming", name))
        return max_seq

    # -- public writer API
    def slot_dir(self, k: int) -> str:
        return os.path.join(self.root, "slots", f"s{k}")

    def slot_generation(self, k: int) -> int:
        """Changes whenever slot k's files were recreated; holders of file maps must remap."""
        return self._state[k].generation

    def _reusable_rank(self, k: int, now: float) -> tuple[int, float] | None:
        """Sort key for reuse (lower first), or None if slot k must be kept."""
        if k in self._writing:
            return None
        s = self._state[k]
        if s.tag is None:
            return (0, 0.0)
        if os.path.exists(os.path.join(self.slot_dir(k), "consumed")):
            return (1, s.published_at)
        if now - s.published_at >= self.hold_seconds:
            return (2, s.published_at)
        return None

    def begin(self, tag: str, *, stream: ExportHeader | None = None) -> int:
        """Reserve a slot for ``tag`` and return its index. Never blocks; raises StoreFull.

        With ``stream`` (the export's descriptor: kind, tag, model, tokens, rope, producer; no layers) the export is
        written in streaming mode: its seq is fixed now, the descriptor goes to the slot's stream log, streaming/<tag>
        points at the slot, and the slot's flock is released so readers can read completed layers before publish().
        """
        if not is_valid_tag(tag):
            raise ValueError(f"invalid tag {tag!r}")
        if stream is not None and (stream.tag != tag or stream.error is not None or stream.layers):
            raise ValueError("a stream descriptor names this tag and carries no layers and no error")
        now = self._clock()
        for k in range(self.slots):
            self._check_alive(k)
        ranked = sorted((r, k) for k in range(self.slots) if (r := self._reusable_rank(k, now)) is not None)
        for _, k in ranked:
            fd = os.open(os.path.join(self.slot_dir(k), "lock"), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)  # a reader is streaming this slot right now
                continue
            self._writing[k] = fd
            s = self._state[k]
            d = os.path.realpath(self.slot_dir(k))
            for area, old in (("tags", s.tag), ("streaming", s.stream_tag)):
                if old is not None:
                    link = os.path.join(self.root, area, old)
                    if os.path.realpath(link) == d:
                        _unlink(link)
            for f in ("header.json", "consumed", STREAM_LOG):
                _unlink(os.path.join(self.slot_dir(k), f))
            s.tag, s.seq, s.published_at = None, 0, 0.0
            s.stream_tag, s.reserved_seq, s.stream_open = None, 0, False
            if stream is not None:
                try:
                    self._begin_stream(k, stream)
                except (OSError, ValueError) as exc:
                    self.abandon(k)
                    raise StoreError(f"slot {k}: cannot start a stream for {tag}: {exc}") from exc
            return k
        held = sum(1 for k in range(self.slots) if self._state[k].tag is not None)
        raise StoreFull(
            f"no reusable slot: {self.slots} slots, {held} held by exports younger than "
            f"{self.hold_seconds:.0f}s and not yet released, {len(self._writing)} being written"
        )

    def _begin_stream(self, k: int, stream: ExportHeader) -> None:
        """Called by begin() while it holds slot k's LOCK_EX: no reader can be inside the slot."""
        self._seq += 1
        h = stream.with_store_fields(seq=self._seq, slot=k, created_unix=self._clock())
        validate(h)
        d = self.slot_dir(k)
        _write_file_atomic(os.path.join(d, STREAM_LOG), b"")
        _stream_append(d, {"stream": STREAM_FORMAT, "header": h.to_dict()})
        _link(self.root, "streaming", h.tag, k)
        s = self._state[k]
        s.stream_tag, s.reserved_seq, s.stream_open = h.tag, h.seq, True
        fd = self._writing[k]  # readers may now take LOCK_SH; the slot stays ours through _writing
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        self._writing[k] = -1

    def stream_seq(self, k: int) -> int:
        """The seq slot k's stream will be published with (0 if slot k is not streaming)."""
        return self._state[k].reserved_seq if k in self._writing else 0

    def stream_layer(self, k: int, layer: LayerRef) -> None:
        """Announce that ``layer`` of slot k's stream is complete. Call only once its bytes are in the file: a reader
        may read them as soon as this returns. A layer is announced once per stream."""
        s = self._state[k]
        if k not in self._writing or not s.stream_open or s.stream_tag is None:
            raise StoreError(f"slot {k} is not streaming")
        _stream_append(self.slot_dir(k), {"layer": layer.to_dict()})

    def stream_failed(self, k: int, error: str) -> None:
        """End slot k's stream as failed, so a reader stops at its next poll (the caller also records the failure)."""
        s = self._state[k]
        if k in self._writing and s.stream_open:
            s.stream_open = False
            with contextlib.suppress(OSError):
                _stream_append(self.slot_dir(k), {"state": "failed", "error": error})

    def prepare_file(self, k: int, name: str, size: int) -> str:
        """Create (mode 0640) or grow slot k's file ``name`` to at least ``size`` bytes; never shrinks. Returns path."""
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad file name {name!r}")
        path = os.path.join(self.slot_dir(k), name)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, FILE_MODE)
        try:
            if os.fstat(fd).st_size < size:
                os.ftruncate(fd, size)
        finally:
            os.close(fd)
        return path

    def reserve(self, k: int, name: str, offset: int, length: int) -> None:
        """Allocate the backing pages of [offset, offset+length) of slot k's file before writing through a map.

        On tmpfs a write through a shared map into a page the filesystem cannot supply kills the writer with SIGBUS
        (the vLLM worker). posix_fallocate fails with ENOSPC instead, which becomes a clean StoreError (and a
        failure record). Pages already allocated cost nothing, so a reused slot only allocates what grew.
        """
        if length <= 0:
            return
        path = os.path.join(self.slot_dir(k), name)
        fd = os.open(path, os.O_RDWR)
        try:
            os.posix_fallocate(fd, offset, length)
        except OSError as exc:
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise StoreFull(f"store filesystem full: cannot allocate {length} bytes for {name} in slot {k}") from exc
            if exc.errno not in (errno.EOPNOTSUPP, errno.EINVAL):
                raise
        finally:
            os.close(fd)

    def write_bytes(self, k: int, name: str, offset: int, data: bytes | memoryview) -> None:
        """Plain write path (no torch): used by tests and by small exports."""
        path = self.prepare_file(k, name, offset + len(data))
        fd = os.open(path, os.O_WRONLY)
        try:
            os.pwrite(fd, data, offset)
        finally:
            os.close(fd)

    def publish(self, k: int, header: ExportHeader) -> ExportHeader:
        """Write header.json, link tags/<tag> -> slot k atomically, release the slot's write lock."""
        if k not in self._writing:
            raise StoreError(f"slot {k} was not begun")
        st = self._state[k]
        streaming = st.stream_open
        if streaming and header.tag != st.stream_tag:
            raise StoreError(f"slot {k} is streaming {st.stream_tag}, not {header.tag}")
        if not streaming:
            self._seq += 1
        h = header.with_store_fields(seq=st.reserved_seq if streaming else self._seq, slot=k, created_unix=self._clock())
        validate(h)
        if h.error is not None:
            raise ValueError("publish a failure with publish_failure()")
        d = self.slot_dir(k)
        for layer in h.layers:
            if os.path.getsize(os.path.join(d, layer.file)) < layer.nbytes:
                raise StoreError(f"slot {k}: {layer.file} shorter than its {layer.nbytes} bytes")
        _write_file_atomic(os.path.join(d, "header.json"), h.to_json())
        _link(self.root, "tags", h.tag, k)
        for other, s in enumerate(self._state):  # the same tag exported again: the older slot is free now
            if other != k and s.tag == h.tag:
                s.tag = None
        s = self._state[k]
        s.tag, s.seq, s.published_at = h.tag, h.seq, h.created_unix
        _unlink(os.path.join(self.root, "failed", f"{h.tag}.json"))
        if streaming:
            s.stream_open = False
            _stream_append(d, {"state": "published", "seq": h.seq})
        self._unlock(k)
        return h

    def abandon(self, k: int) -> None:
        """Give slot k back without publishing (the export failed part way)."""
        if k in self._writing:
            self.stream_failed(k, "abandoned by the writer")
            for f in ("header.json", "consumed"):
                _unlink(os.path.join(self.slot_dir(k), f))
            self._state[k].tag = None
            self._unlock(k)

    def publish_failure(self, header: ExportHeader) -> None:
        """Record that ``header.tag`` could not be exported, so a reader fails at once instead of timing out."""
        if header.error is None:
            raise ValueError("a failure record needs error set")
        self._seq += 1
        h = header.with_store_fields(seq=self._seq, slot=-1, created_unix=self._clock())
        validate(h)
        fdir = os.path.join(self.root, "failed")
        _write_file_atomic(os.path.join(fdir, f"{h.tag}.json"), h.to_json())
        records = sorted((e for e in os.scandir(fdir) if e.name.endswith(".json")), key=lambda e: e.stat().st_mtime)
        for e in records[: max(0, len(records) - self.failed_keep)]:
            _unlink(e.path)

    def _unlock(self, k: int) -> None:
        fd = self._writing.pop(k, -1)  # -1: a streaming slot, whose flock was released at begin()
        if fd >= 0:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def close(self) -> None:
        for k in list(self._writing):
            self.abandon(k)
        if self._writer_fd >= 0:
            os.close(self._writer_fd)
            self._writer_fd = -1


# ----------------------------------------------------------------------------------------------- reader
class OpenExport:
    """A published export held open with a shared lock; the writer will not reuse its slot until close()."""

    def __init__(self, header: ExportHeader, slot_dir: str, lock_fd: int):
        self.header = header
        self.slot_dir = slot_dir
        self._lock_fd = lock_fd

    def path(self, layer_index: int) -> str:
        return os.path.join(self.slot_dir, self.header.layer(layer_index).file)

    def read_range(self, layer_index: int, offset: int, nbytes: int) -> bytes:
        with open(self.path(layer_index), "rb") as f:
            data = os.pread(f.fileno(), nbytes, offset)
        if len(data) != nbytes:
            raise ExportNotFound(f"export {self.header.tag}: layer {layer_index} file is short")
        return data

    def read_blob(self, layer_index: int, role: str) -> bytes:
        layer = self.header.layer(layer_index)
        blob = next((b for b in layer.blobs if b.role == role), None)
        if blob is None:
            raise KeyError(f"layer {layer_index} has no blob {role!r}")
        return self.read_range(layer_index, blob.offset, blob.nbytes)

    def close(self) -> None:
        if self._lock_fd >= 0:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = -1

    def __enter__(self) -> OpenExport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ExportReader:
    """The read side (transfer service, or the EXO bridge on the same host). Needs read access only, except release()."""

    def __init__(self, root: str):
        self.root = root
        self._slots_dir = os.path.realpath(os.path.join(root, "slots"))

    def _slot_of(self, tag: str, area: str = "tags") -> str | None:
        if not is_valid_tag(tag):
            return None
        link = os.path.join(self.root, area, tag)
        if not os.path.islink(link):
            return None
        d = os.path.realpath(link)
        if os.path.dirname(d) != self._slots_dir:
            return None  # a link that escapes the slots directory is never followed
        return d

    def lookup(self, tag: str) -> ExportHeader:
        """The header of a published export. Raises ExportFailed for a failure record, else ExportNotFound."""
        d = self._slot_of(tag)
        if d is not None:
            h = _read_header(d)
            if h is not None and h.tag == tag:
                return h
        if is_valid_tag(tag):
            try:
                with open(os.path.join(self.root, "failed", f"{tag}.json"), "rb") as f:
                    raise ExportFailed(ExportHeader.from_json(f.read()))
            except (OSError, FormatError):
                pass
        raise ExportNotFound(f"no export {tag!r}")

    def open(self, tag: str, *, seq: int | None = None) -> OpenExport:
        """Hold export ``tag`` open (shared lock). Raises ExportNotFound if gone, reused, or being rewritten."""
        d = self._slot_of(tag)
        if d is None:
            self.lookup(tag)  # raises ExportFailed when there is a failure record
            raise ExportNotFound(f"no export {tag!r}")
        try:
            fd = os.open(os.path.join(d, "lock"), os.O_RDONLY)
        except FileNotFoundError:
            raise ExportNotFound(f"export {tag!r}: slot files are gone") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise ExportNotFound(f"export {tag!r}: its slot is being rewritten") from None
        h = _read_header(d)
        if h is None or h.tag != tag or (seq is not None and h.seq != seq):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise ExportNotFound(f"export {tag!r}: slot reused for a newer export")
        return OpenExport(h, d, fd)

    # -- streaming mode
    def stream_state(self, tag: str) -> StreamState:
        """Export ``tag`` while it is streamed (layers completed so far) or once published. Raises ExportFailed for a
        failed stream or a failure record, ExportNotFound when there is nothing yet (or it is gone)."""
        d = self._slot_of(tag, "streaming")
        if d is not None:
            st = _read_stream(d)
            if st is not None and st.header.tag == tag:
                if st.state == "failed":
                    raise ExportFailed(
                        ExportHeader(
                            kind=st.header.kind,
                            tag=tag,
                            model_id=st.header.model_id,
                            num_tokens=st.header.num_tokens,
                            error=st.error or "stream failed",
                        )
                    )
                if st.state == "writing":
                    return StreamState(st.header, published=False)
        return StreamState(self.lookup(tag), published=True)

    def open_stream(self, tag: str, seq: int) -> OpenExport:
        """Hold export ``tag`` (at ``seq``) open for reading, published or still streaming. Only layers listed in the
        returned header may be read: while streaming, those are the layers completed so far. Raises ExportNotFound if
        gone, reused, failed or being rewritten."""
        d = self._slot_of(tag, "streaming") or self._slot_of(tag)
        if d is None:
            self.lookup(tag)
            raise ExportNotFound(f"no export {tag!r}")
        try:
            fd = os.open(os.path.join(d, "lock"), os.O_RDONLY)
        except FileNotFoundError:
            raise ExportNotFound(f"export {tag!r}: slot files are gone") from None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise ExportNotFound(f"export {tag!r}: its slot is being rewritten") from None
        h = _read_header(d)
        if h is not None and h.tag == tag and h.seq == seq:
            return OpenExport(h, d, fd)
        st = _read_stream(d)  # read under the stream log's LOCK_SH, after the slot's LOCK_SH: see the module docstring
        if st is not None and st.header.tag == tag and st.header.seq == seq and st.state != "failed":
            return OpenExport(st.header, d, fd)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise ExportNotFound(f"export {tag!r}: slot reused, failed or not at seq {seq}")

    def release(self, tag: str, seq: int) -> bool:
        """Mark export ``tag`` (at ``seq``) consumed so its slot can be reused now. False if gone or not permitted."""
        d = self._slot_of(tag)
        if d is None:
            return False
        h = _read_header(d)
        if h is None or h.tag != tag or h.seq != seq:
            return False
        try:
            fd = os.open(os.path.join(d, "consumed"), os.O_WRONLY | os.O_CREAT, FILE_MODE)
        except PermissionError:
            return False
        except OSError as exc:
            if exc.errno == errno.EROFS:
                return False
            raise
        os.close(fd)
        return True

    def stats(self, hold_seconds: float | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"root_ok": os.path.exists(os.path.join(self.root, MARKER))}
        if not out["root_ok"]:
            return out
        slots = published = consumed = 0
        now = time.time()
        held = 0
        for e in os.scandir(os.path.join(self.root, "slots")):
            if not e.is_dir():
                continue
            slots += 1
            h = _read_header(e.path)
            if h is None:
                continue
            published += 1
            if os.path.exists(os.path.join(e.path, "consumed")):
                consumed += 1
            elif hold_seconds is not None and now - h.created_unix < hold_seconds:
                held += 1
        failed = sum(1 for e in os.scandir(os.path.join(self.root, "failed")) if e.name.endswith(".json"))
        cap = capacity_report(self.root, 0)
        out.update(
            slots=slots,
            published=published,
            consumed=consumed,
            failed_records=failed,
            fs_bytes=cap["fs_bytes"],
            fs_free_bytes=cap["fs_free_bytes"],
        )
        if hold_seconds is not None:
            out["held"] = held
        return out


# ----------------------------------------------------------------------------------------------- CLI
def paged_slot_bytes(*, layers: int, max_tokens: int, kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """Worst-case bytes of one paged_kv slot (files are pre-sized to max_tokens rows of K and V per layer)."""
    return layers * 2 * max_tokens * kv_heads * head_dim * dtype_bytes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kvhandoff-store", description="Inspect or check an export store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="location safety (RemoveIPC) and capacity for a planned configuration")
    c.add_argument("--root", required=True)
    c.add_argument("--slots", type=int, required=True)
    c.add_argument("--slot-bytes", type=int, required=True, help="largest export, e.g. from paged_slot_bytes()")
    c.add_argument("--uid", type=int, default=None, help="uid the writer will run as (default: this process)")
    c.add_argument(
        "--require-noswap",
        action="store_true",
        help="exit 1 if the store is on tmpfs without noswap (the idle store would be swapped out: cold exports)",
    )
    s = sub.add_parser("stats", help="slot usage")
    s.add_argument("--root", required=True)
    s.add_argument("--hold-seconds", type=float, default=None)
    i = sub.add_parser("init", help="create the store layout now (as the store's account), so health is green early")
    i.add_argument("--root", required=True)
    i.add_argument("--slots", type=int, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "init":
        ExportStore(a.root, slots=a.slots).close()
        print(json.dumps(ExportReader(a.root).stats(), indent=1))
        return 0
    if a.cmd == "stats":
        print(json.dumps(ExportReader(a.root).stats(a.hold_seconds), indent=1))
        return 0
    problem = check_ipc_safety(a.root, euid=a.uid)
    target = a.root if os.path.exists(a.root) else os.path.dirname(os.path.abspath(a.root))
    cap = capacity_report(target, a.slots * a.slot_bytes)
    try:
        st = os.stat(a.root)
        owner = {"uid": st.st_uid, "gid": st.st_gid, "mode": oct(stat.S_IMODE(st.st_mode))}
    except FileNotFoundError:
        owner = None
    disk = shutil.disk_usage(target)
    mount = store_mount(a.root)
    swap = swap_problem(mount)
    doc = {
        "root": a.root,
        "ipc_problem": problem,
        "capacity": cap,
        "owner": owner,
        "disk_total": disk.total,
        "mount": mount and {k: mount[k] for k in ("mount_point", "fstype", "noswap")},
        "swap_problem": swap,
    }
    print(json.dumps(doc, indent=1))
    return 1 if problem or not cap["fits"] or (swap and a.require_noswap) else 0


if __name__ == "__main__":
    sys.exit(main())
