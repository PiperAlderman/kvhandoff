"""Shared file maps for slot files, created once and reused (a fresh tmpfs page on a GB10 (DGX Spark) is slow to fault in;
a lesson from the prototype). Maps are keyed by the slot's generation, so files the store had to recreate (they
were deleted under us) are remapped instead of written into while unlinked.

Pre-faulting (``prefault``): a new map has no page-table entries in this process, even over tmpfs pages another
process (or an earlier map) already wrote. On a GB10, a device-to-host copy that lands on such a page costs about
30 to 60 us per 4 KiB page (measured: the first 32k export in a fresh vLLM process took 177.8 s against
61.8 s for the second), against well under 1 us for a CPU write fault. ``prefault`` asks the kernel to install
writable entries for a byte range first (madvise MADV_POPULATE_WRITE, Linux 5.14+), once per range and map: a range
already covered costs nothing, a longer one only populates what grew. It never fails the caller: on an error it logs
once and the copy faults the pages in itself, as before.
"""

from __future__ import annotations

import ctypes
import errno
import os
from dataclasses import dataclass, field
from typing import Final

import torch

from kvhandoff.logs import get_logger
from kvhandoff.store import ExportStore

log = get_logger("producer.maps")

_MIB = 1 << 20
_ROUND = 64 * _MIB
MADV_POPULATE_WRITE: Final = 23  # include/uapi/asm-generic/mman-common.h (arm64 and x86-64 alike), Linux 5.14+
PAGE: Final = os.sysconf("SC_PAGE_SIZE")


def round_capacity(nbytes: int) -> int:
    """At least 1 MiB; above 64 MiB rounded up to a 64 MiB multiple, so a slightly longer prompt reuses the map."""
    cap = max(nbytes, _MIB)
    if cap > _ROUND:
        cap = (cap + _ROUND - 1) // _ROUND * _ROUND
    return cap


_libc: ctypes.CDLL | None = None


def madvise(addr: int, length: int, advice: int) -> int:
    """madvise(2) on [addr, addr+length) (addr page-aligned). Returns 0, or the errno on failure."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        _libc.madvise.restype = ctypes.c_int
    if _libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)) == 0:
        return 0
    return ctypes.get_errno() or errno.EIO


@dataclass
class _Map:
    generation: int
    tensor: torch.Tensor
    populated: dict[int, int] = field(default_factory=dict[int, int])  # range start offset -> bytes populated from it


class SlotMaps:
    def __init__(self, store: ExportStore):
        self.store = store
        self._maps: dict[tuple[int, str], _Map] = {}
        self.prefault_enabled = True  # turned off for the process after a madvise the kernel does not support
        self.prefault_calls = 0  # madvise calls made (tests and the timing log)
        self._warned = False

    def bytes_map(self, slot: int, name: str, nbytes: int, *, exact: bool = False) -> torch.Tensor:
        """A uint8 tensor over slot file ``name`` of at least ``nbytes`` (exactly ``nbytes`` when ``exact``)."""
        gen = self.store.slot_generation(slot)
        hit = self._maps.get((slot, name))
        if (
            hit is not None
            and hit.generation == gen
            and hit.tensor.numel() >= nbytes
            and (not exact or hit.tensor.numel() == nbytes)
        ):
            return hit.tensor
        cap = nbytes if exact else round_capacity(nbytes)
        path = self.store.prepare_file(slot, name, cap)
        m = torch.from_file(path, shared=True, size=cap, dtype=torch.uint8)
        self._maps[(slot, name)] = _Map(gen, m)  # a new map has no page-table entries: populated starts empty
        return m

    def typed_map(self, slot: int, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A tensor of ``shape`` and ``dtype`` over slot file ``name`` (sized exactly; reused while shape stays)."""
        n = 1
        for s in shape:
            n *= s
        nbytes = n * torch.empty((), dtype=dtype).element_size()
        return self.bytes_map(slot, name, nbytes, exact=True).view(dtype).view(shape)

    def prefault(self, slot: int, name: str, offset: int, nbytes: int) -> int:
        """Install writable page-table entries for bytes [offset, offset+nbytes) of the current map of slot file
        ``name`` (map it first). Ranges are tracked by their start ``offset``: a call that a previous one from the same
        start already covered does nothing. Call only on bytes whose backing pages exist (``ExportStore.reserve``),
        or the kernel refuses the range (EFAULT) instead of allocating it. Returns the bytes newly populated (0 when
        nothing was needed or it failed). Never raises."""
        m = self._maps.get((slot, name))
        if not self.prefault_enabled or m is None or nbytes <= 0:
            return 0
        done = m.populated.get(offset, 0)
        if nbytes <= done:
            return 0
        size = m.tensor.numel()
        begin, end = offset + done, min(offset + nbytes, size)
        if begin >= end:
            return 0
        base = m.tensor.data_ptr()  # the mmap base: page-aligned
        start = (base + begin) // PAGE * PAGE
        stop = min((base + end + PAGE - 1) // PAGE * PAGE, base + (size + PAGE - 1) // PAGE * PAGE)
        self.prefault_calls += 1
        err = madvise(start, stop - start, MADV_POPULATE_WRITE)
        if err == 0:
            m.populated[offset] = end - offset
            return end - begin
        if err in (errno.EINVAL, errno.ENOSYS, errno.EPERM):  # kernel before 5.14, or a map it will not populate
            self.prefault_enabled = False
        if not self._warned:
            self._warned = True
            log.warning(
                "prefault_failed",
                slot=slot,
                file=name,
                error=errno.errorcode.get(err, str(err)),
                disabled=not self.prefault_enabled,
                hint="the copy faults the pages in itself (slower first export into a new map)",
            )
        return 0
