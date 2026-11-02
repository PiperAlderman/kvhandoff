"""Paged-KV export (Llama-style attention): scheduler-side tracking, worker-side gather and layout normalisation,
and the slot writer. Lifted from the hardware-tested prototype connector (vLLM v0.30.0), with the store, the header
and failure handling replaced.

Scheduler side: a request is stored once, in the step that completes its prompt's prefill (under chunked prefill
with several prompts in flight, earlier steps have not computed the later chunks yet). The tracker follows each
tagged request's block ids across steps (new_block_ids appended, or replaced for a resumed request).

Worker side: save_kv_layer gathers the request's rows from the paged layer and normalises them to [2, T, H, D] (K
block then V block), which is also the file layout, so the reader slices without converting.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from kvhandoff import __version__
from kvhandoff.interchange import KIND_PAGED, ExportHeader, LayerRef, RopeInfo, paged_layer
from kvhandoff.logs import get_logger
from kvhandoff.producer.maps import SlotMaps
from kvhandoff.store import ExportStore, StoreError
from kvhandoff.tags import requested_tag, safe_tag

log = get_logger("producer.paged")


def _vmstat_swap() -> tuple[int, int]:
    """Host-wide (pswpin, pswpout) page counts from /proc/vmstat, or (-1, -1) where unavailable. Timing log only."""
    try:
        with open("/proc/vmstat", encoding="ascii") as f:
            d = dict(line.split()[:2] for line in f if line.startswith(("pswpin ", "pswpout ")))
        return int(d["pswpin"]), int(d["pswpout"])
    except (OSError, KeyError, ValueError):
        return -1, -1


_TORCH_TO_WIRE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


class UnsupportedLayout(ValueError):
    pass


# ----------------------------------------------------------------------------------------------- scheduler side
@dataclass
class StoreJob:
    """One request to store this step. Small and picklable: the worker builds the slot mapping from block ids."""

    req_id: str
    tag: str
    num_tokens: int
    block_ids: list[int]
    is_store: bool = True  # vLLM's ExampleConnector convention; start_load_kv skips stores


@dataclass
class _Tracked:
    tag: str
    num_tokens: int
    blocks: list[int] = field(default_factory=list[int])


def _params_of(new_req: Any) -> Mapping[str, Any] | None:
    sp = getattr(new_req, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) or {}
    params = extra.get("kv_transfer_params") if isinstance(extra, dict) else None
    return params if isinstance(params, dict) else None  # pyright: ignore[reportUnknownVariableType]


class RequestTracker:
    """Scheduler-side state of the prototype connector's build_connector_meta, without vLLM types."""

    def __init__(self, block_size: int, *, export_all: bool = False):
        self.block_size = block_size
        self.export_all = export_all
        self._track: dict[str, _Tracked] = {}

    def __len__(self) -> int:
        return len(self._track)

    def drop(self, req_id: str) -> None:
        """The request finished or was aborted before its prefill completed."""
        self._track.pop(req_id, None)

    def _maybe_emit(self, out: list[StoreJob], req_id: str, done_tokens: int) -> None:
        t = self._track[req_id]
        if done_tokens < t.num_tokens:
            return
        del self._track[req_id]
        if len(t.blocks) * self.block_size < t.num_tokens:
            log.error("not_enough_blocks", req_id=req_id, blocks=len(t.blocks), tokens=t.num_tokens)
            return
        out.append(StoreJob(req_id=req_id, tag=t.tag, num_tokens=t.num_tokens, block_ids=list(t.blocks)))

    def step(self, scheduler_output: Any) -> list[StoreJob]:
        """Feed one SchedulerOutput (duck-typed: scheduled_new_reqs, scheduled_cached_reqs, num_scheduled_tokens)."""
        out: list[StoreJob] = []
        sched: Mapping[str, int] = scheduler_output.num_scheduled_tokens
        for new_req in scheduler_output.scheduled_new_reqs:
            tag = requested_tag(_params_of(new_req))
            if tag is None and not self.export_all:
                continue
            req_id = str(new_req.req_id)
            self._track[req_id] = _Tracked(
                tag=safe_tag(tag or "", req_id),
                num_tokens=len(new_req.prompt_token_ids or []),
                blocks=list(new_req.block_ids[0]),
            )
            self._maybe_emit(out, req_id, int(new_req.num_computed_tokens) + int(sched.get(req_id, 0)))
        cached = scheduler_output.scheduled_cached_reqs
        resumed: Iterable[str] = getattr(cached, "resumed_req_ids", ()) or ()
        resumed_set = set(resumed)
        for i, req_id in enumerate(cached.req_ids):
            t = self._track.get(req_id)
            if t is None:
                continue
            nb = cached.new_block_ids[i]
            if nb is not None:
                if req_id in resumed_set:  # a resumed request's block ids replace the old ones
                    t.blocks = list(nb[0])
                else:
                    t.blocks += list(nb[0])
            self._maybe_emit(out, req_id, int(cached.num_computed_tokens[i]) + int(sched.get(req_id, 0)))
        return out


# ----------------------------------------------------------------------------------------------- worker side
def slot_mapping(block_ids: Sequence[int], num_tokens: int, block_size: int) -> torch.Tensor:
    """Flat slot index (block * block_size + offset) of every prompt position."""
    pos = torch.arange(num_tokens)
    blocks = torch.as_tensor(list(block_ids), dtype=torch.long)
    return blocks[pos // block_size] * block_size + pos % block_size


def gather(kv_layer: torch.Tensor, slots: torch.Tensor, block_size: int) -> torch.Tensor:
    """Rows of the request from a paged layer whose dims are [blocks, X, block_size, ...]: returns [T, X, ...].

    vLLM v0.30.0 hands FlashInfer layers as [blocks, 2, block_size, H, D] (giving [T, 2, H, D]); vLLM main's
    standardised view is [blocks, H, block_size, C] (giving [T, H, C], C = 2*D when K and V are packed).
    """
    slots = slots.to(kv_layer.device, non_blocking=True)
    return kv_layer[slots // block_size, :, slots % block_size]


def to_kv_split(kv: torch.Tensor, head_size: int) -> torch.Tensor:
    """Normalise gathered rows to [2, T, H, D] (contiguous). Raises UnsupportedLayout for anything else (MLA etc.)."""
    if kv.dim() == 4 and int(kv.shape[1]) == 2:  # [T, 2, H, D]
        return kv.permute(1, 0, 2, 3).contiguous()
    if kv.dim() == 3 and head_size and int(kv.shape[2]) == 2 * head_size:  # [T, H, 2*D], K|V on the last axis
        return torch.stack((kv[..., :head_size], kv[..., head_size:]), 0).contiguous()
    raise UnsupportedLayout(
        f"gathered KV has shape {tuple(kv.shape)} (head size {head_size}); expected [T, 2, H, D] "
        "or [T, H, 2*D]. MLA and other packed layouts are not exported by the paged connector"
    )


@dataclass
class _Pending:
    job: StoreJob
    slot: int
    layers: dict[int, LayerRef] = field(default_factory=dict[int, LayerRef])
    failed: str | None = None
    recorded: bool = False  # streaming: the failure record was written as soon as the job failed
    t0: float = field(default_factory=time.perf_counter)
    reserve_s: float = 0.0  # timing log: time in store.reserve (posix_fallocate) for this export
    copy_s: float = 0.0  # timing log: time in the device-to-host copies (host-synchronous for a pageable map)
    swap0: tuple[int, int] = (-1, -1)  # timing log: (pswpin, pswpout) when the export began
    prefault_s: float = 0.0  # time spent populating slot map pages for this export


@dataclass(frozen=True)
class PagedExportInfo:
    model_id: str
    num_model_layers: int
    max_tokens: int  # rows each layer file is sized for (max_model_len + margin)
    rope: RopeInfo
    producer: dict[str, Any]


def _device_fence(t: torch.Tensor) -> None:
    """Make sure every copy out of ``t``'s device is complete before a reader is told about it. ``copy_`` into the
    (pageable) slot map is already host-synchronous with non_blocking=False, so this costs nothing today; it keeps the
    stream-log line after the bytes if the copy is ever made non-blocking."""
    if t.is_cuda:
        torch.cuda.current_stream(t.device).synchronize()


class PagedWriter:
    """Writes stores of one step into slots (vLLM TP rank 0 only) and publishes them at wait_for_save.

    With ``stream_layers`` each export is begun in the store's streaming mode, and every layer is announced in the
    slot's stream log as soon as its bytes are written, so a reader can send it on while later layers are computed.
    A job that fails part way ends its stream and gets its failure record at once, not at wait_for_save.

    With ``prefault_slot_maps`` (default on) the rows a layer is about to receive are populated in this process's page
    tables on the CPU (``SlotMaps.prefault``) before the device-to-host copy, once per map and row range. Without it,
    the first export into rows a fresh process has not written lets the GPU copy fault every page itself, which on a
    GB10 costs about 30 to 60 us per 4 KiB page (measured: 177.8 s against 61.8 s at 32k).
    """

    def __init__(
        self,
        store: ExportStore,
        info: PagedExportInfo,
        *,
        timing_log: bool = False,
        stream_layers: bool = False,
        direct_plane_copy: bool = True,
        prefault_slot_maps: bool = True,
    ):
        self.store = store
        self.maps = SlotMaps(store)
        self.info = info
        self.timing_log = timing_log
        self.stream_layers = stream_layers
        self.direct_plane_copy = direct_plane_copy
        self.prefault_slot_maps = prefault_slot_maps
        self._pending: dict[str, _Pending] = {}
        self._dead: set[str] = set()  # tags that could not get a slot this step

    def _fail(self, job: StoreJob, error: str) -> None:
        self.store.publish_failure(
            ExportHeader(
                kind=KIND_PAGED,
                tag=job.tag,
                model_id=self.info.model_id,
                num_tokens=job.num_tokens,
                error=error,
                producer=dict(self.info.producer, req_id=job.req_id),
            )
        )
        log.error("export_failed", tag=job.tag, req_id=job.req_id, error=error)

    def _header(self, job: StoreJob, layers: tuple[LayerRef, ...] = (), extra: dict[str, Any] | None = None) -> ExportHeader:
        return ExportHeader(
            kind=KIND_PAGED,
            tag=job.tag,
            model_id=self.info.model_id,
            num_tokens=job.num_tokens,
            layers=layers,
            num_model_layers=self.info.num_model_layers,
            rope=self.info.rope,
            producer=dict(self.info.producer, req_id=job.req_id, package=__version__),
            extra=extra or {},
        )

    def _pending_for(self, job: StoreJob) -> _Pending | None:
        p = self._pending.get(job.tag)
        if p is not None or job.tag in self._dead:
            return p
        try:
            k = self.store.begin(job.tag, stream=self._header(job) if self.stream_layers else None)
        except StoreError as exc:  # StoreFull, or (streaming) the stream log could not be started
            self._dead.add(job.tag)
            self._fail(job, f"{type(exc).__name__}: {exc}")
            return None
        p = self._pending[job.tag] = _Pending(job=job, slot=k)
        if self.timing_log:
            p.swap0 = _vmstat_swap()
        return p

    def _set_failed(self, p: _Pending, error: str) -> None:
        if p.failed is not None:
            return
        p.failed = error
        if self.stream_layers:  # a reader may be streaming this export now: stop it at its next poll
            self.store.stream_failed(p.slot, error)
            self._fail(p.job, error)
            p.recorded = True

    def mark_failed(self, job: StoreJob, error: str) -> None:
        """The job cannot be exported (e.g. unsupported layout); finish_step turns it into a failure record."""
        p = self._pending_for(job)
        if p is not None:
            self._set_failed(p, error)

    def write_layer(self, job: StoreJob, layer_index: int, layer_name: str, kv_split: torch.Tensor) -> None:
        """kv_split: [2, T, H, D] for this job (after any TP gather and dtype cast)."""
        p = self._pending_for(job)
        if p is None or p.failed:
            return
        try:
            if self.stream_layers and layer_index in p.layers:  # a reader may already have sent the first copy
                raise ValueError(f"layer {layer_index} written twice in one streamed export")
            if kv_split.dim() != 4 or int(kv_split.shape[0]) != 2:
                raise UnsupportedLayout(f"expected [2, T, H, D], got {tuple(kv_split.shape)}")
            t, h, d = (int(x) for x in kv_split.shape[1:])
            if t != job.num_tokens:
                raise ValueError(f"layer {layer_index}: {t} rows gathered, prompt has {job.num_tokens}")
            if t > self.info.max_tokens:
                raise ValueError(f"{t} tokens exceed the {self.info.max_tokens} rows a slot file holds")
            wire = _TORCH_TO_WIRE.get(kv_split.dtype)
            if wire is None:
                raise UnsupportedLayout(f"KV dtype {kv_split.dtype} is not one EXO's wire carries; set out_dtype")
            cap = self.info.max_tokens
            fname = f"L{layer_index}.bin"
            dst = self.maps.typed_map(p.slot, fname, (2, cap, h, d), kv_split.dtype)
            row = h * d * kv_split.element_size()
            t_rs = time.perf_counter()
            self.store.reserve(p.slot, fname, 0, t * row)  # K rows, then V rows: never SIGBUS on a full tmpfs
            self.store.reserve(p.slot, fname, cap * row, t * row)
            p.reserve_s += time.perf_counter() - t_rs
            if self.prefault_slot_maps:  # after reserve: the pages exist, so populating them never allocates or fails
                t_pf = time.perf_counter()
                self.maps.prefault(p.slot, fname, 0, t * row)
                self.maps.prefault(p.slot, fname, cap * row, t * row)
                p.prefault_s += time.perf_counter() - t_pf
            t_cp = time.perf_counter()
            if self.direct_plane_copy:
                # K rows and V rows are each one contiguous run of the file, so each copy_ is a single same-dtype
                # device-to-host memcpy straight into the map (the prototype's pattern, kv_split is contiguous).
                dst[0, :t].copy_(kv_split[0])
                dst[1, :t].copy_(kv_split[1])
            else:
                # One copy into the strided [2, :t] view (it skips the cap - t unused rows between K and V). With a
                # CUDA source PyTorch stages that through a fresh host buffer of the whole view, then copies it in
                # again on the CPU: the cause of a measured 1.5x slower prefill.
                dst[:, :t].copy_(kv_split)  # GPU -> shared host map (same physical memory on GB10)
            p.copy_s += time.perf_counter() - t_cp
            ref = paged_layer(layer_index, layer_name, num_tokens=t, heads=h, head_dim=d, dtype=wire, row_capacity=cap)
            if self.stream_layers:
                _device_fence(kv_split)  # the bytes are complete before the stream-log line that announces them
                self.store.stream_layer(p.slot, ref)
            p.layers[layer_index] = ref
        except (ValueError, StoreError, OSError, RuntimeError) as exc:
            self._set_failed(p, f"{type(exc).__name__}: {exc}")

    def finish_step(self) -> list[ExportHeader]:
        """Publish everything written this step (call after the device is synchronised)."""
        published: list[ExportHeader] = []
        for tag, p in self._pending.items():
            if p.failed is None and not p.layers:
                self._set_failed(p, "no layers were written")
            if p.failed is not None:
                self.store.abandon(p.slot)
                if not p.recorded:
                    self._fail(p.job, p.failed)
                continue
            layers = tuple(p.layers[i] for i in sorted(p.layers))
            hdr = self._header(p.job, layers, {"export_seconds": round(time.perf_counter() - p.t0, 4)})
            try:
                published.append(self.store.publish(p.slot, hdr))
            except (StoreError, ValueError, OSError) as exc:
                self.store.abandon(p.slot)
                self._fail(p.job, f"{type(exc).__name__}: {exc}")
                continue
            if self.timing_log:
                swap1 = _vmstat_swap()
                log.info(
                    "exported",
                    tag=tag,
                    slot=p.slot,
                    tokens=p.job.num_tokens,
                    layers=len(layers),
                    seconds=hdr.extra["export_seconds"],
                    prefault_seconds=round(p.prefault_s, 4),
                    reserve_seconds=round(p.reserve_s, 4),
                    copy_seconds=round(p.copy_s, 4),
                    pswpin=swap1[0] - p.swap0[0] if min(swap1[0], p.swap0[0]) >= 0 else None,
                    pswpout=swap1[1] - p.swap0[1] if min(swap1[1], p.swap0[1]) >= 0 else None,
                )
        self._pending.clear()
        self._dead.clear()
        return published
