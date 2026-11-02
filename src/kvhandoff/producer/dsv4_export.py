"""DeepSeek V4 export: request bookkeeping and per-request export, with no vLLM import.

Lifted from the hardware-tested prototype connector (TP4 on four DGX Sparks, vLLM 0.23.1rc1.dev1053). Why the export
runs from get_finished and not save_kv_layer, and why the connector must be SupportsHMA, is in
kvhandoff/vllm/dsv4_connector.py.

Scheduler side (SchedulerQueue): request_finished_all_groups queues one job per finished, export-tagged request and
asks vLLM to delay freeing its blocks; build_connector_meta hands the queue to every worker in the next step.
Worker side (FinishedTracker): every TP rank reports a job's request id once it has also seen that id in
finished_req_ids (vLLM requires it); only rank 0 exports (DSV4Exporter), once per job.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from kvhandoff import __version__
from kvhandoff.interchange import KIND_DSV4, BlobRef, ExportHeader, LayerRef, RopeInfo
from kvhandoff.logs import get_logger
from kvhandoff.producer import dsv4_convert as cv
from kvhandoff.producer.maps import SlotMaps
from kvhandoff.store import ExportStore, StoreError, StoreFull
from kvhandoff.tags import requested_tag, safe_tag

log = get_logger("producer.dsv4")

# Layer names as registered by DeepseekV4Attention and its sub-caches (nvidia/model.py:807-809 prefix "...layers.{i}.attn";
# attention.py:294-300 ".swa_cache", 314-322 compressor, 735-741 ".indexer.k_cache"; compressor.py:237-242 ".state_cache").
_NAME = re.compile(r"layers\.(\d+)\.attn(\..+)?$")
_SUFFIX: dict[str | None, str] = {
    ".swa_cache": "swa",
    None: "pool",  # the attention layer itself owns the compressed KV cache (attention.py:605-624)
    ".compressor.state_cache": "state",
    ".indexer.k_cache": "idx_pool",
    ".indexer.compressor.state_cache": "idx_state",
}

# DeepSeek-V4-Flash's values; only used when the checkpoint's config.json cannot be read.
ROPE_DEFAULTS: dict[str, Any] = {
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "rope_scaling": {"factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
}
SEEN_CAP = 4096


@dataclass
class DSV4Job:
    req_id: str
    tag: str
    tokens: int
    block_ids: list[list[int]]  # per kv-cache group


# ----------------------------------------------------------------------------------------------- scheduler side
class SchedulerQueue:
    def __init__(self, *, export_all: bool = False):
        self.export_all = export_all
        self._pending: list[DSV4Job] = []

    def on_finished(
        self,
        req_id: str,
        num_prompt_tokens: int,
        kv_transfer_params: Mapping[str, Any] | None,
        block_ids: Iterable[Iterable[int]],
    ) -> bool:
        """True if the request is queued for export (vLLM must then delay freeing its blocks)."""
        tag = requested_tag(kv_transfer_params)
        if tag is None and not self.export_all:
            return False
        if num_prompt_tokens <= 0:  # max_tokens=1: the sampled token's KV is never computed, prompt tokens only
            return False
        self._pending.append(
            DSV4Job(
                req_id=req_id,
                tag=safe_tag(tag or "", req_id),
                tokens=int(num_prompt_tokens),
                block_ids=[list(b) for b in block_ids],
            )
        )
        return True

    def take(self) -> list[DSV4Job]:
        jobs, self._pending = self._pending, []
        return jobs


# ----------------------------------------------------------------------------------------------- worker side
class FinishedTracker:
    """Which job request ids a worker may report in get_finished: done here AND seen in finished_req_ids."""

    def __init__(self, cap: int = SEEN_CAP):
        self.cap = cap
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._done: set[str] = set()

    def step(self, finished_req_ids: Iterable[str], done_now: Iterable[str]) -> set[str] | None:
        for r in finished_req_ids:
            self._seen[r] = None
        while len(self._seen) > self.cap:
            self._seen.popitem(last=False)
        self._done.update(done_now)
        ready = {r for r in self._done if r in self._seen}
        self._done -= ready
        for r in ready:
            self._seen.pop(r, None)
        return ready or None

    @property
    def seen_size(self) -> int:
        return len(self._seen)

    @property
    def pending(self) -> set[str]:
        return set(self._done)


def index_cache_names(names: Iterable[str]) -> dict[int, dict[str, str]]:
    """{layer: {"swa"|"pool"|"state"|"idx_pool"|"idx_state": registered cache name}}"""
    by_layer: dict[int, dict[str, str]] = {}
    for name in names:
        m = _NAME.search(name)
        if m and m.group(2) in _SUFFIX:
            by_layer.setdefault(int(m.group(1)), {})[_SUFFIX[m.group(2)]] = name
    return by_layer


def rope_config_from_checkpoint(model_dir: str) -> dict[str, Any]:
    """RoPE parameters from the checkpoint's config.json (vLLM's rope code mutates hf_config per layer)."""
    cfg = dict(ROPE_DEFAULTS)
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        for k in cfg:
            if raw.get(k) is not None:
                cfg[k] = raw[k]
    except (OSError, ValueError):
        log.warning("rope_config_defaults", model_dir=model_dir)
    return cfg


@dataclass(frozen=True)
class DSV4ModelInfo:
    model_id: str
    num_layers: int
    compress_ratios: tuple[int, ...]
    rope_cfg: dict[str, Any]
    producer: dict[str, Any] = field(default_factory=dict[str, Any])


class DSV4Exporter:
    """Exports one finished request into a slot (TP rank 0). Failures become failure records, never exceptions."""

    def __init__(
        self,
        store: ExportStore,
        info: DSV4ModelInfo,
        *,
        rerope: str = "ratio0",
        rope_check_layers: int = 3,
        timing_log: bool = False,
    ):
        self.store = store
        self.maps = SlotMaps(store)
        self.info = info
        self.rerope = rerope
        self.rope_check_layers = rope_check_layers
        self.timing_log = timing_log
        rope = info.rope_cfg
        self.plain = cv.mlx_inv_freq(cv.ROPE, float(rope["rope_theta"]), None)
        self.comp_yarn = cv.mlx_inv_freq(cv.ROPE, float(rope["compress_rope_theta"]), rope["rope_scaling"])

    def rope_info(self) -> RopeInfo:
        r = self.info.rope_cfg
        return RopeInfo(
            style="gptj",
            applied_by="mixed",
            theta=float(r["rope_theta"]),
            scaling=r["rope_scaling"],
            details={
                "ratio0_layers": "mlx" if self.rerope == "ratio0" else "vllm",
                "compressed_layers": "vllm (identical to mlx YaRN)",
                "compress_rope_theta": float(r["compress_rope_theta"]),
            },
        )

    def _fail(self, job: DSV4Job, error: str) -> None:
        self.store.publish_failure(
            ExportHeader(
                kind=KIND_DSV4,
                tag=job.tag,
                model_id=self.info.model_id,
                num_tokens=job.tokens,
                error=error,
                producer=dict(self.info.producer, req_id=job.req_id),
            )
        )
        log.error("export_failed", tag=job.tag, req_id=job.req_id, error=error)

    def export(
        self, job: DSV4Job, kv_caches: Mapping[str, torch.Tensor], group_of: Mapping[str, int], modules: Mapping[str, Any]
    ) -> ExportHeader | None:
        t0 = time.perf_counter()
        try:
            k = self.store.begin(job.tag)
        except StoreFull as exc:
            self._fail(job, f"StoreFull: {exc}")
            return None
        try:
            layers, rope_check = self._export_layers(job, k, kv_caches, group_of, modules)
            hdr = ExportHeader(
                kind=KIND_DSV4,
                tag=job.tag,
                model_id=self.info.model_id,
                num_tokens=job.tokens,
                layers=tuple(layers),
                num_model_layers=self.info.num_layers,
                rope=self.rope_info(),
                producer=dict(self.info.producer, req_id=job.req_id, package=__version__),
                extra={"rope_check_max_abs": rope_check, "export_seconds": round(time.perf_counter() - t0, 4)},
            )
            out = self.store.publish(k, hdr)
        except (ValueError, KeyError, RuntimeError, StoreError, OSError) as exc:
            self.store.abandon(k)
            self._fail(job, f"{type(exc).__name__}: {exc}")
            return None
        if self.timing_log:
            log.info(
                "exported",
                tag=job.tag,
                req_id=job.req_id,
                slot=k,
                tokens=job.tokens,
                layers=len(out.layers),
                seconds=out.extra["export_seconds"],
                bytes=out.total_bytes(),
            )
        return out

    def _export_layers(
        self, job: DSV4Job, k: int, kv: Mapping[str, torch.Tensor], groups: Mapping[str, int], mods: Mapping[str, Any]
    ) -> tuple[list[LayerRef], dict[str, float]]:
        T = int(job.tokens)
        L = self.info.num_layers
        ratios = self.info.compress_ratios
        by_layer = index_cache_names(kv)
        layers: list[LayerRef] = []
        rope_check: dict[str, float] = {}
        for i in range(L):
            names = by_layer.get(i, {})
            if "swa" not in names:
                raise RuntimeError(f"layer {i}: no SWA cache registered (have {sorted(names)})")
            attn_name = names["swa"][: -len(".swa_cache")]
            attn = mods.get(attn_name)
            if attn is None:
                raise RuntimeError(f"layer {i}: attention module {attn_name} not in the forward context")
            src = cv.LayerSources(ratios[i])

            def put(kind: str, attr: str, i: int = i, names: dict[str, str] = names, src: cv.LayerSources = src) -> None:
                name = names.get(kind)
                if name is None:
                    raise RuntimeError(f"layer {i} (ratio {ratios[i]}): missing {kind} cache")
                setattr(src, attr, kv[name])
                setattr(src, attr + "_blocks", job.block_ids[groups[name]])

            put("swa", "swa")
            src.cos_sin = attn.rotary_emb.cos_sin_cache
            if ratios[i]:
                put("pool", "pool")
                put("state", "state")
                src.ape = attn.compressor.ape.data
            if ratios[i] == 4:
                put("idx_pool", "idx_pool")
                put("idx_state", "idx_state")
                src.idx_ape = attn.indexer.compressor.ape.data
            want = self.comp_yarn if ratios[i] else self.plain
            if i < self.rope_check_layers and src.cos_sin is not None:  # distance of vLLM's rope to mlx's at T-1
                p = T - 1
                ang = p * want.to(src.cos_sin.device)
                cs = src.cos_sin[p].to(torch.float32)
                rope_check[str(i)] = float(torch.max(torch.abs(cs - torch.cat([torch.cos(ang), torch.sin(ang)]))))
            if self.rerope == "ratio0" and ratios[i] == 0:
                src.rerope_inv_freq = self.plain
            layer = self._write_layer(k, i, attn_name, cv.convert_layer(src, T))
            layers.append(
                LayerRef(
                    index=layer.index,
                    name=layer.name,
                    file=layer.file,
                    nbytes=layer.nbytes,
                    blobs=layer.blobs,
                    attrs={"compress_ratio": ratios[i]},
                )
            )
        return layers, rope_check

    def _write_layer(self, k: int, idx: int, name: str, blobs: list[cv.Blob]) -> LayerRef:
        """Write one layer's blobs back to back into slot k's L<idx>.bin (as the prototype's store did)."""
        refs: list[BlobRef] = []
        flat: list[torch.Tensor] = []
        off = 0
        for role, dtype, shape, t in blobs:
            b = cv.blob_bytes(t)
            refs.append(BlobRef(role=role, dtype=dtype, shape=shape, offset=off, nbytes=int(b.numel())))
            off += int(b.numel())
            flat.append(b)
        fname = f"L{idx}.bin"
        dst = self.maps.bytes_map(k, fname, off)
        self.store.reserve(k, fname, 0, off)  # ENOSPC becomes a failure record instead of SIGBUS
        o = 0
        for b in flat:
            dst[o : o + b.numel()].copy_(b)  # device -> shared host map (same physical memory on GB10)
            o += int(b.numel())
        return LayerRef(index=idx, name=name, file=fname, nbytes=off, blobs=tuple(refs))
