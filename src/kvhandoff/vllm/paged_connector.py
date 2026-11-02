"""KVHandoffPagedConnector: vLLM v1 kv_producer for standard attention (Llama-style).

Design points:
  * built on KVConnectorBase_V1 directly (not ExampleConnector), with the 3-argument constructor vLLM's
    kv_connector_module_path loader requires;
  * SupportsHMA with a trivial request_finished_all_groups: the factory (v0.30.0 and main) refuses a non-HMA connector
    while the hybrid KV cache manager is on (factory.py create_connector), and Llama has one KV group anyway;
  * configuration only from kv_connector_extra_config (no environment variables, no default store path);
  * the export store (kvhandoff.store): bounded, back-pressure (StoreFull -> failure record), RemoveIPC check,
    versioned interchange header.

Request selection: kv_transfer_params {"kvhandoff_tag": "<tag>"}, or
export_all. Tensor parallel: every rank gathers and normalises its heads, all ranks all-gather the head axis for every
job in the same order (so the collectives pair up), and rank 0 writes.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from kvhandoff.config import ConnectorConfig
from kvhandoff.interchange import RopeInfo
from kvhandoff.logs import configure, get_logger
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
from kvhandoff.store import ExportStore
from kvhandoff.vllm import TESTED_VLLM, served_name, vllm_version

log = get_logger("vllm.paged")
_LAYER = re.compile(r"layers\.(\d+)\.")
_MAX_TOKENS_MARGIN = 64


class PagedHandoffMetadata(KVConnectorMetadata):
    def __init__(self, requests: list[StoreJob]):
        self.requests = requests


def _tp_group() -> Any:
    try:
        from vllm.distributed import get_tp_group

        return get_tp_group()
    except Exception:  # noqa: BLE001  (scheduler-side instance or no distributed state)
        return None


class KVHandoffPagedConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        configure()
        self._cfg = ConnectorConfig.from_extra(self._kv_transfer_config.kv_connector_extra_config)
        mc = vllm_config.model_config
        self._block_size = int(vllm_config.cache_config.block_size)
        self._max_tokens = int(mc.max_model_len) + _MAX_TOKENS_MARGIN
        try:
            self._head = int(mc.get_head_size())
        except Exception:  # noqa: BLE001
            self._head = 0
        self._slots = self._cfg.slots or int(vllm_config.scheduler_config.max_num_seqs) + 1
        self._tracker = RequestTracker(self._block_size, export_all=self._cfg.export_all)
        self._writer: PagedWriter | None = None  # worker side, TP rank 0 only, created at the first store
        self._slot_cache: dict[str, torch.Tensor] = {}  # tag -> slot mapping, for the current step
        hf = getattr(mc, "hf_config", None)
        self._info = PagedExportInfo(
            model_id=self._cfg.model_id or served_name(mc),
            num_model_layers=int(getattr(hf, "num_hidden_layers", 0) or 0),
            max_tokens=self._max_tokens,
            rope=RopeInfo(
                style="neox",
                applied_by="vllm",
                theta=getattr(hf, "rope_theta", None),
                scaling=getattr(hf, "rope_scaling", None),
            ),
            producer={"connector": type(self).__name__, "vllm": vllm_version(), "block_size": self._block_size},
        )
        ver = vllm_version()
        if ver not in TESTED_VLLM[type(self).__name__]:
            log.warning("untested_vllm", vllm=ver, tested=list(TESTED_VLLM[type(self).__name__]))
        log.info(
            "connector_ready",
            role=str(role),
            store=self._cfg.store_path,
            slots=self._slots,
            max_tokens=self._max_tokens,
            block_size=self._block_size,
            vllm=ver,
        )

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False  # a producer never loads

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return PagedHandoffMetadata(self._tracker.step(scheduler_output))

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._tracker.drop(request.request_id)
        return False, None

    def request_finished_all_groups(self, request: Any, block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        self._tracker.drop(request.request_id)
        return False, None

    # ------------------------------------------------------------------ worker side
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def _writer_for_rank0(self) -> PagedWriter:
        if self._writer is None:
            store = ExportStore(
                self._cfg.store_path,
                slots=self._slots,
                hold_seconds=self._cfg.hold_seconds,
                failed_keep=self._cfg.failed_keep,
                allow_login_uid=self._cfg.allow_login_uid_store,
            )
            self._writer = PagedWriter(
                store,
                self._info,
                timing_log=self._cfg.timing_log,
                stream_layers=self._cfg.stream_layers,
                direct_plane_copy=self._cfg.direct_plane_copy,
                prefault_slot_maps=self._cfg.prefault_slot_maps,
            )
        return self._writer

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        meta = self._connector_metadata
        if not isinstance(meta, PagedHandoffMetadata) or not meta.requests:
            return
        m = _LAYER.search(layer_name)
        if m is None:
            return
        idx = int(m.group(1))
        tp = _tp_group()
        multi = tp is not None and tp.world_size > 1
        rank0 = tp is None or not multi or tp.rank_in_group == 0
        out_dtype = getattr(torch, self._cfg.out_dtype) if self._cfg.out_dtype else None
        for job in meta.requests:  # the same order on every rank, so the all-gathers pair up
            slots = self._slot_cache.get(job.tag)
            if slots is None:
                slots = self._slot_cache[job.tag] = slot_mapping(job.block_ids, job.num_tokens, self._block_size)
            try:
                kv = to_kv_split(gather(kv_layer, slots, self._block_size), self._head)
            except UnsupportedLayout as exc:
                # The layout is a property of the attention backend, so every rank skips the same job here and
                # the remaining all-gathers still pair up.
                if rank0:
                    self._writer_for_rank0().mark_failed(job, f"UnsupportedLayout in {layer_name}: {exc}")
                continue
            if out_dtype is not None:
                kv = kv.to(out_dtype)
            if multi and tp is not None:
                kv = tp.all_gather(kv.contiguous(), dim=2)  # [2, T, H_total, D]
            if rank0:
                self._writer_for_rank0().write_layer(job, idx, layer_name, kv)

    def wait_for_save(self) -> None:
        self._slot_cache.clear()
        if self._writer is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._writer.finish_step()
