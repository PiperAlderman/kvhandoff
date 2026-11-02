"""KVHandoffDSV4Connector: vLLM v1 kv_producer exporting DeepSeek V4's whole per-layer cache state for each finished,
export-tagged request, converted to the DeepseekV4Cache wire blobs of EXO's DeepSeek V4 cache transport (exo-explore/exo#2545).

Why not save_kv_layer: DeepseekV4Attention.attention_impl carries only @eager_break_during_capture
(models/deepseek_v4/attention.py:432 at 0.23.1rc1.dev1053; 542, 737 on main e0d00e4b); @maybe_transfer_kv_layer wraps
only unified_attention_with_output and the MLA op, so save_kv_layer never fires for V4 layers. Instead:
  scheduler  request_finished_all_groups(request, block_ids) (SupportsHMA) is called once per finished request with its
             per-group block ids; each export-tagged request queues a job and returns delay_free=True. The next
             build_connector_meta hands all queued jobs to every worker.
  worker     get_finished(finished_req_ids), every step: TP rank 0 synchronises the device once and exports each job
             into its own slot; every rank then reports the job ids once it has seen them in finished_req_ids, and
             vLLM frees the blocks after all ranks reported. Ranks other than 0 touch no cache and write nothing.
HMA must stay on: without SupportsHMA older vLLM turns the hybrid KV cache manager off (which rewrites SWA and
compressor-state specs to block size 256 and changes the layouts read here); vLLM main refuses to start instead.

Request selection: kv_transfer_params {"kvhandoff_tag": "<tag>"}, or export_all. All TP ranks need this package
importable (jobs are pickled to every worker).
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)

from kvhandoff.config import ConfigError, ConnectorConfig
from kvhandoff.logs import configure, get_logger
from kvhandoff.producer.dsv4_export import (
    DSV4Exporter,
    DSV4Job,
    DSV4ModelInfo,
    FinishedTracker,
    SchedulerQueue,
    rope_config_from_checkpoint,
)
from kvhandoff.store import ExportStore
from kvhandoff.vllm import TESTED_VLLM, served_name, vllm_version

log = get_logger("vllm.dsv4")
DEFAULT_SLOTS = 16


class DSV4HandoffMetadata(KVConnectorMetadata):
    def __init__(self, jobs: list[DSV4Job]):
        self.jobs = jobs


class KVHandoffDSV4Connector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: Any, role: KVConnectorRole, kv_cache_config: Any = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        configure()
        self._cfg = ConnectorConfig.from_extra(self._kv_transfer_config.kv_connector_extra_config)
        if self._cfg.stream_layers:  # V4 exports in get_finished, after the whole request: there is nothing to stream
            raise ConfigError("kv_connector_extra_config: stream_layers applies to KVHandoffPagedConnector only")
        self._queue = SchedulerQueue(export_all=self._cfg.export_all)
        self._finished = FinishedTracker()
        self._kv: dict[str, torch.Tensor] = {}
        self._exporter: DSV4Exporter | None = None  # created lazily on TP rank 0
        ver = vllm_version()
        if ver not in TESTED_VLLM[type(self).__name__]:
            log.warning("untested_vllm", vllm=ver, tested=list(TESTED_VLLM[type(self).__name__]))
        log.info(
            "connector_ready", role=str(role), store=self._cfg.store_path, slots=self._cfg.slots or DEFAULT_SLOTS, vllm=ver
        )

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        return DSV4HandoffMetadata(self._queue.take())

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return False, None  # non-HMA path; never taken while HMA is on

    def request_finished_all_groups(self, request: Any, block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        queued = self._queue.on_finished(
            str(request.request_id), int(request.num_prompt_tokens), request.kv_transfer_params or {}, block_ids
        )
        return queued, None

    # ------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv = dict(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        return  # never called for DeepSeek V4 layers (module docstring)

    def wait_for_save(self) -> None:
        return

    @staticmethod
    def _tp_rank() -> int:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            return 0

    def _modules(self) -> dict[str, Any]:
        try:
            from vllm.forward_context import get_forward_context

            return get_forward_context().no_compile_layers
        except Exception:  # noqa: BLE001
            return self._vllm_config.compilation_config.static_forward_context

    def _group_of(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for g, group in enumerate(self._kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                out[name] = g
        return out

    def _exporter_for_rank0(self) -> DSV4Exporter:
        if self._exporter is None:
            mc = self._vllm_config.model_config
            hf = mc.hf_config
            n = int(hf.num_hidden_layers)
            info = DSV4ModelInfo(
                model_id=self._cfg.model_id or served_name(mc),
                num_layers=n,
                compress_ratios=tuple(int(r) for r in list(hf.compress_ratios)[:n]),
                rope_cfg=rope_config_from_checkpoint(str(mc.model)),
                producer={"connector": type(self).__name__, "vllm": vllm_version()},
            )
            store = ExportStore(
                self._cfg.store_path,
                slots=self._cfg.slots or DEFAULT_SLOTS,
                hold_seconds=self._cfg.hold_seconds,
                failed_keep=self._cfg.failed_keep,
                allow_login_uid=self._cfg.allow_login_uid_store,
            )
            self._exporter = DSV4Exporter(
                store,
                info,
                rerope=self._cfg.rerope,
                rope_check_layers=self._cfg.rope_check_layers,
                timing_log=self._cfg.timing_log,
            )
        return self._exporter

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        meta = self._connector_metadata
        jobs = meta.jobs if isinstance(meta, DSV4HandoffMetadata) else []
        if jobs and self._tp_rank() == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # aux-stream compressor / indexer writes of the last step are done
            exporter = self._exporter_for_rank0()
            groups, modules = self._group_of(), self._modules()
            for job in jobs:  # several prefills can finish in one step; each is exported on its own
                exporter.export(job, self._kv, groups, modules)  # failures become failure records
        return self._finished.step(finished_req_ids, (j.req_id for j in jobs)), None
