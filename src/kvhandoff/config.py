"""Configuration. Every address and path comes from here; nothing in the package has a deployment default.

Three consumers:
  ConnectorConfig  the vLLM connectors, from ``kv_connector_extra_config`` (JSON inside --kv-transfer-config)
  TransferConfig   the transfer service, from the ``[transfer]`` table of a TOML file
  BridgeConfig     the EXO hook, from the ``[bridge]`` table of a TOML file

Unknown keys are errors (a typo must not silently fall back to a default).
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

WIRE_DTYPES = ("float16", "bfloat16")


class ConfigError(ValueError):
    pass


def _check_keys(d: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _req_str(d: Mapping[str, Any], key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}: '{key}' is required (a non-empty string)")
    return v


def _opt_str(d: Mapping[str, Any], key: str, where: str) -> str | None:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{where}: '{key}' must be a non-empty string")
    return v


def _num(d: Mapping[str, Any], key: str, default: float, where: str, *, minimum: float = 0.0) -> float:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or v < minimum:
        raise ConfigError(f"{where}: '{key}' must be a number >= {minimum}")
    return float(v)


def _int(d: Mapping[str, Any], key: str, default: int | None, where: str, *, minimum: int = 0) -> int | None:
    v = d.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ConfigError(f"{where}: '{key}' must be an integer >= {minimum}")
    return v


def _bool(d: Mapping[str, Any], key: str, default: bool, where: str) -> bool:
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"{where}: '{key}' must be true or false")
    return v


def _url(d: Mapping[str, Any], key: str, where: str, *, required: bool) -> str | None:
    v = _req_str(d, key, where) if required else _opt_str(d, key, where)
    if v is not None and not (v.startswith("http://") or v.startswith("https://")):
        raise ConfigError(f"{where}: '{key}' must be an http:// or https:// URL")
    return None if v is None else v.rstrip("/")


# ----------------------------------------------------------------------------------------------- vLLM connectors
@dataclass(frozen=True)
class ConnectorConfig:
    store_path: str
    slots: int | None = None  # None: the connector picks (paged: max_num_seqs + 1, dsv4: 16)
    hold_seconds: float = 120.0
    failed_keep: int = 64
    model_id: str | None = None  # None: vLLM's served model name
    export_all: bool = False
    out_dtype: str | None = None  # paged only: cast exported K/V (e.g. float16 to match an fp16 MLX model)
    rerope: str = "ratio0"  # dsv4 only: "ratio0" re-rotates ratio-0 layers to MLX's plain RoPE, "none" leaves them
    rope_check_layers: int = 3  # dsv4 only: layers whose RoPE distance to MLX is measured and recorded
    allow_login_uid_store: bool = False  # accept a /dev/shm store owned by a login uid (RemoveIPC risk)
    timing_log: bool = False
    stream_layers: bool = False  # paged only: announce each layer as it is written, so it can be sent before publish
    # paged only: copy K and V into the slot map as two contiguous copies (the prototype's pattern) instead of one copy
    # into the strided [2, :T] view, which on a CUDA source goes through a fresh host buffer per layer. Same bytes.
    # Default on since a hardware A/B run: warm-slot prefill matched vLLM without a connector (8k 10.1 s against 9.0 s,
    # 32k 61.8 against 60.4) where the strided copy took 19.0 and 102.3 s. Without prefault_slot_maps, the first export
    # into rows this process has not written yet was slower than the strided path (8k 35.3 against 19.0 s, 32k 177.8
    # against 106.6 s): the GPU copy faulted each page of the new map itself.
    direct_plane_copy: bool = True
    # paged only: populate the page-table entries of the rows each layer is about to receive (madvise
    # MADV_POPULATE_WRITE, after reserve) before the device-to-host copy, once per map and row range. Removes the slow
    # first export in a fresh process (measured), where the GPU copy faulted each 4 KiB page itself.
    # On by default: with a noswap store (see README) it removes the cold copy cost. On a swappable store the cold
    # penalty is the store coming back from swap, which this cannot help.
    prefault_slot_maps: bool = True
    # dsv4 only: at start-up (register_kv_caches, TP rank 0) size, allocate and pre-fault every slot's layer files for
    # prompts of up to this many tokens (capped at max_model_len; 0 turns it off), so no request pays for a fresh slot
    # file (about 0.2 s at 2k, measured). V4 Flash at 32768: about 245 MB per slot, 3.9 GB for the 16 default slots,
    # resident from start-up on a noswap store. Skipped (logged) if that is over half the store's free bytes. The
    # paged connector ignores it: its slot files hold max_model_len rows of every layer (about 10.7 GB per slot for a
    # 70B at 32k), and it already reserves and pre-faults only the rows each export uses.
    presize_tokens: int = 32768

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any] | None) -> ConnectorConfig:
        where = "kv_connector_extra_config"
        d = dict(extra or {})
        if "shared_storage_path" in d and "store_path" not in d:  # the prototype's key
            d["store_path"] = d.pop("shared_storage_path")
        if "retain" in d and "slots" not in d:
            d["slots"] = d.pop("retain")
        _check_keys(d, {f.name for f in fields(cls)}, where)
        out_dtype = _opt_str(d, "out_dtype", where)
        if out_dtype is not None and out_dtype not in WIRE_DTYPES:
            raise ConfigError(f"{where}: out_dtype must be one of {WIRE_DTYPES}")
        rerope = d.get("rerope", "ratio0")
        if rerope not in ("ratio0", "none"):
            raise ConfigError(f"{where}: rerope must be 'ratio0' or 'none'")
        return cls(
            store_path=_req_str(d, "store_path", where),
            slots=_int(d, "slots", None, where, minimum=1),
            hold_seconds=_num(d, "hold_seconds", 120.0, where),
            failed_keep=_int(d, "failed_keep", 64, where, minimum=1) or 64,
            model_id=_opt_str(d, "model_id", where),
            export_all=_bool(d, "export_all", False, where),
            out_dtype=out_dtype,
            rerope=str(rerope),
            rope_check_layers=_int(d, "rope_check_layers", 3, where) or 0,
            allow_login_uid_store=_bool(d, "allow_login_uid_store", False, where),
            timing_log=_bool(d, "timing_log", False, where),
            stream_layers=_bool(d, "stream_layers", False, where),
            direct_plane_copy=_bool(d, "direct_plane_copy", True, where),
            prefault_slot_maps=_bool(d, "prefault_slot_maps", True, where),
            presize_tokens=_int(d, "presize_tokens", 32768, where) or 0,
        )


# ----------------------------------------------------------------------------------------------- transfer service
@dataclass(frozen=True)
class TransferConfig:
    bind: str
    port: int
    store_path: str
    token_file: str | None  # None only with insecure_no_auth = true (tests, never production)
    insecure_no_auth: bool = False
    hold_seconds: float = 120.0  # only for reporting in /healthz; retention itself is the writer's setting
    max_concurrent_transfers: int = 16  # layer/header requests served at once; more get 503 + Retry-After
    max_connections: int = 64  # open connections; more are closed at once
    io_timeout_s: float = 30.0  # per-socket read/write timeout
    header_timeout_s: float = 10.0  # whole request line + headers must arrive within this (slow clients cannot hold slots)
    allow_group_readable_token: bool = False
    log_level: str = "info"

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> TransferConfig:
        where = "[transfer]"
        _check_keys(d, {f.name for f in fields(cls)}, where)
        insecure = _bool(d, "insecure_no_auth", False, where)
        token_file = _opt_str(d, "token_file", where)
        if token_file is None and not insecure:
            raise ConfigError(f"{where}: 'token_file' is required (set insecure_no_auth = true only for tests)")
        port = _int(d, "port", None, where, minimum=1)
        if port is None or port > 65535:
            raise ConfigError(f"{where}: 'port' is required (1-65535)")
        return cls(
            bind=_req_str(d, "bind", where),
            port=port,
            store_path=_req_str(d, "store_path", where),
            token_file=token_file,
            insecure_no_auth=insecure,
            hold_seconds=_num(d, "hold_seconds", 120.0, where),
            max_concurrent_transfers=_int(d, "max_concurrent_transfers", 16, where, minimum=1) or 16,
            max_connections=_int(d, "max_connections", 64, where, minimum=1) or 64,
            io_timeout_s=_num(d, "io_timeout_s", 30.0, where, minimum=0.1),
            header_timeout_s=_num(d, "header_timeout_s", 10.0, where, minimum=0.1),
            allow_group_readable_token=_bool(d, "allow_group_readable_token", False, where),
            log_level=str(d.get("log_level", "info")),
        )


# ----------------------------------------------------------------------------------------------- EXO bridge
@dataclass(frozen=True)
class BridgeConfig:
    mode: str  # "paged" (Llama-style KVChunk stream) or "dsv4" (DeepSeek V4 ArraysState stream)
    vllm_url: str
    vllm_model: str
    vllm_api_key_file: str | None = None
    transfer_url: str | None = None  # fetch exports over HTTP from the transfer service ...
    transfer_token_file: str | None = None
    local_store_path: str | None = None  # ... or read them from a store on this host (paged mode, same node as vLLM)
    timeout_s: float = 50.0  # whole hand-off budget; keep under EXO's 60 s client socket timeout
    max_inflight: int = 4  # concurrent productions (vLLM prefills) from this bridge
    queue_wait_s: float = 5.0  # how long a hand-off may wait for an in-flight place before failing
    fetch_threads: int = 8  # dsv4: layer fetches in parallel, across all hand-offs of this process
    result_cache_entries: int = 8  # dsv4: converted results kept for the other pipeline ranks of a prompt
    result_cache_bytes: int = 4 << 30
    pad_tokens: int | None = None  # paged: extra tokens appended to the prompt; None = block_size + 1 (as tested)
    block_size: int = 16
    wire_dtype: str | None = None  # paged: dtype sent to EXO; None = the export's dtype (bf16 can be cast to fp16)
    expect_model_id: str | None = None  # refuse exports whose header names another model
    stream_layers: bool = False  # paged: send each layer as soon as vLLM has written it, while the prefill runs
    allow_group_readable_token: bool = False
    log_level: str = "info"

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> BridgeConfig:
        where = "[bridge]"
        _check_keys(d, {f.name for f in fields(cls)}, where)
        mode = _req_str(d, "mode", where)
        if mode not in ("paged", "dsv4"):
            raise ConfigError(f"{where}: mode must be 'paged' or 'dsv4'")
        transfer_url = _url(d, "transfer_url", where, required=False)
        local = _opt_str(d, "local_store_path", where)
        if (transfer_url is None) == (local is None):
            raise ConfigError(f"{where}: set exactly one of 'transfer_url' or 'local_store_path'")
        if mode == "dsv4" and local is not None:
            raise ConfigError(f"{where}: dsv4 mode fetches from the transfer service (the export lives on vLLM rank 0)")
        token = _opt_str(d, "transfer_token_file", where)
        if transfer_url is not None and token is None:
            raise ConfigError(f"{where}: 'transfer_token_file' is required with 'transfer_url'")
        wire = _opt_str(d, "wire_dtype", where)
        if wire is not None and wire not in WIRE_DTYPES:
            raise ConfigError(f"{where}: wire_dtype must be one of {WIRE_DTYPES}")
        timeout = _num(d, "timeout_s", 50.0, where, minimum=1.0)
        stream_layers = _bool(d, "stream_layers", False, where)
        if stream_layers and mode != "paged":
            raise ConfigError(f"{where}: stream_layers applies to paged mode only (dsv4 exports when the request ends)")
        return cls(
            mode=mode,
            vllm_url=_url(d, "vllm_url", where, required=True) or "",
            vllm_model=_req_str(d, "vllm_model", where),
            vllm_api_key_file=_opt_str(d, "vllm_api_key_file", where),
            transfer_url=transfer_url,
            transfer_token_file=token,
            local_store_path=local,
            timeout_s=timeout,
            max_inflight=_int(d, "max_inflight", 4, where, minimum=1) or 4,
            queue_wait_s=_num(d, "queue_wait_s", 5.0, where),
            fetch_threads=_int(d, "fetch_threads", 8, where, minimum=1) or 8,
            result_cache_entries=_int(d, "result_cache_entries", 8, where, minimum=1) or 8,
            result_cache_bytes=_int(d, "result_cache_bytes", 4 << 30, where, minimum=1) or (4 << 30),
            pad_tokens=_int(d, "pad_tokens", None, where),
            block_size=_int(d, "block_size", 16, where, minimum=1) or 16,
            wire_dtype=wire,
            expect_model_id=_opt_str(d, "expect_model_id", where),
            stream_layers=stream_layers,
            allow_group_readable_token=_bool(d, "allow_group_readable_token", False, where),
            log_level=str(d.get("log_level", "info")),
        )

    @property
    def effective_pad_tokens(self) -> int:
        return self.block_size + 1 if self.pad_tokens is None else self.pad_tokens


def load_toml(path: str, table: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    t = doc.get(table)
    if not isinstance(t, dict):
        raise ConfigError(f"{path}: no [{table}] table")
    return t  # pyright: ignore[reportUnknownVariableType]
