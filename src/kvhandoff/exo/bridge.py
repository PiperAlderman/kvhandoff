"""The EXO side of a hand-off: one EXO PrefillRequest in, one vLLM prefill, one export streamed out.

paged mode (Llama-style):
  1. append ``pad_tokens`` copies of the last token (default block_size + 1, as hardware-tested; attention is causal,
     so they cannot change the KV of the tokens before them), POST to vLLM with {"kvhandoff_tag": <fresh tag>};
  2. wait for the export's header (local store or transfer service), check kind, token count, model;
  3. stream Header, one KVChunk per layer (rows [start_pos, n), cast bf16->fp16 if asked), Done; the next layer is
     fetched while the current one is written; release the export.
  With stream_layers (docs/layerwise.md) the POST runs in the background, and each layer is sent as soon as the connector
  has written it (the connector needs stream_layers too; without it the export is sent once published, as above).
  The wire is unchanged. Done is sent only once vLLM's request has succeeded and the published header lists exactly
  the layers that were sent.
dsv4 mode (DeepSeek V4):
  results are keyed by a hash of the token ids. The pipeline ranks of one prompt share one vLLM call (per-key lock);
  different prompts run in parallel. Nothing is written to EXO's stream until the whole result is in memory, so any
  failure raises first and EXO answers with an error (the Mac prefills locally). Results are kept, LRU, bounded by
  entries and bytes, for ranks that arrive later. The whole V4 state is sent whatever start_pos is (the V4 cache is
  not trimmable); Done carries T - start_pos, the convention of exo-explore/exo#2545's adapter. The prompt is not padded.

Failure: every error raises HandoffError before EXO's stream is written to, except in paged mode after the first
layer (the export is held for the bridge, so this needs a genuine fault, or with stream_layers a failure of vLLM's
request or of the export while it is written). After the first byte the bridge ends the stream with EXO's error frame
before raising, so the Mac's client stops at once and prefills locally; its cache is untouched because EXO's client
buffers the whole stream (or, with EXO's per-layer ingest, stages it) and ingests only at Done.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from kvhandoff.auth import load_token
from kvhandoff.config import BridgeConfig, load_toml
from kvhandoff.interchange import KIND_DSV4, KIND_PAGED, ExportHeader, LayerRef
from kvhandoff.logs import configure, get_logger
from kvhandoff.store import StreamState
from kvhandoff.tags import PARAM_TAG, new_tag
from kvhandoff.transfer.client import Buffer, Deadline, ExportSource, HandoffError, LocalSource, TransferClient, VllmClient

log = get_logger("exo.bridge")

# stream_layers: how often the export's progress is polled (backing off from the first value to the second, and starting
# again for every layer). A layer of a 70B prompt at 32k takes about 0.5 s to compute and 0.11 s to send at 10 GbE.
_STREAM_POLL_MIN = 0.002
_STREAM_POLL_MAX = 0.02


def stream_poll_sleep(attempt: int) -> float:
    """Seconds before the next stream poll: _STREAM_POLL_MIN growing by 1.5x per attempt up to _STREAM_POLL_MAX.

    The exponent is capped: ``1.5 ** attempt`` overflows a float (OverflowError, errno 34) after about 1,750 attempts,
    which at the 20 ms cap is about 37 s of waiting. A streaming bridge in front of a buffered connector waits for the
    whole prefill before the first layer (48 to 55 s for a 70B prompt at 27k on a GB10), so every such hand-off failed
    at about 37 s and the decode node prefilled locally.
    """
    return min(_STREAM_POLL_MAX, _STREAM_POLL_MIN * (1.5 ** min(attempt, 16)))


CONFIG_ENV = "KVHANDOFF_BRIDGE_CONFIG"  # path of the bridge TOML; the hook is active only when it is set


def exo_protocol() -> Any:
    """EXO's wire protocol module (exo.worker.disaggregated.protocol), imported on first use."""
    return importlib.import_module("exo.worker.disaggregated.protocol")


# ----------------------------------------------------------------------------------------------- dtype on the wire
def bf16_to_fp16(data: Buffer) -> bytes:
    """bf16 words -> fp16 bytes through float32 (exact widening, then IEEE rounding), as the prototype bridge did."""
    import numpy as np

    words = np.frombuffer(data, dtype=np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float16).tobytes()


@dataclass
class _Result:
    tokens: int
    layers: list[tuple[int, list[tuple[str, tuple[int, ...], Buffer]]]]
    nbytes: int
    tag: str


@dataclass
class _Entry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    result: _Result | None = None
    error: str | None = None
    failed_at: float = 0.0


class ResultCache:
    """dsv4 results shared by the pipeline ranks of one prompt; LRU, bounded by entries and bytes, never evicting an
    entry that is being produced."""

    def __init__(self, max_entries: int, max_bytes: int):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def entry(self, key: str) -> _Entry:
        with self._lock:
            e = self._entries.get(key)
            if e is None:
                e = self._entries[key] = _Entry()
            self._entries.move_to_end(key)
            self._evict(keep=key)
            return e

    def _evict(self, keep: str) -> None:
        def total() -> int:
            return sum(x.result.nbytes for x in self._entries.values() if x.result is not None)

        for k in list(self._entries):
            if len(self._entries) <= self.max_entries and total() <= self.max_bytes:
                break
            if k != keep and not self._entries[k].lock.locked():
                del self._entries[k]

    def trim(self) -> None:
        with self._lock:
            self._evict(keep="")

    def __len__(self) -> int:
        return len(self._entries)

    def bytes(self) -> int:
        with self._lock:
            return sum(x.result.nbytes for x in self._entries.values() if x.result is not None)


def token_key(tokens: list[int]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.hexdigest()[:24]


class Bridge:
    def __init__(self, cfg: BridgeConfig, *, source: ExportSource | None = None, vllm: VllmClient | None = None):
        self.cfg = cfg
        if source is None:
            if cfg.local_store_path is not None:
                source = LocalSource(cfg.local_store_path)
            else:
                assert cfg.transfer_url is not None and cfg.transfer_token_file is not None
                token = load_token(cfg.transfer_token_file, allow_group_read=cfg.allow_group_readable_token)
                source = TransferClient(cfg.transfer_url, token)
        self.source = source
        if vllm is None:
            key = None
            if cfg.vllm_api_key_file:
                key = load_token(cfg.vllm_api_key_file, allow_group_read=cfg.allow_group_readable_token)
            vllm = VllmClient(cfg.vllm_url, cfg.vllm_model, key)
        self.vllm = vllm
        self._inflight = threading.BoundedSemaphore(cfg.max_inflight)
        self._pool = ThreadPoolExecutor(max_workers=cfg.fetch_threads, thread_name_prefix="kvhandoff-fetch")
        # stream_layers: vLLM's request runs here while its layers are sent; _inflight bounds it to max_inflight
        self._posts = ThreadPoolExecutor(max_workers=cfg.max_inflight, thread_name_prefix="kvhandoff-prefill")
        self.results = ResultCache(cfg.result_cache_entries, cfg.result_cache_bytes)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._posts.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ shared
    def _slot(self, deadline: Deadline) -> None:
        if not self._inflight.acquire(timeout=deadline.left(self.cfg.queue_wait_s)):
            raise HandoffError(f"{self.cfg.max_inflight} hand-offs already in flight")

    def _check(self, h: ExportHeader, kind: str) -> None:
        if h.kind != kind:
            raise HandoffError(f"export {h.tag} is {h.kind}, this bridge expects {kind}")
        if self.cfg.expect_model_id and h.model_id != self.cfg.expect_model_id:
            raise HandoffError(f"export {h.tag} is from model {h.model_id!r}, expected {self.cfg.expect_model_id!r}")
        if not h.layers:
            raise HandoffError(f"export {h.tag} has no layers")

    def resolve(self, request: Any, wfile: BinaryIO) -> bool:
        """EXO's ResolveHandler: returns True once the stream is written; raises on failure (EXO sends the error)."""
        t0 = time.perf_counter()
        try:
            if self.cfg.mode == "paged":
                self._serve_paged(request, wfile)
            else:
                self._serve_dsv4(request, wfile)
        except Exception as exc:
            log.warning(
                "handoff_failed",
                request_id=str(getattr(request, "request_id", "")),
                mode=self.cfg.mode,
                error=f"{type(exc).__name__}: {exc}",
                ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            raise
        return True

    # ------------------------------------------------------------------ paged
    def _wire(self, src_dtype: str) -> tuple[str, Callable[[Buffer], Buffer]]:
        wire = self.cfg.wire_dtype or src_dtype
        if wire != src_dtype and not (src_dtype == "bfloat16" and wire == "float16"):
            raise HandoffError(f"cannot send {src_dtype} KV as {wire}")
        return wire, (bf16_to_fp16 if wire != src_dtype else (lambda b: b))

    @staticmethod
    def _write_chunk(
        proto: Any,
        wfile: BinaryIO,
        layer: LayerRef,
        blobs: list[Buffer],
        span: tuple[int, int],
        wire: str,
        convert: Callable[[Buffer], Buffer],
    ) -> int:
        """One KVChunk with rows [start, n) of ``layer``; returns the payload bytes."""
        start, n = span
        kb = layer.blobs[0]
        _, heads, dim = kb.shape
        row = kb.nbytes // kb.shape[0]  # bytes per token row (heads x head_dim x item size)
        keys = convert(blobs[0][start * row : n * row])
        values = convert(blobs[1][start * row : n * row])
        proto.write_kv_chunk(
            wfile,
            layer_idx=layer.index,
            num_tokens=n - start,
            n_heads=heads,
            head_dim=dim,
            dtype=wire,
            keys=keys,
            values=values,
        )
        return len(keys) + len(values)

    @staticmethod
    def _error_frame(proto: Any, wfile: BinaryIO, exc: BaseException) -> None:
        """A failure after the first byte: end the stream with EXO's error frame before raising, so the Mac's client
        stops at once and falls back, and a stream cut at a frame boundary can never pass for a complete one."""
        with contextlib.suppress(Exception):
            proto.write_error(wfile, code=500, message=f"kvhandoff: {type(exc).__name__}: {exc}"[:1000])
            wfile.flush()

    def _serve_paged(self, request: Any, wfile: BinaryIO) -> None:
        if self.cfg.stream_layers:
            self._serve_paged_stream(request, wfile)
            return
        proto = exo_protocol()
        tokens = [int(t) for t in request.token_ids]
        n, start = len(tokens), int(request.start_pos)
        if n == 0 or not 0 <= start < n:
            raise HandoffError(f"bad request: {n} tokens, start_pos {start}")
        deadline = Deadline(self.cfg.timeout_s)
        padded = tokens + [tokens[-1]] * self.cfg.effective_pad_tokens
        self._slot(deadline)
        try:
            tag = new_tag("p")
            t_vllm = self.vllm.prefill(padded, {PARAM_TAG: tag}, deadline)
            t1 = time.perf_counter()
            h = self.source.wait_header(tag, deadline)
            t_wait = time.perf_counter() - t1
        finally:
            self._inflight.release()
        self._check(h, KIND_PAGED)
        if h.num_tokens < n:
            raise HandoffError(f"export {tag} holds {h.num_tokens} tokens, the prompt has {n}")
        layers = sorted(h.layers, key=lambda x: x.index)
        wire, convert = self._wire(layers[0].blobs[0].dtype)
        t2 = time.perf_counter()

        def fetch(i: int, budget: Deadline) -> list[Buffer]:
            return self.source.layer_blobs(h, layers[i], budget)

        ahead: Future[list[Buffer]] = self._pool.submit(fetch, 0, deadline)
        first = ahead.result(timeout=deadline.left())  # fail before writing anything if layer 0 is unavailable
        # From here EXO's client is reading (its socket timeout is per receive), so each further layer gets its own
        # budget instead of the remainder of the pre-header one.
        proto.write_header(
            wfile,
            proto.Header(
                request_id=request.request_id, model_id=request.model_id, num_layers=len(layers), dtype=wire, start_pos=start
            ),
        )
        sent = 0
        blobs = first
        try:
            for i, layer in enumerate(layers):
                if i > 0:
                    blobs = ahead.result(timeout=self.cfg.timeout_s)
                if i + 1 < len(layers):
                    ahead = self._pool.submit(fetch, i + 1, Deadline(self.cfg.timeout_s))
                sent += self._write_chunk(proto, wfile, layer, blobs, (start, n), wire, convert)
            proto.write_done(wfile, total_tokens=n - start)
            wfile.flush()
        except Exception as exc:
            self._error_frame(proto, wfile, exc)
            raise
        self.source.release(h)
        log.info(
            "handoff",
            mode="paged",
            tag=tag,
            tokens=n,
            start_pos=start,
            layers=len(layers),
            wire=wire,
            vllm_s=round(t_vllm, 3),
            wait_s=round(t_wait, 3),
            stream_s=round(time.perf_counter() - t2, 3),
            bytes=sent,
            export_s=h.extra.get("export_seconds"),
        )

    # ------------------------------------------------------------------ paged, streamed layer by layer
    @staticmethod
    def _post_failed(post: Future[float]) -> None:
        if post.done() and (exc := post.exception()) is not None:
            raise HandoffError(f"vLLM prefill failed while streaming: {exc}") from exc

    def _await(
        self, tag: str, budget: Deadline, post: Future[float], stop: threading.Event, ready: Callable[[StreamState], bool]
    ) -> StreamState:
        """Poll the export until ``ready``; fail as soon as vLLM's request, the export, or this hand-off fails."""
        attempt = 0
        while True:
            if stop.is_set():
                raise HandoffError(f"hand-off of {tag} stopped")
            self._post_failed(post)
            st = self.source.stream_state(tag, budget)
            if st is not None and ready(st):
                return st
            time.sleep(min(stream_poll_sleep(attempt), budget.left()))
            attempt += 1

    def _await_layer(self, tag: str, index: int, budget: Deadline, post: Future[float], stop: threading.Event) -> StreamState:
        def ready(st: StreamState) -> bool:
            if any(x.index == index for x in st.header.layers):
                return True
            if st.published:
                raise HandoffError(f"export {tag} was published without layer {index}")
            return False

        return self._await(tag, budget, post, stop, ready)

    def _fetch_streamed(
        self, tag: str, index: int, budget: Deadline, post: Future[float], stop: threading.Event
    ) -> tuple[LayerRef, list[Buffer]]:
        st = self._await_layer(tag, index, budget, post, stop)
        layer = st.header.layer(index)
        return layer, self.source.layer_blobs(st.header, layer, budget, live=True)

    def _serve_paged_stream(self, request: Any, wfile: BinaryIO) -> None:
        """paged mode with stream_layers: vLLM's request runs in the background while the export's layers are sent
        to EXO as the connector writes them (stream_layers in kv_connector_extra_config). Against a connector that
        does not stream, the first state seen is the published export, and this is the buffered path. Done is sent
        only once vLLM's request has succeeded and the published header lists exactly the layers that were sent."""
        proto = exo_protocol()
        tokens = [int(t) for t in request.token_ids]
        n, start = len(tokens), int(request.start_pos)
        if n == 0 or not 0 <= start < n:
            raise HandoffError(f"bad request: {n} tokens, start_pos {start}")
        deadline = Deadline(self.cfg.timeout_s)
        padded = tokens + [tokens[-1]] * self.cfg.effective_pad_tokens
        self._slot(deadline)
        tag = new_tag("p")
        t0 = time.perf_counter()
        try:
            post = self._posts.submit(self.vllm.prefill, padded, {PARAM_TAG: tag}, deadline)
        except BaseException:
            self._inflight.release()
            raise
        post.add_done_callback(lambda _f: self._inflight.release())  # the in-flight limit bounds vLLM prefills
        stop = threading.Event()
        try:
            first = self._await(tag, deadline, post, stop, lambda st: bool(st.header.layers))
            h = first.header
            self._check(h, KIND_PAGED)
            if h.num_tokens < n:
                raise HandoffError(f"export {tag} holds {h.num_tokens} tokens, the prompt has {n}")
            if first.published:
                expect = sorted(x.index for x in h.layers)
            elif h.num_model_layers > 0:
                expect = list(range(h.num_model_layers))
            else:  # the producer does not say how many layers there are: wait for the whole export
                h = self._await(tag, deadline, post, stop, lambda st: st.published).header
                expect = sorted(x.index for x in h.layers)
            layer0, blobs = self._fetch_streamed(tag, expect[0], deadline, post, stop)
            t_first = time.perf_counter() - t0
            wire, convert = self._wire(layer0.blobs[0].dtype)
        except BaseException:
            stop.set()
            raise
        # From here EXO's client is reading; each further layer gets its own budget, as in the buffered path.
        proto.write_header(
            wfile,
            proto.Header(
                request_id=request.request_id, model_id=request.model_id, num_layers=len(expect), dtype=wire, start_pos=start
            ),
        )
        sent = 0
        streamed: dict[int, LayerRef] = {}
        try:
            layer = layer0
            ahead: Future[tuple[LayerRef, list[Buffer]]] | None = None
            for pos, idx in enumerate(expect):
                if ahead is not None:
                    layer, blobs = ahead.result(timeout=self.cfg.timeout_s + 1.0)
                if pos + 1 < len(expect):
                    ahead = self._pool.submit(
                        self._fetch_streamed, tag, expect[pos + 1], Deadline(self.cfg.timeout_s), post, stop
                    )
                if layer.index != idx:
                    raise HandoffError(f"export {tag}: expected layer {idx}, got {layer.index}")
                streamed[idx] = layer
                sent += self._write_chunk(proto, wfile, layer, blobs, (start, n), wire, convert)
            t_layers = time.perf_counter() - t0
            final = self._await(tag, Deadline(self.cfg.timeout_s), post, stop, lambda st: st.published).header
            t_vllm = post.result(timeout=self.cfg.timeout_s)
            if final.seq != h.seq or sorted(x.index for x in final.layers) != expect:
                raise HandoffError(f"export {tag} was published with other layers than were streamed")
            if any(final.layer(i) != ref for i, ref in streamed.items()):
                raise HandoffError(f"export {tag}: a streamed layer differs from the published one")
            proto.write_done(wfile, total_tokens=n - start)
            wfile.flush()
        except Exception as exc:
            stop.set()
            self._error_frame(proto, wfile, exc)
            raise
        self.source.release(final)
        log.info(
            "handoff",
            mode="paged",
            streamed=True,
            tag=tag,
            tokens=n,
            start_pos=start,
            layers=len(expect),
            wire=wire,
            vllm_s=round(t_vllm, 3),
            first_layer_s=round(t_first, 3),
            last_layer_s=round(t_layers, 3),
            total_s=round(time.perf_counter() - t0, 3),
            bytes=sent,
            export_s=final.extra.get("export_seconds"),
        )

    # ------------------------------------------------------------------ dsv4
    def _produce_dsv4(self, tokens: list[int], key: str, deadline: Deadline) -> _Result:
        self._slot(deadline)
        try:
            tag = f"{key}-{new_tag('d')[2:10]}"
            t_vllm = self.vllm.prefill(tokens, {PARAM_TAG: tag}, deadline)
            t1 = time.perf_counter()
            h = self.source.wait_header(tag, deadline)
            t_wait = time.perf_counter() - t1
        finally:
            self._inflight.release()
        self._check(h, KIND_DSV4)
        if h.num_tokens != len(tokens):
            raise HandoffError(f"export {tag} has {h.num_tokens} tokens, the prompt has {len(tokens)}")
        t2 = time.perf_counter()
        futures = [(layer, self._pool.submit(self.source.layer_blobs, h, layer, deadline)) for layer in h.layers]
        layers: list[tuple[int, list[tuple[str, tuple[int, ...], Buffer]]]] = []
        nbytes = 0
        for layer, fut in futures:
            data = fut.result(timeout=deadline.left())
            layers.append((layer.index, [(b.dtype, b.shape, d) for b, d in zip(layer.blobs, data, strict=True)]))
            nbytes += sum(len(d) for d in data)
        layers.sort(key=lambda x: x[0])
        self.source.release(h)
        log.info(
            "handoff",
            mode="dsv4",
            tag=tag,
            tokens=len(tokens),
            layers=len(layers),
            vllm_s=round(t_vllm, 3),
            wait_s=round(t_wait, 3),
            fetch_s=round(time.perf_counter() - t2, 3),
            bytes=nbytes,
            export_s=h.extra.get("export_seconds"),
            rope_check=h.extra.get("rope_check_max_abs"),
        )
        return _Result(tokens=len(tokens), layers=layers, nbytes=nbytes, tag=tag)

    def dsv4_result(self, tokens: list[int]) -> _Result:
        """The converted layers for this prompt: produced once, shared by concurrent pipeline ranks."""
        deadline = Deadline(self.cfg.timeout_s)
        key = token_key(tokens)
        ent = self.results.entry(key)
        arrived = time.monotonic()
        if not ent.lock.acquire(timeout=deadline.left()):
            raise HandoffError("timed out waiting for another rank's hand-off of this prompt")
        try:
            if ent.result is not None:
                return ent.result
            if ent.error is not None and ent.failed_at >= arrived:  # the attempt this rank waited on failed
                raise HandoffError(f"hand-off failed for this prompt: {ent.error}")
            ent.error = None
            try:
                ent.result = self._produce_dsv4(tokens, key, deadline)
            except Exception as exc:
                ent.error = f"{type(exc).__name__}: {exc}"
                ent.failed_at = time.monotonic()
                raise
            return ent.result
        finally:
            ent.lock.release()
            self.results.trim()

    def _serve_dsv4(self, request: Any, wfile: BinaryIO) -> None:
        proto = exo_protocol()
        tokens = [int(t) for t in request.token_ids]
        start = int(request.start_pos)
        res = self.dsv4_result(tokens)  # raises before anything is written
        proto.write_header(
            wfile,
            proto.Header(
                request_id=request.request_id,
                model_id=request.model_id,
                num_layers=len(res.layers),
                dtype="bfloat16",
                start_pos=start,
            ),
        )
        for idx, blobs in res.layers:
            proto.write_arrays_state(wfile, idx, [proto.TensorBlob(dtype=d, shape=s, data=b) for d, s, b in blobs])
        proto.write_done(wfile, total_tokens=res.tokens - start)
        wfile.flush()


# ----------------------------------------------------------------------------------------------- hook entry point
_bridge: Bridge | None = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> Bridge:
    global _bridge
    with _BRIDGE_LOCK:
        if _bridge is None:
            path = os.environ.get(CONFIG_ENV)
            if not path:
                raise HandoffError(f"{CONFIG_ENV} is not set")
            cfg = BridgeConfig.from_mapping(load_toml(path, "bridge"))
            configure(cfg.log_level)
            _bridge = Bridge(cfg)
            log.info(
                "bridge_ready",
                mode=cfg.mode,
                vllm=cfg.vllm_url,
                source=cfg.transfer_url or cfg.local_store_path,
                timeout_s=cfg.timeout_s,
                max_inflight=cfg.max_inflight,
            )
        return _bridge


def resolve(request: Any, wfile: BinaryIO) -> bool:
    """What the EXO runner hook calls (see kvhandoff.exo.hook)."""
    return get_bridge().resolve(request, wfile)
