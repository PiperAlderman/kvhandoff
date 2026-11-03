"""CPU micro-benchmark for the DeepSeek V4 short-prompt hand-off fixes (3/11/2026).

Runs against any kvhandoff source tree, so the same script measures the tree before and after the fixes:

  python3 bench/short_prompt_overhead_bench.py <kvhandoff src dir> [--store-dir /dev/shm] [--json out.json]

(a) Slot files. A V4 Flash export of 1882 tokens (a measured 2k hand-off: 43 layers, sizes from the converter's own
    arithmetic) written through DSV4Exporter._write_layer into: a fresh store (first export into new slot files), the
    same slot again (used), a slot first used by a 5% shorter prompt (grown), and, where the tree has presize(), a
    slot presized for 32768 tokens at start-up (presize's own time is reported separately). Conversion is left out:
    each layer is one bf16 CPU tensor of the layer's size. On this x86 VM the copy is a CPU memcpy, so a fresh page
    costs a CPU fault; on a GB10 Spark the device-to-host copy pays 30 to 60 us per fresh 4 KiB page instead, so the
    GB10 gap is larger than shown here.
(b) Header parse per layer request. One published export with a header shaped like a measured one (43 layers, 43 kB), then 43
    ExportReader.open() calls per hand-off, in sequence and on 8 threads (the bridge's fetch_threads), as the
    transfer service does for each layer GET. Each hand-off republishes first, so every hand-off has a new header.
(c) Header wait. LocalSource.wait_header() while another thread publishes the export after a delay of 20 to 300 ms;
    the overshoot is the time from publish() returning to wait_header() returning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

FLASH_RATIOS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)]  # V4 Flash: 43 layers
T_2K = 1882
WINDOW, HEAD, IDX_DIM = 128, 512, 128


def layer_nbytes(t: int, ratio: int) -> int:
    """convert_layer's blob bytes (same arithmetic as dsv4_convert.layer_nbytes; repeated so older trees can run)."""
    total = 4 * 22 + 2 * (min(t, WINDOW) * HEAD * 2)
    for r, d in ([(ratio, HEAD)] if ratio else []) + ([(4, IDX_DIM)] if ratio == 4 else []):
        width = (2 if r == 4 else 1) * d
        n, rem = divmod(t, r)
        total += 2 * rem * width * 2 + (2 * r * width * 2 if r == 4 and n else 0) + n * d * 2
    return total


def median_ms(xs: list[float]) -> float:
    return round(statistics.median(xs) * 1000, 2)


# ----------------------------------------------------------------------------------------------- (a)
def bench_slots(store_dir: str, reps: int) -> dict[str, Any]:
    import torch

    from kvhandoff.producer.dsv4_export import DSV4Exporter, DSV4ModelInfo
    from kvhandoff.store import ExportStore

    info = DSV4ModelInfo(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        num_layers=len(FLASH_RATIOS),
        compress_ratios=tuple(FLASH_RATIOS),
        rope_cfg={"rope_theta": 10000.0, "compress_rope_theta": 160000.0, "rope_scaling": None},
    )
    torch.set_num_threads(1)  # one copy thread, as a single map write; steadier on a shared VM
    srcs: dict[int, torch.Tensor] = {}

    def blobs(t: int, i: int) -> list[Any]:
        n = layer_nbytes(t, FLASH_RATIOS[i])
        if n not in srcs:
            srcs[n] = torch.randn(n // 2).to(torch.bfloat16)
        return [("x", "bfloat16", (n // 2,), srcs[n])]

    def export(exp: Any, t: int) -> float:
        k = exp.store.begin(f"b-{time.monotonic_ns()}")
        t0 = time.perf_counter()
        for i in range(len(FLASH_RATIOS)):
            exp._write_layer(k, i, f"model.layers.{i}.attn", blobs(t, i))
        dt = time.perf_counter() - t0
        exp.store.abandon(k)
        return dt

    out: dict[str, list[float]] = {"fresh": [], "used": [], "grown": [], "presized_first": [], "presize_startup": []}
    for _ in range(reps):
        root = tempfile.mkdtemp(prefix="kvh-bench-", dir=store_dir)
        try:
            store = ExportStore(os.path.join(root, "s"), slots=1, hold_seconds=0, allow_login_uid=True)
            exp = DSV4Exporter(store, info)
            out["fresh"].append(export(exp, T_2K))
            out["used"].append(export(exp, T_2K))
            store.close()
            store = ExportStore(os.path.join(root, "g"), slots=1, hold_seconds=0, allow_login_uid=True)
            exp = DSV4Exporter(store, info)
            export(exp, int(T_2K / 1.05))
            out["grown"].append(export(exp, T_2K))
            store.close()
            if hasattr(DSV4Exporter, "presize"):
                store = ExportStore(os.path.join(root, "p"), slots=1, hold_seconds=0, allow_login_uid=True)
                exp = DSV4Exporter(store, info)
                t0 = time.perf_counter()
                exp.presize(32768)
                out["presize_startup"].append(time.perf_counter() - t0)
                out["presized_first"].append(export(exp, T_2K))
                store.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return {k: median_ms(v) for k, v in out.items() if v}


# ----------------------------------------------------------------------------------------------- (b)
def measured_header(tag: str) -> Any:
    from kvhandoff.interchange import KIND_DSV4, BlobRef, ExportHeader, LayerRef, RopeInfo

    def layer(i: int) -> LayerRef:
        r = FLASH_RATIOS[i]
        roles = ["state_header", "window_k", "window_v"]
        if r:
            roles += [f"compressor.{a}" for a in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool")]
        if r == 4:
            roles += [f"indexer.{a}" for a in ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool")]
        refs, off = [], 0
        for role in roles:
            n = 88 if role == "state_header" else 256
            shape = (22,) if role == "state_header" else (1, 1, 128)
            dtype = "float32" if role == "state_header" else "bfloat16"
            refs.append(BlobRef(role=role, dtype=dtype, shape=shape, offset=off, nbytes=n))
            off += n
        return LayerRef(
            index=i, name=f"model.layers.{i}.attn", file=f"L{i}.bin", nbytes=off, blobs=tuple(refs), attrs={"compress_ratio": r}
        )

    return ExportHeader(
        kind=KIND_DSV4,
        tag=tag,
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        num_tokens=T_2K,
        layers=tuple(layer(i) for i in range(len(FLASH_RATIOS))),
        num_model_layers=len(FLASH_RATIOS),
        rope=RopeInfo(
            style="gptj",
            applied_by="mixed",
            theta=10000.0,
            scaling={"factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
            details={"ratio0_layers": "mlx", "compressed_layers": "vllm", "compress_rope_theta": 160000.0},
        ),
        producer={"connector": "KVHandoffDSV4Connector", "vllm": "0.23.1rc1", "req_id": "cmpl-x", "package": "0.1.0"},
        extra={"rope_check_max_abs": {"0": 0.53, "1": 0.53, "2": 1e-8}, "export_seconds": 0.2469},
    )


def publish(store: Any, tag: str) -> Any:
    h = measured_header(tag)
    k = store.begin(tag)
    for layer in h.layers:
        store.write_bytes(k, layer.file, 0, b"\0" * layer.nbytes)
    return store.publish(k, h)


def bench_header(store_dir: str, reps: int) -> dict[str, Any]:
    from kvhandoff.store import ExportReader, ExportStore

    root = tempfile.mkdtemp(prefix="kvh-bench-", dir=store_dir)
    try:
        store = ExportStore(os.path.join(root, "s"), slots=1, hold_seconds=0, allow_login_uid=True)
        reader = ExportReader(store.root)
        size = len(publish(store, "h-0").to_json())
        seq_t: list[float] = []
        thr_t: list[float] = []
        pool = ThreadPoolExecutor(8)

        def one(tag: str, seq: int) -> None:
            with reader.open(tag, seq=seq):
                pass

        for rep in range(reps):
            h = publish(store, f"h-{rep}")
            t0 = time.perf_counter()
            for _ in range(43):
                one(h.tag, h.seq)
            seq_t.append(time.perf_counter() - t0)
            h = publish(store, f"t-{rep}")
            t0 = time.perf_counter()
            list(pool.map(one, [h.tag] * 43, [h.seq] * 43))
            thr_t.append(time.perf_counter() - t0)
        pool.shutdown()
        store.close()
        return {"header_bytes": size, "43_opens_sequential": median_ms(seq_t), "43_opens_8_threads": median_ms(thr_t)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ----------------------------------------------------------------------------------------------- (c)
def bench_wait(store_dir: str, reps: int) -> dict[str, Any]:
    from kvhandoff.store import ExportStore
    from kvhandoff.transfer.client import Deadline, LocalSource

    root = tempfile.mkdtemp(prefix="kvh-bench-", dir=store_dir)
    try:
        store = ExportStore(os.path.join(root, "s"), slots=2, hold_seconds=0, allow_login_uid=True)
        src = LocalSource(store.root)
        over: list[float] = []
        delays = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        for rep in range(reps):
            for d in delays:
                tag = f"w-{rep}-{int(d * 1000)}"
                published: list[float] = []

                def later(tag: str = tag, d: float = d, published: list[float] = published) -> None:
                    time.sleep(d)
                    publish(store, tag)
                    published.append(time.perf_counter())

                th = threading.Thread(target=later)
                th.start()
                src.wait_header(tag, Deadline(10.0))
                done = time.perf_counter()
                th.join()
                over.append(done - published[0])
        store.close()
        return {
            "overshoot_median": median_ms(over),
            "overshoot_mean": round(statistics.mean(over) * 1000, 2),
            "overshoot_max": round(max(over) * 1000, 2),
            "n": len(over),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="the kvhandoff tree's src directory")
    ap.add_argument("--store-dir", default="/dev/shm" if os.path.isdir("/dev/shm") else None)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.src))
    os.environ.setdefault("KVHANDOFF_LOG_LEVEL", "error")
    res = {
        "src": os.path.abspath(a.src),
        "store_dir": a.store_dir,
        "a_slot_write_ms": bench_slots(a.store_dir, a.reps),
        "b_header_ms": bench_header(a.store_dir, a.reps * 7),
        "c_header_wait_ms": bench_wait(a.store_dir, a.reps // 2 + 1),
    }
    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
