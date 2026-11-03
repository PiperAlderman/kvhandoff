"""DeepSeek V4 cache conversion: vLLM paged caches -> the DeepseekV4Cache wire blobs of exo-explore/exo#2545. Pure torch.

Moved unchanged in arithmetic from the prototype converter (570/570 CPU self-test checks, 574/574 on GPU
against vLLM's own cache-writer kernels; hardware hand-offs). tests/test_dsv4.py holds this module to
byte-identical output with that converter (tests/data/dsv4_golden.json). What changed: type hints, blob roles, and
the store moved to kvhandoff.store.

Source layouts (vLLM v0.23.1rc1.dev1053; paths under vllm/):
  * fp8_ds_mla page, one per block of `bs` rows, viewed as uint8 [nblk, bs, 584]
    (v1/attention/backends/mla/sparse_swa.py:141-144):
        [0, bs*576)        per row: 448 fp8 e4m3 (nope) + 64 bf16 (rope, already rotated)
        [bs*576, bs*584)   per row: 8 UE8M0 bytes, 7 used, value = fp8 * 2**(byte - 127), one per 64 nope dims
  * Indexer cache, uint8 [nblk, bs, 132]: [0, bs*128) fp8 e4m3 per row, then [bs*128, bs*132) one float32 scale (2**e)
    per row (common/ops/fused_compress_quant_cache.py:342-344, 457-472; v1/attention/backends/mla/indexer.py:138-146).
  * Compressor state, float32 [nblk, bs, 2*coff*d]: row = [kv (coff*d) | score + ape[pos % ratio] (coff*d)]
    (common/ops/save_partial_states.py:89-100; models/deepseek_v4/compressor.py:237-242, 302).
Target: mlx_lm deepseek_v4.py DeepseekV4Cache.state as serialised by deepseek_v4_cache_to_blobs in exo-explore/exo#2545
(wire version 1: float32 header [version, keep, max_size, offset, _idx, window_present, then per branch 5 presence flags,
buffer_count, -1, -1], then window keys, window values, then each present branch array).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

HEAD = 512
ROPE = 64
NOPE = HEAD - ROPE  # 448
QBLOCK = 64
ROW_BYTES = NOPE + 2 * ROPE  # 576
SCALE_BYTES = 8
PAGE_ROW = ROW_BYTES + SCALE_BYTES  # 584
IDX_DIM = 128
WINDOW = 128
BRANCH_ARRAYS = 5  # buffer_kv, buffer_gate, prev_kv, prev_gate, pool
BRANCH_FIELDS = ("buffer_kv", "buffer_gate", "prev_kv", "prev_gate", "pool")
BRANCH_KEYS = ("compressor", "indexer")  # _DEEPSEEK_V4_BRANCH_KEYS in exo-explore/exo#2545, wire order
WIRE_VERSION = 1  # _DEEPSEEK_V4_WIRE_VERSION in exo-explore/exo#2545

Blob = tuple[str, str, tuple[int, ...], torch.Tensor]  # (role, dtype, shape, tensor)


# ----------------------------------------------------------------------------------------------- page readers
def _block_span(block_ids: Sequence[int], bs: int, first: int, n: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    b0 = first // bs
    b1 = (first + n - 1) // bs + 1
    if b1 > len(block_ids):
        raise ValueError(f"rows [{first},{first + n}) need block {b1 - 1}, request has {len(block_ids)} blocks")
    idx = torch.as_tensor(list(block_ids[b0:b1]), dtype=torch.long, device=device)
    return idx, b1 - b0, first - b0 * bs


def read_fp8_ds_mla(cache: torch.Tensor, block_ids: Sequence[int], first: int, n: int) -> torch.Tensor:
    """Rows [first, first+n) of a fp8_ds_mla cache ([nblk, bs, 584] uint8) -> float32 [n, 512]."""
    if cache.dtype != torch.uint8 or cache.dim() != 3 or cache.shape[-1] != PAGE_ROW:
        raise ValueError(f"expected uint8 [nblk, bs, 584] fp8_ds_mla cache, got {cache.dtype} {tuple(cache.shape)}")
    bs = int(cache.shape[1])
    idx, nb, lo = _block_span(block_ids, bs, first, n, cache.device)
    pages = cache.index_select(0, idx).reshape(nb, bs * PAGE_ROW)  # index_select honours a padded stride(0)
    data = pages[:, : bs * ROW_BYTES].reshape(nb * bs, ROW_BYTES)[lo : lo + n]
    sc = pages[:, bs * ROW_BYTES :].reshape(nb * bs, SCALE_BYTES)[lo : lo + n]
    nope = data[:, :NOPE].contiguous().view(torch.float8_e4m3fn).to(torch.float32).view(n, NOPE // QBLOCK, QBLOCK)
    scale = torch.exp2(sc[:, : NOPE // QBLOCK].to(torch.float32) - 127.0)
    nope = (nope * scale.unsqueeze(-1)).reshape(n, NOPE)
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16).to(torch.float32)  # [n, 64]
    return torch.cat([nope, rope], dim=-1)


def read_indexer_fp8(cache: torch.Tensor, block_ids: Sequence[int], first: int, n: int) -> torch.Tensor:
    """Rows [first, first+n) of the FP8 indexer cache ([nblk, bs, 132] uint8) -> float32 [n, 128]."""
    if cache.dtype != torch.uint8 or cache.dim() != 3 or cache.shape[-1] != IDX_DIM + 4:
        raise ValueError(f"expected uint8 [nblk, bs, 132] indexer cache, got {cache.dtype} {tuple(cache.shape)}")
    bs = int(cache.shape[1])
    idx, nb, lo = _block_span(block_ids, bs, first, n, cache.device)
    pages = cache.index_select(0, idx).reshape(nb, bs * (IDX_DIM + 4))
    data = pages[:, : bs * IDX_DIM].reshape(nb * bs, IDX_DIM)[lo : lo + n]
    sc = pages[:, bs * IDX_DIM :].contiguous().view(torch.float32).reshape(nb * bs)[lo : lo + n]
    return data.contiguous().view(torch.float8_e4m3fn).to(torch.float32) * sc.unsqueeze(-1)


def read_state_rows(cache: torch.Tensor, block_ids: Sequence[int], first: int, n: int) -> torch.Tensor:
    """Per-token compressor state rows for positions [first, first+n) -> float32 [n, 2*coff*d]."""
    if cache.dtype != torch.float32 or cache.dim() != 3:
        raise ValueError(f"expected float32 [nblk, bs, 2*coff*d] state cache, got {cache.dtype} {tuple(cache.shape)}")
    bs = int(cache.shape[1])
    pos = torch.arange(first, first + n)
    blocks = [int(block_ids[int(p) // bs]) for p in pos]
    if any(b <= 0 for b in blocks):  # block 0 is vLLM's null block (freed, out of the sliding window)
        raise ValueError(f"state rows [{first},{first + n}) include a freed (null) block: {blocks}")
    bt = torch.as_tensor(blocks, dtype=torch.long, device=cache.device)
    off = (pos % bs).to(cache.device)
    return cache[bt, off].to(torch.float32)


# ----------------------------------------------------------------------------------------------- RoPE
def mlx_inv_freq(dims: int, base: float, yarn: dict[str, Any] | None) -> torch.Tensor:
    """inv_freq exactly as mlx_lm DeepseekV4RoPE builds it (deepseek_v4.py:104-139)."""
    inv = 1.0 / (base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims))
    if yarn:
        factor = float(yarn["factor"])
        orig = float(yarn["original_max_position_embeddings"])
        bf = float(yarn.get("beta_fast", 32))
        bsl = float(yarn.get("beta_slow", 1))

        def corr(r: float) -> float:
            return dims * math.log(orig / (r * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(bf)), 0)
        high: float = min(math.ceil(corr(bsl)), dims - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dims // 2, dtype=torch.float32) - low) / (high - low)
        smooth = 1 - torch.clamp(ramp, 0, 1)
        inv = inv / factor * (1 - smooth) + inv * smooth
    return inv


def rerope(x: torch.Tensor, pos: torch.Tensor, cos_sin_src: torch.Tensor, inv_freq_dst: torch.Tensor) -> torch.Tensor:
    """x: float32 [n, 512]; its last 64 dims were rotated (GPT-J interleaved pairs) with vLLM's cos_sin_cache rows
    cos_sin_src [n, 64] = [cos(32) | sin(32)]. Undo that rotation and apply mlx's (angle = pos * inv_freq_dst)."""
    half = ROPE // 2
    r = x[:, NOPE:]
    e, o = r[:, 0::2], r[:, 1::2]
    c, s = cos_sin_src[:, :half].to(torch.float32), cos_sin_src[:, half:].to(torch.float32)
    norm = c * c + s * s  # 1 unless an mscale was folded in
    e1 = (e * c + o * s) / norm
    o1 = (o * c - e * s) / norm
    ang = pos.to(torch.float32).unsqueeze(-1) * inv_freq_dst.to(x.device).unsqueeze(0)
    c2, s2 = torch.cos(ang), torch.sin(ang)
    out = torch.stack([e1 * c2 - o1 * s2, o1 * c2 + e1 * s2], dim=-1).reshape(x.shape[0], ROPE)
    return torch.cat([x[:, :NOPE], out], dim=-1)


# ----------------------------------------------------------------------------------------------- one layer
class LayerSources:
    """Everything the converter needs for one decoder layer. Any field may be None where the layer has no such cache."""

    def __init__(self, ratio: int):
        self.ratio = ratio
        self.swa: torch.Tensor | None = None
        self.swa_blocks: Sequence[int] | None = None
        self.pool: torch.Tensor | None = None
        self.pool_blocks: Sequence[int] | None = None
        self.state: torch.Tensor | None = None
        self.state_blocks: Sequence[int] | None = None
        self.ape: torch.Tensor | None = None
        self.idx_pool: torch.Tensor | None = None
        self.idx_pool_blocks: Sequence[int] | None = None
        self.idx_state: torch.Tensor | None = None
        self.idx_state_blocks: Sequence[int] | None = None
        self.idx_ape: torch.Tensor | None = None
        self.cos_sin: torch.Tensor | None = None  # vLLM rotary_emb.cos_sin_cache [max_pos, 64]
        self.rerope_inv_freq: torch.Tensor | None = None  # set to re-rope the SWA rows (layers with compress ratio 0)


def _need(x: Any, what: str) -> Any:
    if x is None:
        raise ValueError(f"layer source {what} is missing")
    return x


def _branch(
    state: torch.Tensor,
    state_blocks: Sequence[int],
    ape: torch.Tensor,
    pool_reader: Callable[[int], torch.Tensor],
    ratio: int,
    T: int,
    d: int,
) -> tuple[list[torch.Tensor | None], int]:
    coff = 2 if ratio == 4 else 1
    w = coff * d
    if state.shape[-1] != 2 * w:
        raise ValueError(f"state width {state.shape[-1]} != 2*{w} (ratio {ratio}, d {d})")
    n, r = divmod(T, ratio)
    c = n * ratio
    out: list[torch.Tensor | None] = [None] * BRANCH_ARRAYS
    if r:  # mlx buffer: raw kv / raw score (no ape) of positions [c, T)  (mlx deepseek_v4.py:1079-1082, 1621)
        rows = read_state_rows(state, state_blocks, c, r)
        pos = torch.arange(c, T, device=rows.device) % ratio
        out[0] = rows[:, :w]
        out[1] = rows[:, w:] - ape.to(rows.device, torch.float32)[pos]
    if ratio == 4 and n >= 1:  # mlx prev window [c-4, c): raw kv, score WITH ape (mlx deepseek_v4.py:1648-1649)
        rows = read_state_rows(state, state_blocks, c - ratio, ratio)
        out[2] = rows[:, :w]
        out[3] = rows[:, w:]
    if n:
        out[4] = pool_reader(n)
    return [None if a is None else a.to(torch.bfloat16).unsqueeze(0).contiguous() for a in out], r


def convert_layer(src: LayerSources, T: int) -> list[Blob]:
    """The wire blobs for one layer as (role, dtype, shape, tensor).

    Order and header as deepseek_v4_cache_to_blobs in exo-explore/exo#2545.
    """
    W = min(T, WINDOW)
    first = T - W
    x = read_fp8_ds_mla(_need(src.swa, "swa"), _need(src.swa_blocks, "swa_blocks"), first, W)  # [W, 512] post-RoPE
    if src.rerope_inv_freq is not None:
        pos = torch.arange(first, T, device=x.device)
        x = rerope(x, pos, _need(src.cos_sin, "cos_sin")[pos], src.rerope_inv_freq)
    keys = x.to(torch.bfloat16).reshape(1, 1, W, HEAD).contiguous()
    head = [float(WIRE_VERSION), 0.0, float(WINDOW), float(T), float(W), 1.0]  # version, keep, max_size, offset, _idx, present
    arrays: list[tuple[str, torch.Tensor]] = [("window_k", keys), ("window_v", keys)]  # K == V (shared MQA head)
    branches: list[tuple[list[torch.Tensor | None], int]] = []
    if src.ratio:
        pool, pool_blocks = _need(src.pool, "pool"), _need(src.pool_blocks, "pool_blocks")
        branches.append(
            _branch(
                _need(src.state, "state"),
                _need(src.state_blocks, "state_blocks"),
                _need(src.ape, "ape"),
                lambda n: read_fp8_ds_mla(pool, pool_blocks, 0, n),
                src.ratio,
                T,
                HEAD,
            )
        )
    else:
        branches.append(([None] * BRANCH_ARRAYS, 0))
    if src.ratio == 4:
        ipool, ipool_blocks = _need(src.idx_pool, "idx_pool"), _need(src.idx_pool_blocks, "idx_pool_blocks")
        branches.append(
            _branch(
                _need(src.idx_state, "idx_state"),
                _need(src.idx_state_blocks, "idx_state_blocks"),
                _need(src.idx_ape, "idx_ape"),
                lambda n: read_indexer_fp8(ipool, ipool_blocks, 0, n),
                4,
                T,
                IDX_DIM,
            )
        )
    else:
        branches.append(([None] * BRANCH_ARRAYS, 0))
    for key, (vals, count) in zip(BRANCH_KEYS, branches, strict=True):
        head += [0.0 if v is None else 1.0 for v in vals]
        head += [float(count), -1.0, -1.0]  # buffer_count, buffer_lengths None, pool_lengths None
        arrays += [(f"{key}.{name}", v) for name, v in zip(BRANCH_FIELDS, vals, strict=True) if v is not None]
    blobs: list[Blob] = [("state_header", "float32", (len(head),), torch.tensor(head, dtype=torch.float32))]
    blobs += [(role, "bfloat16", tuple(int(s) for s in a.shape), a) for role, a in arrays]
    return blobs


def layer_nbytes(T: int, ratio: int) -> int:
    """Bytes of convert_layer(src, T)'s blobs for a layer with compress ratio ``ratio``, without converting anything.

    Same arithmetic as convert_layer and _branch: the float32 header (6 fields, then 8 per branch), the window K and V
    (bf16 [min(T, 128), 512] each), then per branch the buffer kv and gate (T mod ratio rows of coff*d), the ratio-4
    previous window (4 rows of coff*d, each of kv and gate) and the pool (T // ratio rows of d), all bf16."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    w = min(T, WINDOW)
    total = 4 * (6 + 2 * (BRANCH_ARRAYS + 3)) + 2 * (w * HEAD * 2)
    branches = ([(ratio, HEAD)] if ratio else []) + ([(4, IDX_DIM)] if ratio == 4 else [])
    for r, d in branches:
        width = (2 if r == 4 else 1) * d
        n, rem = divmod(T, r)
        total += 2 * rem * width * 2  # buffer_kv, buffer_gate
        if r == 4 and n >= 1:
            total += 2 * r * width * 2  # prev_kv, prev_gate
        total += n * d * 2  # pool
    return total


def layer_capacity(T: int, ratio: int) -> int:
    """The most bytes any prompt of 1 to T tokens needs for this layer. Sizes are not monotonic in T (the buffer rows
    restart at every multiple of the ratio), but for each t <= T - 128 some t' in (T - 128, T] has the same remainder
    (both ratios divide 128) and a window and pool at least as large, so the last 128 lengths are enough."""
    return max(layer_nbytes(t, ratio) for t in range(max(1, T - WINDOW + 1), T + 1))


def blob_bytes(t: torch.Tensor) -> torch.Tensor:
    """Flat little-endian byte view of a tensor (bf16 as its raw 16-bit words, like EXO's array_to_bytes)."""
    return t.contiguous().view(torch.uint8).reshape(-1)
