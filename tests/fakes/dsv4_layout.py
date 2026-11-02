"""vLLM's DeepSeek V4 cache layouts, fabricated on CPU with an independent encoder.

Moved from the prototype self-test's input generator (same seeding and draw order, so the inputs are the ones the
prototype's 570 checks ran on). ``make_layer`` takes the LayerSources class so the same inputs can feed both the
prototype converter (golden generation) and kvhandoff's.
"""

from __future__ import annotations

import math
from typing import Any

import torch

BF, F8, U8 = torch.bfloat16, torch.float8_e4m3fn, torch.uint8
YARN: dict[str, Any] = {"factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1}
HEAD, ROPE, NOPE, QBLOCK, ROW_BYTES, SCALE_BYTES, IDX_DIM, WINDOW = 512, 64, 448, 64, 576, 8, 128, 128


class Rng:
    def __init__(self, seed: int = 1234):
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(self, *shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=self.g) * scale).to(BF).float()  # bf16-representable fp32

    def block_table(self, n_logical: int, nblk: int) -> list[int]:
        perm = (torch.randperm(nblk - 1, generator=self.g) + 1).tolist()  # never the null block 0
        return perm[:n_logical]


def vllm_yarn_inv_freq(dim: int, base: float, y: dict[str, Any]) -> torch.Tensor:
    """DeepseekScalingRotaryEmbedding._compute_inv_freq (deepseek_scaling_rope.py:78-107), extrapolation_factor 1."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (y["factor"] * pos_freqs)

    def cdim(n: float) -> float:
        return (dim * math.log(y["original_max_position_embeddings"] / (n * 2 * math.pi))) / (2 * math.log(base))

    low: float = max(math.floor(cdim(y["beta_fast"])), 0)
    high: float = min(math.ceil(cdim(y["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return inter * (1 - mask) + extra * mask


def cos_sin_cache(inv_freq: torch.Tensor, max_pos: int) -> torch.Tensor:
    t = torch.arange(max_pos, dtype=torch.float32)
    f = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def rot_ref(x64: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    z = torch.view_as_complex(x64.double().reshape(*x64.shape[:-1], -1, 2).contiguous())
    ang = pos.double().unsqueeze(-1) * inv_freq.double().unsqueeze(0)
    return torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).reshape(x64.shape).float()


def ue8m0_quant(v: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = v.shape
    q = v.to(BF).float().view(n, d // block, block)
    e = torch.ceil(torch.log2(q.abs().amax(-1).clamp_min(1e-4) / 448.0))
    f8 = (q * torch.exp2(-e).unsqueeze(-1)).clamp(-448, 448).to(F8)
    return f8.reshape(n, d), e


def paged(
    rows: torch.Tensor,
    block_ids: list[int],
    bs: int,
    nblk: int,
    row_bytes: int,
    tail: torch.Tensor | None,
    tail_bytes: int,
    pad: int = 64,
) -> torch.Tensor:
    per = row_bytes + tail_bytes
    stride0 = bs * per + pad
    raw = torch.zeros(nblk * stride0, dtype=U8)
    for i in range(rows.shape[0]):
        b = block_ids[i // bs]
        if b == 0:
            continue
        o = i % bs
        raw[b * stride0 + o * row_bytes : b * stride0 + (o + 1) * row_bytes] = rows[i]
        if tail is not None:
            t0 = b * stride0 + bs * row_bytes + o * tail_bytes
            raw[t0 : t0 + tail_bytes] = tail[i]
    return raw.as_strided((nblk, bs, per), (stride0, per, 1))


def enc_ds_mla(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    f8, e = ue8m0_quant(x[:, :NOPE], QBLOCK)
    rope = x[:, NOPE:].to(BF).contiguous().view(U8).reshape(n, 2 * ROPE)
    rows = torch.cat([f8.view(U8), rope], dim=-1)
    sc = torch.zeros(n, SCALE_BYTES, dtype=U8)
    sc[:, :7] = (e + 127).clamp(0, 255).to(U8)
    return rows, sc


def rope_tables(max_t: int) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """vLLM-side cos/sin caches (YaRN on every layer) and mlx-side inverse frequencies, as the selftest builds them."""
    from kvhandoff.producer import dsv4_convert as cv

    inv = {"plain": cv.mlx_inv_freq(64, 10000.0, None), "comp": cv.mlx_inv_freq(64, 160000.0, YARN)}
    vp, vc = vllm_yarn_inv_freq(64, 10000.0, YARN), vllm_yarn_inv_freq(64, 160000.0, YARN)
    cs = {"plain_yarn": {"inv": vp, "cache": cos_sin_cache(vp, max_t)}, "comp": {"inv": vc, "cache": cos_sin_cache(vc, max_t)}}
    return cs, inv


def make_layer(
    rng: Rng,
    sources_cls: Any,
    ratio: int,
    T: int,
    cs_vllm: dict[str, dict[str, torch.Tensor]],
    inv: dict[str, torch.Tensor],
) -> Any:
    """One layer's vLLM caches for a T-token prompt, as a ``sources_cls(ratio)`` (a LayerSources)."""
    src = sources_cls(ratio)
    raw = rng.rnd(T, HEAD)
    pos = torch.arange(T)
    cs = cs_vllm["comp"] if ratio else cs_vllm["plain_yarn"]
    stored = raw.clone()
    stored[:, NOPE:] = rot_ref(raw[:, NOPE:], pos, cs["inv"])
    rows, sc = enc_ds_mla(stored)
    nb = (T + 63) // 64
    bids = rng.block_table(nb, nb + 5)
    for j in range(nb):
        if (j + 1) * 64 <= T - WINDOW:
            bids[j] = 0
    src.swa = paged(rows, bids, 64, nb + 5, ROW_BYTES, sc, SCALE_BYTES)
    src.swa_blocks = bids
    src.cos_sin = cs["cache"]
    if ratio == 0:
        src.rerope_inv_freq = inv["plain"]
    if ratio:
        for kind, d in (("comp", HEAD),) + ((("idx", IDX_DIM),) if ratio == 4 else ()):
            coff = 2 if ratio == 4 else 1
            w = coff * d
            kv, score, ape = rng.rnd(T, w), rng.rnd(T, w), rng.rnd(ratio, w, scale=0.5)
            st_rows = torch.cat([kv, score + ape[pos % ratio]], dim=-1)
            sbs, swin = (4, 8) if ratio == 4 else (8, 128)
            snb = (T + sbs - 1) // sbs
            sb = rng.block_table(snb, snb + 3)
            for j in range(snb):
                if (j + 1) * sbs <= T - swin:
                    sb[j] = 0
            state = torch.zeros(snb + 3, sbs, 2 * w)
            for p in range(T):
                if sb[p // sbs]:
                    state[sb[p // sbs], p % sbs] = st_rows[p]
            n = T // ratio
            pool_true = rng.rnd(max(n, 1), d)
            s_per = 256 // ratio
            pnb = max((n + s_per - 1) // s_per, 1)
            pb = rng.block_table(pnb, pnb + 2)
            if kind == "comp":
                prow, psc = enc_ds_mla(pool_true)
                src.pool, src.pool_blocks = paged(prow, pb, s_per, pnb + 2, ROW_BYTES, psc, SCALE_BYTES), pb
                src.state, src.state_blocks, src.ape = state, sb, ape
            else:
                f8i, ei = ue8m0_quant(pool_true, IDX_DIM)
                scl = torch.exp2(ei).reshape(-1).to(torch.float32).contiguous().view(U8).reshape(-1, 4)
                src.idx_pool, src.idx_pool_blocks = paged(f8i.view(U8), pb, s_per, pnb + 2, IDX_DIM, scl, 4), pb
                src.idx_state, src.idx_state_blocks, src.idx_ape = state, sb, ape
    return src


def expected_header(ratio: int, T: int) -> list[float]:
    w = min(T, 128)
    h = [1, 0, 128, T, w, 1]
    for kind in ("comp", "idx"):
        if (kind == "comp" and ratio) or (kind == "idx" and ratio == 4):
            n, r = divmod(T, ratio)
            h += [int(r > 0), int(r > 0), int(ratio == 4 and n > 0), int(ratio == 4 and n > 0), int(n > 0), r, -1, -1]
        else:
            h += [0, 0, 0, 0, 0, 0, -1, -1]
    return [float(x) for x in h]
