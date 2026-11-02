"""Feed kvhandoff's blobs to the receiver of EXO's DeepSeek V4 cache transport (https://github.com/exo-explore/exo/pull/2545,
inject_deepseek_v4_cache), re-serialise with its sender and compare bytes; also check the rebuilt cache's offset and
window (run with an MLX venv, PYTHONPATH=<EXO checkout with that PR applied>/src)."""

import json
import sys
from pathlib import Path

from exo.worker.disaggregated.protocol import TensorBlob
from exo.worker.engines.mlx.disaggregated.adapter import deepseek_v4_cache_to_blobs, inject_deepseek_v4_cache
from mlx_lm.models.deepseek_v4 import DeepseekV4Cache

d = Path(sys.argv[1])
ok = 0
cases = json.loads((d / "cases.json").read_text())
for c in cases:
    raw = (d / f"{c['name']}.bin").read_bytes()
    pos = 0
    blobs = []
    for b in c["blobs"]:
        blobs.append(TensorBlob(dtype=b["dtype"], shape=tuple(b["shape"]), data=raw[pos : pos + b["nbytes"]]))
        pos += b["nbytes"]
    cache = DeepseekV4Cache(128)
    inject_deepseek_v4_cache(cache, blobs)
    back = deepseek_v4_cache_to_blobs(cache)
    same = [(x.dtype, tuple(x.shape), x.data) for x in back] == [(x.dtype, tuple(x.shape), x.data) for x in blobs]
    good = same and cache.offset == c["T"] and cache.local.offset == c["T"]
    ok += good
    print(
        c["name"],
        "offset",
        cache.offset,
        "window",
        None if cache.local.keys is None else tuple(cache.local.keys.shape),
        "byte-identical re-serialise:",
        same,
    )
print(f"DSV4-COMPAT {ok}/{len(cases)}")
sys.exit(0 if ok == len(cases) else 1)
