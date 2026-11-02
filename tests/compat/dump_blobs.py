"""Write kvhandoff's DeepSeek V4 blobs for a few layers to a directory (run with the torch venv)."""

import json
import sys
from pathlib import Path

sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1] + "/tests"]
from kvhandoff.producer import dsv4_convert as cv
from test_dsv4 import fake_model

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cases = []
for t in (5, 300, 1030):
    ratios = [0, 4, 128]
    kv, groups, mods, bids, srcs = fake_model(t, ratios, seed=t)
    for i, r in enumerate(ratios):
        blobs = cv.convert_layer(srcs[i], t)
        name = f"T{t}_r{r}"
        meta = []
        with open(out / f"{name}.bin", "wb") as f:
            for role, dtype, shape, x in blobs:
                b = cv.blob_bytes(x).numpy().tobytes()
                meta.append({"role": role, "dtype": dtype, "shape": list(shape), "nbytes": len(b)})
                f.write(b)
        cases.append({"name": name, "T": t, "ratio": r, "blobs": meta})
(out / "cases.json").write_text(json.dumps(cases))
print(len(cases), "cases")
