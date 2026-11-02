"""Regenerate dsv4_golden.json from the PROTOTYPE converter (dsv4_convert.py), the code that ran on
hardware. kvhandoff's converter must reproduce these digests byte for byte (tests/test_dsv4.py).

  python tests/data/make_dsv4_golden.py <directory holding the prototype dsv4_convert.py>

Inputs come from tests/fakes/dsv4_layout.py (the selftest's phase A generator, same seed and draw order).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from fakes import dsv4_layout as fk  # noqa: E402

CASES = [(t, r) for t in (1, 3, 4, 5, 9, 127, 128, 129, 257, 515, 1030) for r in (0, 4, 128)]


def digest_blobs(blobs: list[tuple[str, tuple[int, ...], bytes]]) -> list[list[object]]:
    return [[dtype, list(shape), hashlib.sha256(data).hexdigest()] for dtype, shape, data in blobs]


def main() -> int:
    proto_dir = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("proto_dsv4_convert", proto_dir / "dsv4_convert.py")
    assert spec is not None and spec.loader is not None
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    rng = fk.Rng(1234)
    cs, inv = fk.rope_tables(1100)
    out: dict[str, Any] = {"source": "prototype dsv4_convert.py", "seed": 1234, "cases": []}
    for t, ratio in CASES:
        src = fk.make_layer(rng, cv.LayerSources, ratio, t, cs, inv)
        blobs = cv.convert_layer(src, t)
        flat = [(d, tuple(s), cv.blob_bytes(x).numpy().tobytes()) for d, s, x in blobs]
        out["cases"].append({"T": t, "ratio": ratio, "blobs": digest_blobs(flat)})
    (HERE / "dsv4_golden.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out['cases'])} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
