"""The cache-interchange header: what an export holds, in a form a reader can check before it trusts any bytes.

One JSON document per export (``header.json`` in the export's slot, ``GET /v1/exports/<tag>`` on the transfer service).
It is written by the producer (a vLLM connector) and read by the transfer service and the EXO bridge.

Versioning: ``format_version`` is ``[major, minor]``. A reader rejects an unknown major and ignores fields it does not
know within a known major. Minor bumps may only add optional fields.

Two kinds of export exist:

``paged_kv``   standard attention (Llama-style). Each layer file holds K then V, each ``[T, H, D]`` contiguous in the
               blob's dtype (the producer pre-sizes files to ``max_tokens`` rows, so V starts at a fixed offset; the
               blob offsets say where). Sent to EXO as one ``KVChunk`` per layer.
``dsv4_state`` DeepSeek V4. Each layer file holds the blobs of one EXO ``ArraysState`` back to back: a float32 header
               then bf16 arrays, exactly the layout of ``deepseek_v4_cache_to_blobs`` in exo-explore/exo#2545 (its own
               wire version 1, carried in that float32 header and not changed here).

The per-layer EXO messages are not part of this header: the header describes storage, EXO's protocol describes the
stream, and the bridge maps one to the other without changing a byte of the DeepSeek blobs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Final

from kvhandoff.tags import is_valid_tag

FORMAT: Final = "kvhandoff"
FORMAT_VERSION: Final = (1, 0)
KIND_PAGED: Final = "paged_kv"
KIND_DSV4: Final = "dsv4_state"
KINDS: Final = frozenset({KIND_PAGED, KIND_DSV4})

# The dtypes EXO's wire carries (exo.worker.disaggregated.protocol.DType) and their item sizes.
DTYPE_SIZES: Final[dict[str, int]] = {"bfloat16": 2, "float16": 2, "float32": 4}

# Blob roles. paged_kv: k, v. dsv4_state: header (float32), window_k, window_v, then <branch>.<array>.
PAGED_ROLES: Final = ("k", "v")


class FormatError(ValueError):
    """The header is malformed or inconsistent."""


class UnsupportedFormatVersion(FormatError):
    """The header's major version is not one this reader understands."""


@dataclass(frozen=True)
class BlobRef:
    role: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "dtype": self.dtype, "shape": list(self.shape), "offset": self.offset, "nbytes": self.nbytes}


@dataclass(frozen=True)
class LayerRef:
    index: int  # global decoder-layer index (EXO's layer_idx)
    name: str  # the producer's layer name (vLLM attention layer), for the layer map
    file: str  # file name inside the slot
    nbytes: int  # bytes of the file that are meaningful (files may be pre-sized larger)
    blobs: tuple[BlobRef, ...]
    attrs: dict[str, Any] = field(default_factory=dict[str, Any])  # e.g. {"compress_ratio": 4} for DeepSeek V4

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "file": self.file,
            "nbytes": self.nbytes,
            "blobs": [b.to_dict() for b in self.blobs],
        }
        if self.attrs:
            d["attrs"] = dict(self.attrs)
        return d


@dataclass(frozen=True)
class RopeInfo:
    """How the positional rotation already inside the exported keys was applied, so a reader can refuse a mismatch.

    style       "neox" (rotate-half pairs, Llama in vLLM and MLX) or "gptj" (interleaved pairs, DeepSeek V4)
    applied_by  who rotated the keys as stored: "vllm" (untouched), "mlx" (re-rotated to MLX's convention)
    theta, scaling: the producer model's RoPE parameters (scaling as in the HF config, or None)
    per_layer_group: optional, e.g. DeepSeek V4: {"ratio0": "mlx", "compressed": "vllm"} and the compress theta
    """

    style: str
    applied_by: str
    theta: float | None = None
    scaling: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "applied_by": self.applied_by,
            "theta": self.theta,
            "scaling": self.scaling,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExportHeader:
    kind: str
    tag: str
    model_id: str
    num_tokens: int  # the export covers token positions [first_position, first_position + num_tokens)
    layers: tuple[LayerRef, ...] = ()
    num_model_layers: int = 0
    first_position: int = 0
    rope: RopeInfo | None = None
    producer: dict[str, Any] = field(default_factory=dict[str, Any])
    error: str | None = None
    seq: int = 0  # set by the store at publish; a reader re-checks it to detect a reused slot
    slot: int = -1
    created_unix: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict[str, Any])
    format_version: tuple[int, int] = FORMAT_VERSION

    @property
    def ok(self) -> bool:
        return self.error is None

    def layer(self, index: int) -> LayerRef:
        for layer in self.layers:
            if layer.index == index:
                return layer
        raise KeyError(f"export {self.tag} has no layer {index}")

    def total_bytes(self) -> int:
        return sum(sum(b.nbytes for b in layer.blobs) for layer in self.layers)

    def with_store_fields(self, *, seq: int, slot: int, created_unix: float) -> ExportHeader:
        return ExportHeader(
            kind=self.kind,
            tag=self.tag,
            model_id=self.model_id,
            num_tokens=self.num_tokens,
            layers=self.layers,
            num_model_layers=self.num_model_layers,
            first_position=self.first_position,
            rope=self.rope,
            producer=dict(self.producer),
            error=self.error,
            seq=seq,
            slot=slot,
            created_unix=created_unix,
            extra=dict(self.extra),
            format_version=self.format_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "format_version": list(self.format_version),
            "kind": self.kind,
            "tag": self.tag,
            "seq": self.seq,
            "slot": self.slot,
            "created_unix": self.created_unix,
            "model": {"id": self.model_id, "num_layers": self.num_model_layers},
            "tokens": {"count": self.num_tokens, "first_position": self.first_position},
            "rope": None if self.rope is None else self.rope.to_dict(),
            "producer": dict(self.producer),
            "error": self.error,
            "layers": [layer.to_dict() for layer in self.layers],
            "extra": dict(self.extra),
        }

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True).encode()

    @classmethod
    def from_json(cls, raw: bytes | str) -> ExportHeader:
        try:
            doc = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise FormatError(f"header is not JSON: {exc}") from exc
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc: Any) -> ExportHeader:
        d = _obj(doc, "header")
        if d.get("format") != FORMAT:
            raise FormatError(f"not a {FORMAT} header (format={d.get('format')!r})")
        ver = d.get("format_version")
        if not (isinstance(ver, list) and len(ver) == 2 and all(isinstance(v, int) for v in ver)):  # pyright: ignore[reportUnknownVariableType]
            raise FormatError(f"bad format_version {ver!r}")
        major, minor = int(ver[0]), int(ver[1])  # pyright: ignore[reportUnknownArgumentType]
        if major != FORMAT_VERSION[0]:
            raise UnsupportedFormatVersion(f"format version {major}.{minor}; this reader understands {FORMAT_VERSION[0]}.x")
        kind = _str(d.get("kind"), "kind")
        if kind not in KINDS:
            raise FormatError(f"unknown kind {kind!r}")
        tag = _str(d.get("tag"), "tag")
        if not is_valid_tag(tag):
            raise FormatError(f"invalid tag {tag!r}")
        model = _obj(d.get("model"), "model")
        tokens = _obj(d.get("tokens"), "tokens")
        error = d.get("error")
        if error is not None and not isinstance(error, str):
            raise FormatError("error must be a string or null")
        rope_d = d.get("rope")
        rope = None
        if rope_d is not None:
            r = _obj(rope_d, "rope")
            scaling = r.get("scaling")
            rope = RopeInfo(
                style=_str(r.get("style"), "rope.style"),
                applied_by=_str(r.get("applied_by"), "rope.applied_by"),
                theta=None if r.get("theta") is None else float(r["theta"]),
                scaling=None if scaling is None else dict(_obj(scaling, "rope.scaling")),
                details=dict(_obj(r.get("details", {}), "rope.details")),
            )
        layers = tuple(_layer(x) for x in _list(d.get("layers", []), "layers"))
        hdr = cls(
            kind=kind,
            tag=tag,
            model_id=_str(model.get("id"), "model.id"),
            num_tokens=_int(tokens.get("count"), "tokens.count"),
            first_position=_int(tokens.get("first_position", 0), "tokens.first_position"),
            layers=layers,
            num_model_layers=_int(model.get("num_layers", 0), "model.num_layers"),
            rope=rope,
            producer=dict(_obj(d.get("producer", {}), "producer")),
            error=error,
            seq=_int(d.get("seq", 0), "seq"),
            slot=_int(d.get("slot", -1), "slot", minimum=-1),
            created_unix=float(d.get("created_unix", 0.0)),
            extra=dict(_obj(d.get("extra", {}), "extra")),
            format_version=(major, minor),
        )
        validate(hdr)
        return hdr


def validate(h: ExportHeader) -> None:
    """Consistency checks shared by writer (before publish) and reader (after decode)."""
    if h.kind not in KINDS:
        raise FormatError(f"unknown kind {h.kind!r}")
    if not is_valid_tag(h.tag):
        raise FormatError(f"invalid tag {h.tag!r}")
    if h.num_tokens < 0 or h.first_position < 0:
        raise FormatError("token counts must be >= 0")
    if h.error is not None:
        return  # a failure record carries no layers worth checking
    seen: set[int] = set()
    files: set[str] = set()
    for layer in h.layers:
        if layer.index in seen:
            raise FormatError(f"layer {layer.index} listed twice")
        seen.add(layer.index)
        if layer.file in files or "/" in layer.file or layer.file.startswith("."):
            raise FormatError(f"bad or duplicate layer file name {layer.file!r}")
        files.add(layer.file)
        for b in layer.blobs:
            size = DTYPE_SIZES.get(b.dtype)
            if size is None:
                raise FormatError(f"layer {layer.index}: dtype {b.dtype!r} is not one EXO's wire carries")
            if any(s < 0 for s in b.shape) or b.offset < 0:
                raise FormatError(f"layer {layer.index}: negative shape or offset")
            if math.prod(b.shape) * size != b.nbytes:
                raise FormatError(f"layer {layer.index} blob {b.role}: nbytes {b.nbytes} != prod{b.shape} x {size}")
            if b.offset + b.nbytes > layer.nbytes:
                raise FormatError(f"layer {layer.index} blob {b.role}: ends past the layer's {layer.nbytes} bytes")
        if h.kind == KIND_PAGED:
            roles = tuple(b.role for b in layer.blobs)
            if roles != PAGED_ROLES:
                raise FormatError(f"paged_kv layer {layer.index}: blobs must be k, v (got {roles})")
            k, v = layer.blobs
            if k.shape != v.shape or k.dtype != v.dtype or len(k.shape) != 3 or k.shape[0] != h.num_tokens:
                raise FormatError(f"paged_kv layer {layer.index}: k/v must both be [tokens, heads, head_dim]")
        elif not layer.blobs or layer.blobs[0].dtype != "float32" or len(layer.blobs[0].shape) != 1:
            raise FormatError(f"dsv4_state layer {layer.index}: first blob must be the float32 state header")


def paged_layer(
    index: int, name: str, *, num_tokens: int, heads: int, head_dim: int, dtype: str, row_capacity: int
) -> LayerRef:
    """The LayerRef for a paged_kv layer file laid out as [2, row_capacity, heads, head_dim] (K block then V block)."""
    size = DTYPE_SIZES[dtype]
    nbytes = num_tokens * heads * head_dim * size
    v_off = row_capacity * heads * head_dim * size
    blobs = (
        BlobRef("k", dtype, (num_tokens, heads, head_dim), 0, nbytes),
        BlobRef("v", dtype, (num_tokens, heads, head_dim), v_off, nbytes),
    )
    return LayerRef(index=index, name=name, file=f"L{index}.bin", nbytes=v_off + nbytes, blobs=blobs)


# ----------------------------------------------------------------------------------------------- strict JSON helpers
def _obj(x: Any, what: str) -> dict[str, Any]:
    if not isinstance(x, dict):
        raise FormatError(f"{what} must be an object")
    return x  # pyright: ignore[reportUnknownVariableType]


def _list(x: Any, what: str) -> list[Any]:
    if not isinstance(x, list):
        raise FormatError(f"{what} must be a list")
    return x  # pyright: ignore[reportUnknownVariableType]


def _str(x: Any, what: str) -> str:
    if not isinstance(x, str):
        raise FormatError(f"{what} must be a string")
    return x


def _int(x: Any, what: str, minimum: int = 0) -> int:
    if not isinstance(x, int) or isinstance(x, bool) or x < minimum:
        raise FormatError(f"{what} must be an integer >= {minimum}")
    return x


def _layer(x: Any) -> LayerRef:
    d = _obj(x, "layer")
    blobs: list[BlobRef] = []
    for bx in _list(d.get("blobs"), "layer.blobs"):
        b = _obj(bx, "blob")
        shape = tuple(_int(s, "blob.shape[]") for s in _list(b.get("shape"), "blob.shape"))
        blobs.append(
            BlobRef(
                role=_str(b.get("role"), "blob.role"),
                dtype=_str(b.get("dtype"), "blob.dtype"),
                shape=shape,
                offset=_int(b.get("offset"), "blob.offset"),
                nbytes=_int(b.get("nbytes"), "blob.nbytes"),
            )
        )
    return LayerRef(
        index=_int(d.get("index"), "layer.index"),
        name=_str(d.get("name", ""), "layer.name"),
        file=_str(d.get("file"), "layer.file"),
        nbytes=_int(d.get("nbytes"), "layer.nbytes"),
        blobs=tuple(blobs),
        attrs=dict(_obj(d.get("attrs", {}), "layer.attrs")),
    )
