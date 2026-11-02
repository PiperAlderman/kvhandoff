from __future__ import annotations

import json

import pytest

from kvhandoff.interchange import (
    FORMAT_VERSION,
    KIND_DSV4,
    KIND_PAGED,
    BlobRef,
    ExportHeader,
    FormatError,
    LayerRef,
    RopeInfo,
    UnsupportedFormatVersion,
    paged_layer,
    validate,
)
from kvhandoff.tags import is_valid_tag, new_tag, requested_tag, safe_tag


def _paged(tokens: int = 10, layers: int = 3) -> ExportHeader:
    return ExportHeader(
        kind=KIND_PAGED,
        tag="p-abc",
        model_id="llama70b",
        num_tokens=tokens,
        layers=tuple(
            paged_layer(
                i,
                f"model.layers.{i}.self_attn.attn",
                num_tokens=tokens,
                heads=8,
                head_dim=128,
                dtype="float16",
                row_capacity=64,
            )
            for i in range(layers)
        ),
        num_model_layers=80,
        rope=RopeInfo(style="neox", applied_by="vllm", theta=500000.0, scaling={"rope_type": "llama3", "factor": 8.0}),
        producer={"connector": "KVHandoffPagedConnector", "vllm": "0.30.0"},
        seq=7,
        slot=2,
        created_unix=1.5,
        extra={"export_seconds": 0.1},
    )


def _dsv4() -> ExportHeader:
    head = BlobRef("state_header", "float32", (22,), 0, 88)
    k = BlobRef("window_k", "bfloat16", (1, 1, 128, 512), 88, 128 * 512 * 2)
    v = BlobRef("window_v", "bfloat16", (1, 1, 128, 512), 88 + 128 * 512 * 2, 128 * 512 * 2)
    layer = LayerRef(
        index=0,
        name="model.layers.0.attn",
        file="L0.bin",
        nbytes=88 + 2 * 128 * 512 * 2,
        blobs=(head, k, v),
        attrs={"compress_ratio": 0},
    )
    return ExportHeader(
        kind=KIND_DSV4,
        tag="abc-1234",
        model_id="deepseek-v4-flash",
        num_tokens=300,
        layers=(layer,),
        num_model_layers=43,
        rope=RopeInfo(
            style="gptj", applied_by="mixed", theta=10000.0, scaling={"factor": 16}, details={"ratio0_layers": "mlx"}
        ),
    )


@pytest.mark.parametrize("make", [_paged, _dsv4])
def test_round_trip_is_exact(make: object) -> None:
    h = make()  # type: ignore[operator]
    back = ExportHeader.from_json(h.to_json())
    assert back == h
    assert back.to_json() == h.to_json()


def test_header_carries_versioned_model_layer_dtype_token_and_rope_fields() -> None:
    d = json.loads(_paged().to_json())
    assert d["format"] == "kvhandoff" and d["format_version"] == list(FORMAT_VERSION)
    assert d["model"] == {"id": "llama70b", "num_layers": 80}
    assert d["tokens"] == {"count": 10, "first_position": 0}
    assert d["rope"]["style"] == "neox" and d["rope"]["applied_by"] == "vllm" and d["rope"]["theta"] == 500000.0
    assert [layer["name"] for layer in d["layers"]][:1] == ["model.layers.0.self_attn.attn"]
    assert {b["dtype"] for layer in d["layers"] for b in layer["blobs"]} == {"float16"}


def test_paged_layer_offsets_follow_the_file_layout() -> None:
    layer = paged_layer(5, "x", num_tokens=10, heads=8, head_dim=128, dtype="bfloat16", row_capacity=64)
    k, v = layer.blobs
    assert (k.offset, k.nbytes) == (0, 10 * 8 * 128 * 2)
    assert v.offset == 64 * 8 * 128 * 2  # V starts after max_tokens rows of K
    assert layer.nbytes == v.offset + v.nbytes and layer.file == "L5.bin"


def test_unknown_major_is_rejected_and_unknown_minor_fields_are_ignored() -> None:
    d = json.loads(_paged().to_json())
    d["format_version"] = [2, 0]
    with pytest.raises(UnsupportedFormatVersion):
        ExportHeader.from_json(json.dumps(d))
    d["format_version"] = [1, 7]
    d["something_new"] = {"x": 1}
    d["layers"][0]["future_field"] = True
    h = ExportHeader.from_json(json.dumps(d))
    assert h.format_version == (1, 7) and len(h.layers) == 3


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.update(format="other"), "not a kvhandoff header"),
        (lambda d: d.update(kind="mystery"), "unknown kind"),
        (lambda d: d.update(tag="../etc"), "invalid tag"),
        (lambda d: d["layers"][0]["blobs"][0].update(dtype="int8"), "not one EXO's wire carries"),
        (lambda d: d["layers"][0]["blobs"][0].update(nbytes=3), "nbytes"),
        (lambda d: d["layers"][0]["blobs"][1].update(offset=10**9), "ends past"),
        (lambda d: d["layers"][1].update(index=0), "listed twice"),
        (lambda d: d["layers"][0].update(file="../x.bin"), "file name"),
        (lambda d: d["layers"][0]["blobs"].reverse(), "must be k, v"),
        (lambda d: d["tokens"].update(count=-1), "integer"),
        (lambda d: d.update(format_version="1.0"), "format_version"),
    ],
)
def test_malformed_headers_are_refused(mutate: object, match: str) -> None:
    d = json.loads(_paged().to_json())
    mutate(d)  # type: ignore[operator]
    with pytest.raises(FormatError, match=match):
        ExportHeader.from_json(json.dumps(d))


def test_not_json_is_a_format_error() -> None:
    with pytest.raises(FormatError):
        ExportHeader.from_json(b"\xff not json")


def test_failure_record_needs_no_layers() -> None:
    h = ExportHeader(kind=KIND_DSV4, tag="t-1", model_id="m", num_tokens=5, error="StoreFull: no slot")
    validate(h)
    back = ExportHeader.from_json(h.to_json())
    assert not back.ok and back.error == "StoreFull: no slot"


def test_dsv4_first_blob_must_be_the_state_header() -> None:
    h = _dsv4()
    layer = h.layers[0]
    bad = LayerRef(index=0, name="", file="L0.bin", nbytes=layer.nbytes, blobs=layer.blobs[1:])
    with pytest.raises(FormatError, match="state header"):
        validate(ExportHeader(kind=KIND_DSV4, tag="t", model_id="m", num_tokens=1, layers=(bad,)))


def test_tags() -> None:
    assert is_valid_tag("p-0123456789abcdef") and is_valid_tag("a.b_c-d")
    for bad in ("", ".hidden", "-x", "a/b", "a..b", "x" * 97, "spa ce"):
        assert not is_valid_tag(bad)
    assert safe_tag("good-tag", "req-1") == "good-tag"
    assert safe_tag("../evil", "cmpl-12/34") == "r-cmpl-12_34"
    assert is_valid_tag(safe_tag("", "..weird..id"))
    t1, t2 = new_tag(), new_tag()
    assert t1 != t2 and is_valid_tag(t1)


def test_requested_tag_reads_only_the_kvhandoff_key() -> None:
    assert requested_tag(None) is None
    assert requested_tag({}) is None
    assert requested_tag({"other": 1}) is None
    assert requested_tag({"kvhandoff_tag": "a"}) == "a"
    assert requested_tag({"kvhandoff_tag": ""}) is None
