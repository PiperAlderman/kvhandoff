"""The bridge's pre-header budget scales with the prompt, as EXO's first-layer wait does (stdlib only, no torch)."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from kvhandoff.config import BridgeConfig, ConfigError
from kvhandoff.exo.bridge import Bridge
from kvhandoff.interchange import ExportHeader, LayerRef
from kvhandoff.store import StreamState
from kvhandoff.transfer.client import Buffer, Deadline, HandoffError, VllmClient

BASE = {"mode": "paged", "vllm_url": "http://vllm:8000", "vllm_model": "m", "local_store_path": "/s"}
DSV4 = {"mode": "dsv4", "vllm_url": "http://vllm:8000", "vllm_model": "m", "transfer_url": "http://t:1"}


def cfg(**kw: Any) -> BridgeConfig:
    return BridgeConfig.from_mapping({**BASE, **kw})


def test_defaults_track_exo_first_layer_wait_ten_seconds_inside() -> None:
    c = cfg()
    assert (c.timeout_s, c.timeout_per_1k_tokens_s, c.timeout_max_s) == (50.0, 4.0, 600.0)
    for tokens in (0, 2_000, 8_000, 27_000, 32_000, 100_000):
        exo_wait = 60 + 4 * tokens / 1000  # EXO_REMOTE_PREFILL_WAIT_SECONDS + ..._PER_1K_TOKENS (exo-explore/exo#2575 defaults)
        assert c.prefill_timeout_s(tokens) == pytest.approx(exo_wait - 10)
    assert c.prefill_timeout_s(27_000) == pytest.approx(158.0)
    assert c.prefill_timeout_s(1_000_000) == 600.0  # capped
    assert c.prefill_timeout_s(-5) == 50.0


def test_an_explicit_timeout_s_alone_stays_fixed_as_before() -> None:
    c = cfg(timeout_s=300)
    assert c.timeout_per_1k_tokens_s == 0.0 and c.timeout_max_s == 600.0
    assert c.prefill_timeout_s(0) == c.prefill_timeout_s(32_000) == 300.0
    big = cfg(timeout_s=900)  # the cap never undercuts a fixed budget
    assert big.timeout_max_s == 900.0 and big.prefill_timeout_s(32_000) == 900.0


def test_timeout_s_with_a_new_key_is_the_base() -> None:
    c = cfg(timeout_s=30, timeout_per_1k_tokens_s=2)
    assert c.prefill_timeout_s(10_000) == pytest.approx(50.0)
    c = cfg(timeout_s=30, timeout_max_s=120)
    assert c.timeout_per_1k_tokens_s == 4.0 and c.prefill_timeout_s(50_000) == 120.0
    assert cfg(timeout_per_1k_tokens_s=0).prefill_timeout_s(32_000) == 50.0


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"timeout_per_1k_tokens_s": -1}, "timeout_per_1k_tokens_s"),
        ({"timeout_per_1k_tokens_s": True}, "timeout_per_1k_tokens_s"),
        ({"timeout_per_1k_tokens_s": "4"}, "timeout_per_1k_tokens_s"),
        ({"timeout_max_s": 0.5}, "timeout_max_s"),
        ({"timeout_max_s": False}, "timeout_max_s"),
        ({"timeout_max_s": 40}, r"'timeout_max_s' \(40\) must be >= 'timeout_s' \(50\)"),
        ({"timeout_s": 100, "timeout_max_s": 99}, "must be >="),
        ({"timeout_s": 0.5}, "timeout_s"),
    ],
)
def test_bad_timeout_values_are_refused(bad: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        cfg(**bad)


class _CaptureVllm(VllmClient):
    """Records the budget of the deadline the bridge hands vLLM, then fails the hand-off."""

    def __init__(self) -> None:
        super().__init__("http://vllm:8000", "m")
        self.budgets: list[float] = []

    def prefill(self, tokens: list[int], kv_transfer_params: dict[str, object], deadline: Deadline) -> float:
        self.budgets.append(deadline.budget)
        raise HandoffError("captured")


class _NoExport:
    """An export source that never has anything (vLLM fails first)."""

    def wait_header(self, tag: str, deadline: Deadline) -> ExportHeader:
        raise HandoffError("no export")

    def stream_state(self, tag: str, deadline: Deadline) -> StreamState | None:
        return None

    def layer_blobs(self, header: ExportHeader, layer: LayerRef, deadline: Deadline, *, live: bool = False) -> list[Buffer]:
        raise HandoffError("no export")

    def release(self, header: ExportHeader) -> None:
        return None


def _req(n: int, start: int) -> Any:
    return SimpleNamespace(request_id="r", model_id="m", token_ids=list(range(n)), start_pos=start)


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_paged_budget_counts_the_tokens_to_prefill(stream: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kvhandoff.exo.bridge.exo_protocol", lambda: None)  # never reached: vLLM fails first
    v = _CaptureVllm()
    br = Bridge(cfg(stream_layers=stream), source=_NoExport(), vllm=v)
    try:
        with pytest.raises(HandoffError, match="captured"):
            br.resolve(_req(27_000, 0), io.BytesIO())
        with pytest.raises(HandoffError, match="captured"):
            br.resolve(_req(32_000, 30_000), io.BytesIO())  # EXO counts len(token_ids) - start_pos
    finally:
        br.close()
    assert v.budgets == [pytest.approx(158.0), pytest.approx(58.0)]


def test_dsv4_budget_counts_the_tokens_to_prefill() -> None:
    v = _CaptureVllm()
    br = Bridge(BridgeConfig.from_mapping({**DSV4, "transfer_token_file": "/t"}), source=_NoExport(), vllm=v)
    try:
        with pytest.raises(HandoffError, match="captured"):
            br.dsv4_result(list(range(8_000)), 2_000)
        with pytest.raises(HandoffError, match="captured"):
            br.dsv4_result(list(range(1_000)))
    finally:
        br.close()
    assert v.budgets == [pytest.approx(74.0), pytest.approx(54.0)]


def test_busy_poll_back_off_never_overflows() -> None:
    """A long prompt's budget can outlast 1,750 polls of a busy (503) transfer service; 1.5 ** 1750 overflows."""
    from kvhandoff.transfer.client import _poll_sleep

    assert _poll_sleep(0) == pytest.approx(0.005)
    for attempt in (100, 1_750, 10_000):
        assert _poll_sleep(attempt) == pytest.approx(0.1)


def test_shipped_examples_scale_the_budget() -> None:
    from pathlib import Path

    from kvhandoff.config import load_toml

    ex = Path(__file__).resolve().parent.parent / "deploy" / "examples"
    for name in ("bridge-llama.toml", "bridge-dsv4.toml"):
        c = BridgeConfig.from_mapping(load_toml(str(ex / name), "bridge"))
        assert c.prefill_timeout_s(27_000) == pytest.approx(158.0), name
