"""Clients the EXO bridge uses: the transfer service (HTTP, bearer token), a store on the same host (direct reads),
and vLLM's OpenAI endpoint (the prefill request). All calls take a deadline; nothing blocks past it."""

from __future__ import annotations

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from kvhandoff.auth import bearer_header
from kvhandoff.interchange import ExportHeader, FormatError, LayerRef
from kvhandoff.store import ExportFailed, ExportNotFound, ExportReader, StreamState


class HandoffError(RuntimeError):
    """A hand-off failed; the EXO side answers with an error and the Mac prefills locally."""


class DeadlineExceeded(HandoffError, TimeoutError):
    pass


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds
        self.budget = seconds

    def left(self, cap: float | None = None) -> float:
        t = self.end - time.monotonic()
        if t <= 0:
            raise DeadlineExceeded(f"hand-off exceeded its {self.budget:.0f}s budget")
        return t if cap is None else min(t, cap)


Buffer = bytes | memoryview  # blob payloads: views into one response body, never copied per blob


class ExportSource(Protocol):
    """Where the bridge gets an export from."""

    def wait_header(self, tag: str, deadline: Deadline) -> ExportHeader: ...

    def stream_state(self, tag: str, deadline: Deadline) -> StreamState | None:
        """The export as streamed so far, or published; None while there is nothing to see yet. Raises HandoffError
        for a failed export."""
        ...

    def layer_blobs(self, header: ExportHeader, layer: LayerRef, deadline: Deadline, *, live: bool = False) -> list[Buffer]:
        """``live``: ``header`` came from stream_state() and the export may still be streaming."""
        ...

    def release(self, header: ExportHeader) -> None: ...


def _poll_sleep(attempt: int) -> float:
    """Back-off for a busy transfer service (503): 5 ms growing to 100 ms."""
    return min(0.1, 0.005 * (1.5**attempt))


# wait_header runs after vLLM has answered the prefill request, when the export is a few tens of milliseconds from
# being published (its own copy), so it polls every HEADER_POLL_FAST seconds for the first HEADER_POLL_FAST_FOR
# seconds and only then backs off. It used to back off from 5 ms to 100 ms at once, so the poll after the publish
# came 32 ms late on median and up to 103 ms (measured on hardware). A 404 costs the service about 0.1 to 0.2 ms.
HEADER_POLL_FAST = 0.003
HEADER_POLL_FAST_FOR = 2.0
HEADER_POLL_SLOW = 0.02
HEADER_POLL_SLOW_AFTER = 10.0
HEADER_POLL_SLOWEST = 0.05


def header_poll_sleep(waited: float) -> float:
    """Seconds to sleep before the next header poll, after ``waited`` seconds of waiting."""
    if waited < HEADER_POLL_FAST_FOR:
        return HEADER_POLL_FAST
    return HEADER_POLL_SLOW if waited < HEADER_POLL_SLOW_AFTER else HEADER_POLL_SLOWEST


def _check_header(h: ExportHeader, tag: str) -> ExportHeader:
    if h.tag != tag:
        raise HandoffError(f"asked for export {tag}, got {h.tag}")
    if h.error is not None:
        raise HandoffError(f"producer could not export {tag}: {h.error}")
    return h


# ----------------------------------------------------------------------------------------------- HTTP
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is answered as the HTTP error it is, never followed: urllib would resend the Authorization header."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


# Every request that carries a bearer token goes through this opener: redirects are not followed, and proxy settings
# from the environment (http_proxy and friends) are ignored, so a token only ever goes to the configured URL.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


class TransferClient:
    def __init__(self, base_url: str, token: str):
        self.base = base_url.rstrip("/")
        self._auth = bearer_header(token)

    def _request(
        self, method: str, path: str, timeout: float, headers: dict[str, str] | None = None
    ) -> tuple[int, bytes, http.client.HTTPMessage | None]:
        req = urllib.request.Request(
            self.base + path,
            method=method,
            data=b"" if method == "POST" else None,
            headers={**self._auth, **(headers or {})},
        )
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            e.read()
            return e.code, b"", None
        except (urllib.error.URLError, OSError) as e:  # refused, reset, timed out: fail now, do not poll a dead service
            raise HandoffError(f"transfer service {self.base} unreachable: {e}") from e

    def wait_header(self, tag: str, deadline: Deadline) -> ExportHeader:
        t0 = time.monotonic()
        busy = 0
        while True:
            code, body, _ = self._request("GET", f"/v1/exports/{tag}", deadline.left(10.0))
            if code == 200:
                try:
                    return _check_header(ExportHeader.from_json(body), tag)
                except FormatError as exc:
                    raise HandoffError(f"export {tag}: bad header: {exc}") from exc
            if code == 401:
                raise HandoffError("transfer service refused the token (401)")
            if code not in (404, 503):
                raise HandoffError(f"transfer service answered {code} for {tag}")
            if code == 503:  # busy: back off on its own schedule, never at the fast header pace
                time.sleep(min(_poll_sleep(busy + 6), deadline.left()))
                busy += 1
                continue
            busy = 0
            time.sleep(min(header_poll_sleep(time.monotonic() - t0), deadline.left()))

    def stream_state(self, tag: str, deadline: Deadline) -> StreamState | None:
        code, body, _ = self._request("GET", f"/v1/exports/{tag}/stream", deadline.left(10.0))
        if code in (404, 503):
            return None
        if code == 401:
            raise HandoffError("transfer service refused the token (401)")
        if code != 200:
            raise HandoffError(f"transfer service answered {code} for the stream of {tag}")
        try:
            doc = json.loads(body)
            h = _check_header(ExportHeader.from_dict(doc["header"]), tag)
            return StreamState(h, published=bool(doc["published"]))
        except (ValueError, KeyError, TypeError) as exc:  # FormatError is a ValueError
            raise HandoffError(f"export {tag}: bad stream state: {exc}") from exc

    def layer_blobs(self, header: ExportHeader, layer: LayerRef, deadline: Deadline, *, live: bool = False) -> list[Buffer]:
        attempt = 0
        extra = {"X-KVH-Seq": str(header.seq)}
        if live:
            extra["X-KVH-Live"] = "1"
        while True:
            code, body, hdrs = self._request(
                "GET", f"/v1/exports/{header.tag}/layers/{layer.index}", deadline.left(30.0), extra
            )
            if code == 503:
                time.sleep(min(_poll_sleep(attempt + 6), deadline.left()))
                attempt += 1
                continue
            break
        if code == 404:
            raise HandoffError(f"export {header.tag} is gone (evicted or reused) at layer {layer.index}")
        if code != 200 or hdrs is None:
            raise HandoffError(f"layer {layer.index} of {header.tag}: HTTP {code}")
        if hdrs.get("X-KVH-Seq") != str(header.seq):
            raise HandoffError(f"export {header.tag} was reused while fetching layer {layer.index}")
        want = sum(b.nbytes for b in layer.blobs)
        if len(body) != want:
            raise HandoffError(f"layer {layer.index} of {header.tag}: {len(body)} of {want} bytes")
        out: list[Buffer] = []
        pos = 0
        view = memoryview(body)
        for b in layer.blobs:
            out.append(view[pos : pos + b.nbytes])  # zero-copy slice of the response body
            pos += b.nbytes
        return out

    def release(self, header: ExportHeader) -> None:
        with contextlib.suppress(HandoffError):  # best effort: the producer's hold_seconds frees the slot anyway
            self._request("POST", f"/v1/exports/{header.tag}/release?seq={header.seq}", 5.0)


# ----------------------------------------------------------------------------------------------- same host
class LocalSource:
    """Reads a store directly (bridge on the same host as vLLM, store group-readable). Each layer is read under the
    slot's shared lock and checked against the header's seq, as over HTTP."""

    def __init__(self, root: str):
        self.reader = ExportReader(root)

    def wait_header(self, tag: str, deadline: Deadline) -> ExportHeader:
        t0 = time.monotonic()
        while True:
            try:
                return _check_header(self.reader.lookup(tag), tag)
            except ExportFailed as exc:
                raise HandoffError(str(exc)) from exc
            except ExportNotFound:
                time.sleep(min(header_poll_sleep(time.monotonic() - t0), deadline.left()))

    def stream_state(self, tag: str, deadline: Deadline) -> StreamState | None:
        deadline.left()
        try:
            st = self.reader.stream_state(tag)
        except ExportFailed as exc:
            raise HandoffError(str(exc)) from exc
        except ExportNotFound:
            return None
        _check_header(st.header, tag)
        return st

    def layer_blobs(self, header: ExportHeader, layer: LayerRef, deadline: Deadline, *, live: bool = False) -> list[Buffer]:
        deadline.left()
        try:
            with (
                self.reader.open_stream(header.tag, header.seq) if live else self.reader.open(header.tag, seq=header.seq) as ex
            ):
                if live and ex.header.layer(layer.index) != layer:
                    raise HandoffError(f"export {header.tag}: layer {layer.index} changed while streaming")
                return [ex.read_range(layer.index, b.offset, b.nbytes) for b in layer.blobs]
        except (ExportNotFound, ExportFailed, KeyError) as exc:
            raise HandoffError(f"export {header.tag} is gone at layer {layer.index}: {exc}") from exc

    def release(self, header: ExportHeader) -> None:
        self.reader.release(header.tag, header.seq)  # False without write access; hold_seconds frees it then


# ----------------------------------------------------------------------------------------------- vLLM
class VllmClient:
    """POST /v1/completions with max_tokens 1 and kv_transfer_params naming the export tag."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers.update(bearer_header(api_key))

    def prefill(self, tokens: list[int], kv_transfer_params: dict[str, object], deadline: Deadline) -> float:
        body = json.dumps(
            {"model": self.model, "prompt": tokens, "max_tokens": 1, "temperature": 0, "kv_transfer_params": kv_transfer_params}
        ).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(self.url, data=body, headers=self._headers)
        try:
            with _OPENER.open(req, timeout=deadline.left()) as r:
                r.read()
        except urllib.error.HTTPError as e:
            raise HandoffError(f"vLLM answered {e.code} to the prefill request") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            deadline.left()
            raise HandoffError(f"vLLM prefill request failed: {e}") from e
        return time.perf_counter() - t0
