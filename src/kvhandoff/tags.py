"""Export tags: the per-request names that become file names in the store and path segments on the transfer service."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from typing import Any, Final

_TAG: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

# The kv_transfer_params key that asks a connector for an export.
PARAM_TAG: Final = "kvhandoff_tag"


def is_valid_tag(tag: str) -> bool:
    """Only [A-Za-z0-9._-], at most 96 characters, no leading dot or dash (no hidden files, no option look-alikes)."""
    return bool(_TAG.match(tag)) and ".." not in tag


def safe_tag(tag: str, fallback: str) -> str:
    """``tag`` when valid, else a sanitised form of ``fallback`` (the vLLM request id) prefixed ``r-``."""
    if tag and is_valid_tag(tag):
        return tag
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", fallback or "req")[:90].lstrip("._-").replace("..", "_") or "req"
    return "r-" + clean


def new_tag(prefix: str = "h") -> str:
    """A fresh random tag for one hand-off (64 bits of randomness)."""
    return f"{prefix}-{secrets.token_hex(8)}"


def requested_tag(params: Mapping[str, Any] | None) -> str | None:
    """The tag a request asked to be exported under, or None if it did not ask for an export.

    ``{"kvhandoff_tag": "<tag>"}`` asks for an export.
    """
    if not params:
        return None
    value = params.get(PARAM_TAG)
    if value:
        return str(value)
    return None
