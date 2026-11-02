"""Shared fixtures. EXO-dependent tests need EXO's protocol module: set EXO_SRC to an EXO checkout's ``src`` directory
(e.g. the upstream-prs clone); only protocol.py is loaded (by file), so EXO's dependencies are not needed, only msgspec."""

from __future__ import annotations

import importlib.util
import os
import socket
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def write_token(path: Path, token: str = "t" * 48, mode: int = 0o600) -> Path:
    path.write_text(token + "\n")
    path.chmod(mode)
    return path


def load_exo_protocol() -> Any:
    """EXO's protocol module registered as exo.worker.disaggregated.protocol, or None."""
    name = "exo.worker.disaggregated.protocol"
    if name in sys.modules:
        return sys.modules[name]
    src = os.environ.get("EXO_SRC")
    if not src:
        return None
    path = Path(src) / "exo" / "worker" / "disaggregated" / "protocol.py"
    if not path.exists() or importlib.util.find_spec("msgspec") is None:
        return None
    for pkg in ("exo", "exo.worker", "exo.worker.disaggregated"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # type: ignore[attr-defined]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def exo_protocol() -> Any:
    mod = load_exo_protocol()
    if mod is None:
        pytest.skip("EXO protocol not available (set EXO_SRC to an EXO checkout's src directory; needs msgspec)")
    return mod


@pytest.fixture
def store_root(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "store"
    yield str(root)
