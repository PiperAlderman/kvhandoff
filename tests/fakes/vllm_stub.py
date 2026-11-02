"""Just enough of vLLM for the connectors to import and run on CPU (the attributes KVConnectorBase_V1.__init__ sets on
vLLM main base.py:214-232), plus a KVConnectorFactory with the real register_connector semantics."""

from __future__ import annotations

import enum
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, ClassVar

STUB_MODULES = (
    "vllm",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.distributed.kv_transfer.kv_connector.v1.base",
    "vllm.distributed.kv_transfer.kv_connector.factory",
    "vllm.forward_context",
)


class TP:
    """A fake tensor-parallel group; all_gather concatenates shards registered by the test."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.gather_fn: Any = None

    def all_gather(self, x: Any, dim: int = -1) -> Any:
        assert self.gather_fn is not None
        return self.gather_fn(x, dim)


STATE: dict[str, Any] = {"tp": None, "tp_rank": 0, "forward": None}


def _build() -> dict[str, types.ModuleType]:
    mods = {name: types.ModuleType(name) for name in STUB_MODULES}
    mods["vllm"].__version__ = "0.30.0"  # type: ignore[attr-defined]
    base = mods["vllm.distributed.kv_transfer.kv_connector.v1.base"]

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class SupportsHMA(ABC):
        @abstractmethod
        def request_finished_all_groups(self, request: Any, block_ids: Any) -> Any: ...

    class KVConnectorBase_V1(ABC):  # noqa: N801
        """The abstract methods of KVConnectorBase_V1 at vLLM v0.30.0 (checked against the tag's base.py) and main."""

        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any):
            self._connector_metadata = None
            self._vllm_config = vllm_config
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._kv_cache_config = kv_cache_config
            self._role = role

        def _get_connector_metadata(self) -> Any:
            return self._connector_metadata

        @abstractmethod
        def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None: ...
        @abstractmethod
        def wait_for_layer_load(self, layer_name: str) -> None: ...
        @abstractmethod
        def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None: ...
        @abstractmethod
        def wait_for_save(self) -> None: ...
        @abstractmethod
        def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> Any: ...
        @abstractmethod
        def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None: ...
        @abstractmethod
        def build_connector_meta(self, scheduler_output: Any) -> Any: ...

    base.KVConnectorRole = KVConnectorRole  # type: ignore[attr-defined]
    base.KVConnectorMetadata = KVConnectorMetadata  # type: ignore[attr-defined]
    base.SupportsHMA = SupportsHMA  # type: ignore[attr-defined]
    base.KVConnectorBase_V1 = KVConnectorBase_V1  # type: ignore[attr-defined]

    factory = mods["vllm.distributed.kv_transfer.kv_connector.factory"]

    class KVConnectorFactory:
        _registry: ClassVar[dict[str, Any]] = {}

        @classmethod
        def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
            if name in cls._registry:
                raise ValueError(f"Connector '{name}' is already registered.")
            cls._registry[name] = (module_path, class_name)

    factory.KVConnectorFactory = KVConnectorFactory  # type: ignore[attr-defined]

    dist = mods["vllm.distributed"]
    dist.get_tp_group = lambda: STATE["tp"] or TP()  # type: ignore[attr-defined]
    dist.get_tensor_model_parallel_rank = lambda: STATE["tp_rank"]  # type: ignore[attr-defined]
    fwd = mods["vllm.forward_context"]

    def get_forward_context() -> Any:
        if STATE["forward"] is None:
            raise RuntimeError("no forward context")
        return types.SimpleNamespace(no_compile_layers=STATE["forward"])

    fwd.get_forward_context = get_forward_context  # type: ignore[attr-defined]
    return mods


@contextmanager
def installed() -> Generator[dict[str, types.ModuleType]]:
    """Install the stub vLLM for the duration; connector modules imported inside are dropped afterwards."""
    saved = {k: v for k, v in sys.modules.items() if k == "vllm" or k.startswith("vllm.") or k.startswith("kvhandoff.vllm.")}
    for k in saved:
        del sys.modules[k]
    mods = _build()
    sys.modules.update(mods)
    STATE.update(tp=None, tp_rank=0, forward=None)
    try:
        yield mods
    finally:
        for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.") or k.startswith("kvhandoff.vllm.")]:
            del sys.modules[k]
        sys.modules.update(saved)
