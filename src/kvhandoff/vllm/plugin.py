"""``vllm.general_plugins`` entry point: register both connectors by name.

vLLM calls general plugins in every process, possibly more than once; KVConnectorFactory.register_connector raises
on a duplicate name, so this swallows that and is idempotent. Registration is lazy (module path + class name), so
nothing heavy is imported here.
"""

from __future__ import annotations

from typing import Final

CONNECTORS: Final = (
    ("KVHandoffPagedConnector", "kvhandoff.vllm.paged_connector", "KVHandoffPagedConnector"),
    ("KVHandoffDSV4Connector", "kvhandoff.vllm.dsv4_connector", "KVHandoffDSV4Connector"),
)


def register() -> list[str]:
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    except ImportError:
        return []
    done: list[str] = []
    for name, module, cls in CONNECTORS:
        try:
            KVConnectorFactory.register_connector(name, module, cls)
            done.append(name)
        except ValueError:
            pass  # already registered in this process
    return done
