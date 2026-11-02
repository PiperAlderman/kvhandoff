"""vLLM KV-connector plugin: KVHandoffPagedConnector (Llama-style) and KVHandoffDSV4Connector (DeepSeek V4).

Both are kv_producer connectors. Load either with kv_connector_module_path (no plugin needed)::

    --kv-transfer-config '{"kv_connector":"KVHandoffPagedConnector",
                           "kv_connector_module_path":"kvhandoff.vllm.paged_connector",
                           "kv_role":"kv_producer",
                           "kv_connector_extra_config":{"store_path":"<store>"}}'

or by name alone once the package is installed (the ``vllm.general_plugins`` entry point registers both names).
"""

from __future__ import annotations

from typing import Any, Final

# vLLM builds these connectors were run against on hardware. Others log a warning at start (a hardware validation run adds
# to this list).
TESTED_VLLM: Final[dict[str, tuple[str, ...]]] = {
    "KVHandoffPagedConnector": ("0.30.0",),
    "KVHandoffDSV4Connector": ("0.23.1rc1.dev1053+gf2317c227",),
}


def vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # noqa: BLE001  (version lookup must never break the connector)
        return "unknown"


def served_name(model_config: Any) -> str:
    """vLLM's served model name (first one), else the model path or id."""
    name = getattr(model_config, "served_model_name", None)
    if isinstance(name, list | tuple) and name:
        return str(name[0])  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(name, str) and name:
        return name
    return str(getattr(model_config, "model", "unknown"))
