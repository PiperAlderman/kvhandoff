"""kvhandoff: hand a vLLM prefill to EXO/MLX decode.

Layers (each importable without the ones below it):
  core       interchange header, config, tags, auth, logging, export store, transfer service   (stdlib only)
  producer   torch logic: paged-KV gather and layout normalisation, DeepSeek V4 conversion       (torch, no vLLM)
  vllm       the two KV connectors and the plugin entry point                                     (vLLM)
  exo        the EXO prefill-server hook that streams exports in EXO's wire format               (EXO, numpy)
"""

__version__ = "0.1.0"
