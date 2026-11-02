"""Optional EXO integration: a prefill-server resolve handler that asks vLLM for the prefill and streams the export in
EXO's disaggregation wire format (KVChunk per layer for paged_kv, ArraysState per layer for DeepSeek V4, exactly the
bytes the exo-explore/exo#2545 receiver expects). EXO's own protocol module does the encoding, imported lazily."""
