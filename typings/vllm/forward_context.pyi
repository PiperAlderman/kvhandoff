from typing import Any

class ForwardContext:
    no_compile_layers: dict[str, Any]

def get_forward_context() -> ForwardContext: ...
