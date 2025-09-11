from typing import Any, Callable

import sax

sax.set_port_naming_strategy("optical")

models: dict[str, Callable[..., Any]] = {}
