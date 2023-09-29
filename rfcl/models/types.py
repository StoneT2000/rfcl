from dataclasses import dataclass
from typing import Any


@dataclass
class NetworkConfig:
    type: str
    arch_cfg: Any
