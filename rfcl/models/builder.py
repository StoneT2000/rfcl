"""
Build various base models with configurations
"""
from dataclasses import asdict
from typing import Callable

import flax.linen as nn
from dacite import from_dict

from .mlp import MLP, MLPConfig
from .types import NetworkConfig

ACTIVATIONS = dict(relu=nn.relu, gelu=nn.gelu, tanh=nn.tanh, sigmoid=nn.sigmoid, log_softmax=nn.log_softmax)


def activation_to_fn(activation: str) -> Callable:
    if activation is None:
        return None
    if activation in ACTIVATIONS:
        return ACTIVATIONS[activation]
    else:
        raise ValueError(f"{activation} is not handled as an activation. Handled activations are {list(ACTIVATIONS.keys())}")


def build_network_from_cfg(cfg: NetworkConfig):
    if cfg.type == "mlp":
        cfg = from_dict(data_class=MLPConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation)
        cfg.arch_cfg.output_activation = activation_to_fn(cfg.arch_cfg.output_activation)
        return MLP(**asdict(cfg.arch_cfg))
