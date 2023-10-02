"""Categorical class"""
import chex
import distrax
import flax.linen as nn


Array = chex.Array
Scalar = chex.Scalar


class Categorical(nn.Module):
    """
    Categorical exploration module, returning a distribution over given logits
    """

    categorical = True

    def __call__(self, a) -> distrax.Distribution:
        dist = distrax.Categorical(logits=a)
        return dist

    def _log_prob_from_distribution(self, dist: distrax.Distribution, a: Array) -> Array:
        return dist.log_prob(a)
