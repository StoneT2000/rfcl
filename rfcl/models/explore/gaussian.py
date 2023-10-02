"""Gaussian class"""
import chex
import distrax
import flax.linen as nn
import jax.numpy as jnp


Array = chex.Array
Scalar = chex.Scalar


class Gaussian(nn.Module):
    """Gaussian exploration module, returning a gaussian distribution with a given mean and scale"""

    categorical = False
    act_dims: int
    log_std_scale: float = -0.5

    def setup(self) -> None:
        self.log_std = self.param(
            "log_std",
            lambda rng, act_dims, log_std_scale: jnp.ones(act_dims) * log_std_scale,
            self.act_dims,
            self.log_std_scale,
        )

    def __call__(self, a) -> distrax.Distribution:
        dist = distrax.MultivariateNormalDiag(a, jnp.exp(self.log_std))
        return dist

    def _log_prob_from_distribution(self, dist: distrax.Distribution, a: Array) -> Array:
        return dist.log_prob(a)
