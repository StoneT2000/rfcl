"""
Models for SAC
"""
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Type

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from chex import Array, PRNGKey
from flax import struct
from tensorflow_probability.substrates import jax as tfp

from rfcl.models import Model
from rfcl.models.model import Params


tfd = tfp.distributions
tfb = tfp.bijectors

class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)

class Critic(nn.Module):
    feature_extractor: nn.Module

    @nn.compact
    def __call__(self, obs: Array, acts: Array) -> Array:
        x = jnp.concatenate([obs, acts], -1)
        features = self.feature_extractor(x)
        value = nn.Dense(1)(features)
        return jnp.squeeze(value, -1)

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class DiagGaussianActor(nn.Module):
    feature_extractor: nn.Module
    act_dims: int
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    tanh_squash_distribution: bool = True

    state_dependent_std: bool = True
    log_std_range: Tuple[float, float] = (-5.0, 2.0)

    def setup(self) -> None:
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
            self.log_std = nn.Dense(self.act_dims, kernel_init=default_init(1))
        else:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.act_dims,))

        # scale of orthgonal initialization is recommended to be (high - low) / 2.
        # We always assume envs use normalized actions [-1, 1] so we init with 1
        self.action_head = nn.Dense(self.act_dims, kernel_init=default_init(1))

    def __call__(self, x, deterministic=False):
        x = self.feature_extractor(x)
        a = self.action_head(x)
        if not self.tanh_squash_distribution:
            a = nn.tanh(a)
        if deterministic:
            return nn.tanh(a)
        if self.state_dependent_std:
            log_std = self.log_std(x)
            log_std = nn.tanh(log_std)
        else:
            log_std = self.log_std
        log_std = self.log_std_range[0] + 0.5 * (self.log_std_range[1] - self.log_std_range[0]) * (log_std + 1)
        dist = tfd.MultivariateNormalDiag(a, jnp.exp(log_std))
        # distrax has some numerical imprecision bug atm where calling sample then log_prob can raise NaNs. tfd is more stable at the moment
        # dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=1))
            dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
        return dist


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self):
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


@struct.dataclass
class ActorCritic:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model

    @classmethod
    def create(
        cls,
        rng_key: PRNGKey,
        sample_obs: Array,
        sample_acts: Array,
        actor: DiagGaussianActor = None,
        critic_feature_extractor: nn.Module = None,
        actor_optim: optax.GradientTransformation = optax.adam(3e-4),
        critic_optim: optax.GradientTransformation = optax.adam(3e-4),
        initial_temperature: float = 1.0,
        temperature_optim: optax.GradientTransformation = optax.adam(3e-4),
        num_qs: int = 10,
        num_min_qs: int = 2
    ) -> "ActorCritic":
        rng_key, actor_rng_key, critic_rng_key, value_rng_key, temp_rng_key = jax.random.split(rng_key, 5)

        assert actor is not None
        actor = Model.create(actor, actor_rng_key, sample_obs, actor_optim)
        assert critic_feature_extractor is not None
        critic_cls = partial(Critic, feature_extractor=critic_feature_extractor)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_model = Model.create(critic_def, critic_rng_key, [sample_obs, sample_acts], critic_optim)
        
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic_model = Model.create(target_critic_def, critic_rng_key, [sample_obs, sample_acts])
        target_critic_model = target_critic_model.replace(params=critic_model.params) # this makes the two models the same and we use a subsampler to subsample the target values

        temp = Model.create(Temperature(initial_temperature), temp_rng_key, tx=temperature_optim)

        return cls(actor=actor, critic=critic_model, target_critic=target_critic_model, temp=temp)

    @partial(jax.jit)
    def act(self, rng_key: PRNGKey, actor: DiagGaussianActor, obs):
        """Sample actions deterministicly"""
        return actor(obs, deterministic=True), {}

    @partial(jax.jit)
    def sample(self, rng_key: PRNGKey, actor: DiagGaussianActor, obs):
        """Sample actions from distribution"""
        return actor(obs).sample(seed=rng_key), {}

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
            temp=self.temp.state_dict(),
        )

    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state_dict()))

    def load(self, params_dict: Params, load_critic=True):
        actor = self.actor.load_state_dict(params_dict["actor"])
        critic = self.critic
        target_critic = self.target_critic
        if load_critic:
            critic = self.critic.load_state_dict(params_dict["critic"])
            target_critic = self.target_critic.load_state_dict(params_dict["target_critic"])
        temp = self.temp.load_state_dict(params_dict["temp"])
        return self.replace(actor=actor, critic=critic, target_critic=target_critic, temp=temp)

    def load_from_path(self, load_path: str, load_critic=True):
        with open(load_path, "rb") as f:
            params_dict = flax.serialization.from_bytes(self.state_dict(), f.read())
        self.load(params_dict, load_critic=load_critic)
        return self
