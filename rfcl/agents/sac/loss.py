"""
Update / Loss Functions for SAC
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax import struct

from rfcl.agents.sac.config import TimeStep
from rfcl.agents.sac.networks import ActorCritic
from rfcl.models import Model


@struct.dataclass
class CriticUpdateAux:
    critic_loss: Array = None
    q: Array = None

@struct.dataclass
class TempUpdateAux:
    temp_loss: Array = None
    temp: Array = None

@struct.dataclass
class ActorUpdateAux:
    actor_loss: Array = None
    entropy: Array = None

@struct.dataclass
class UpdateMetrics:
    actor: ActorUpdateAux
    critic: CriticUpdateAux
    temp: TempUpdateAux

def update_critic(
    key: PRNGKey,
    ac: ActorCritic,
    batch: TimeStep,
    discount: float,
    backup_entropy: bool,
    num_min_qs: int,
    num_qs: int
) -> Tuple[Model, CriticUpdateAux]:
    dist = ac.actor(batch.next_env_obs)
    # next_actions, next_log_probs = dist.sample_and_log_prob(seed=key) # doing it together is unstable, we opt for the stable alternative below
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
        if num_sample is not None:
            all_indx = jnp.arange(0, num_qs)
            indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

            if "Ensemble_0" in params:
                ens_params = jax.tree_util.tree_map(
                    lambda param: param[indx], params["Ensemble_0"]
                )
                params = params.copy(add_or_replace={"Ensemble_0": ens_params})
            else:
                params = jax.tree_util.tree_map(lambda param: param[indx], params)
        return params
    target_params = subsample_ensemble(
        key, ac.target_critic.params, num_min_qs, num_qs
    )
    next_qs = ac.target_critic.apply_fn(target_params, batch.next_env_obs, next_actions)

    next_q = next_qs.min(axis=0)
    target_q = batch.reward + discount * batch.mask * next_q

    if backup_entropy:
        target_q -= discount * batch.mask * ac.temp() * next_log_probs

    def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Any]:
        qs = ac.critic.apply_fn(critic_params, batch.env_obs, batch.action)
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, CriticUpdateAux(critic_loss=critic_loss, q=qs.mean())

    grad_fn = jax.grad(critic_loss_fn, has_aux=True)
    grads, aux = grad_fn(ac.critic.params)
    new_critic = ac.critic.apply_gradients(grads=grads)

    return new_critic, aux

def update_actor(key: PRNGKey, ac: ActorCritic, batch: TimeStep) -> Tuple[Model, ActorUpdateAux]:
    def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Any]:
        dist = ac.actor.apply_fn(actor_params, batch.env_obs)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        qs = ac.critic(batch.env_obs, actions)
        q = qs.mean(axis=0)
        actor_loss = (log_probs * ac.temp() - q).mean()
        return actor_loss, ActorUpdateAux(actor_loss=actor_loss, entropy=-log_probs.mean())

    grad_fn = jax.grad(actor_loss_fn, has_aux=True)
    grads, aux = grad_fn(ac.actor.params)
    new_actor = ac.actor.apply_gradients(grads=grads)
    return new_actor, aux


def update_temp(temp: Model, entropy: float, target_entropy: float) -> Tuple[Model, TempUpdateAux]:
    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn(temp_params)
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, TempUpdateAux(temp=temperature, temp_loss=temp_loss)

    grad_fn = jax.grad(temperature_loss_fn, has_aux=True)
    grads, aux = grad_fn(temp.params)
    new_temp = temp.apply_gradients(grads=grads)
    return new_temp, aux


def update_target(critic: Model, target_critic: Model, tau: float) -> Model:
    """
    update target_critic with polyak averaging
    """
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)