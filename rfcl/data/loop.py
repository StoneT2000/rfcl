"""
Environment Loops
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Callable, List, Tuple, TypeVar, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from flax import struct
from tqdm import tqdm

from rfcl.utils import tools


EnvObs = TypeVar("EnvObs")
EnvState = TypeVar("EnvState")
EnvAction = TypeVar("EnvAction")


@struct.dataclass
class DefaultTimeStep:
    env_obs: EnvObs
    action: EnvAction
    reward: Array
    next_env_obs: EnvObs
    ep_ret: float
    ep_len: int
    terminated: bool
    truncated: bool


@struct.dataclass
class EnvLoopState:
    env_state: EnvState = None
    env_obs: EnvObs = None
    ep_ret: float = None
    ep_len: int = None
    # TODO add info?


class BaseEnvLoop(ABC):
    rollout_callback: Callable
    num_envs: int

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs

    @abstractmethod
    def rollout(
        self, rng_keys: List[PRNGKey], loop_state: Union[EnvLoopState, None], params: Any, apply_fn: Callable, steps_per_env: int, progress_bar: bool
    ) -> Tuple[DefaultTimeStep, EnvLoopState]:
        raise NotImplementedError("Rollout not defined")

    @abstractmethod
    def reset_loop(self, rng_key: PRNGKey) -> EnvLoopState:
        """
        reset the looper and give a new obsveration and states (if available). This is simply a wrapper that auto vmaps the reset function, preventing
        not jitted functions need to vmap the reset themselves or generate many RNG keys which can be slow.
        """
        raise NotImplementedError("reset loop not defined")

import jax
import jax.dlpack
import torch
import torch.utils.dlpack
def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))
def jax_to_torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
class GymLoop(BaseEnvLoop):
    """
    RL loop for non jittable environments environment

    Args :
        rollout_callback: Callable
            A rollout_callback function that is called after each env step. The output of this function is appended into a rollout buffer that is returned at the end of the
            rollout function
    """

    def __init__(self, env: gym.Env, num_envs: int = 1, rollout_callback: Callable = None, env_type: str = "gym:cpu") -> None:
        self.env = env
        self.num_envs = num_envs
        self.rollout_callback = rollout_callback
        self.env_type = env_type
        self.is_torch_gpu_env = env_type == "gym:gpu"
        super().__init__(num_envs=num_envs)

    def reset_loop(self, rng_key: PRNGKey):
        obs, _ = self.env.reset()
        return EnvLoopState(
            env_state=None,
            env_obs=obs,
            ep_ret=np.zeros((self.num_envs,), dtype=float),
            ep_len=np.zeros((self.num_envs,), dtype=int),
        )

    def rollout(
        self,
        rng_keys: List[PRNGKey],
        loop_state: Union[EnvLoopState, None],
        params: Any,
        apply_fn: Callable,
        steps_per_env: int,
        progress_bar: bool = False,
    ):
        """
        Rollout across N parallelized non-jitted, non-state parameterized, environments with an actor function apply_fn and
        return the rollout buffer as well the next loop state. The rollout buffer can be customized using the rollout_callback function
        and will also always contain a list of all the final infos.

        Args :
            rng_keys : initial PRNGKeys to use for any randomness. Will only use the last one in list

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps : number of steps to rollout

            loop_state : Initial environment observations, states, returns, and lengths to step forward from. If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state. If not None, it will not call env.reset first.
        """
        rng_key = rng_keys[-1]
        if loop_state is None:
            observations, _ = self.env.reset()
            ep_returns, ep_lengths = (
                np.zeros(self.num_envs, dtype=float),
                np.zeros(self.num_envs, dtype=int),
            )
        else:
            observations = loop_state.env_obs
            ep_returns = loop_state.ep_ret
            ep_lengths = loop_state.ep_len

        data = defaultdict(list)
        p_bar = range(steps_per_env)
        if progress_bar:
            p_bar = tqdm(range(steps_per_env))
        for t in p_bar:
            rng_key, rng_fn_key = jax.random.split(rng_key)

            if self.is_torch_gpu_env:
                observations = torch_to_jax(observations)
            actions, aux = apply_fn(rng_fn_key, params, observations)
            if self.is_torch_gpu_env:
                actions = jax_to_torch(actions)
            else:
                actions = tools.any_to_np(actions)
            (
                next_observations,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(actions)
            ep_lengths = ep_lengths + 1
            if self.is_torch_gpu_env:
                ep_returns = ep_returns + rewards.cpu().numpy()
            else:
                ep_returns = ep_returns + tools.any_to_np(rewards)
            # determine true next observations s_{t+1} if some episodes truncated and not s_0 for terminated or truncated episodes
            true_next_observations = next_observations
            if tools.is_jax_arr(true_next_observations):
                # for envs returning jax arrays, we expect consistency and a final_observation/next_obs to always be returned
                true_next_observations = infos["final_observation"]
            else:
                
                if self.is_torch_gpu_env:
                    pass
                else:
                    if "final_observation" in infos:  # TODO with wrapper to replace with next_observation key
                        true_next_observations = next_observations.copy()
                        for idx, (terminated, truncated) in enumerate(zip(terminations, truncations)):
                            final_obs = infos["final_observation"][idx]
                            if final_obs is not None:
                                true_next_observations[idx] = final_obs

            if self.rollout_callback is not None:
                rb = self.rollout_callback(
                    action=actions,  # a_{t}
                    env_obs=observations,  # s_{t}
                    reward=rewards,  # r_{t}
                    ep_ret=ep_returns.copy(),  # sum_{i=0}^t r_{i}
                    ep_len=ep_lengths.copy(),  # t
                    # s_{t+1} and not s_0 of new episode
                    next_env_obs=true_next_observations,
                    terminated=terminations,
                    truncated=truncations,
                    info=infos,
                    aux=aux,
                )
            else:
                if self.is_torch_gpu_env:
                    rb = dict(
                        env_obs=np.array(observations),
                        next_env_obs=true_next_observations.cpu().numpy(),
                        action=actions.cpu().numpy(),
                        reward=rewards.cpu().numpy(),
                        ep_ret=ep_returns.copy(),
                        ep_len=ep_lengths.copy(),
                        terminated=terminations.cpu().numpy(),
                        truncated=truncations.cpu().numpy(),
                    )
                else:
                    rb = dict(
                        env_obs=observations,
                        next_env_obs=true_next_observations,
                        action=actions,
                        reward=rewards,
                        ep_ret=ep_returns.copy(),
                        ep_len=ep_lengths.copy(),
                        terminated=terminations,
                        truncated=truncations,
                    )
            
            if "final_info" in infos:
                if self.is_torch_gpu_env:
                    data["final_info"] = tools.to_numpy(infos["final_info"])
                else:
                    for info, keep in zip(infos["final_info"], infos["_final_info"]):
                        if keep:
                            data["final_info"].append(info)
                        else:
                            data["final_info"].append(None)  # TEMP TODO
            for k, v in rb.items():
                data[k].append(v)
            observations = next_observations

            if self.is_torch_gpu_env:
                dones = (terminations | truncations).cpu().numpy()
            else:
                dones = (terminations | truncations)
            ep_returns[dones] = 0
            ep_lengths[dones] = 0
        # stack data
        for k in data:
            data[k] = np.stack(data[k])
            if k != "final_info":
                data[k] = data[k].reshape(self.num_envs, steps_per_env, -1)  # (num_envs, steps, D)

        loop_state = EnvLoopState(env_obs=next_observations, env_state=None, ep_ret=ep_returns, ep_len=ep_lengths)
        return data, loop_state


class JaxLoop(BaseEnvLoop):
    """
    Env loop for jax based environments


    Args :
        env_reset : An environment reset function thta takes a PRNGKey and
            returns the initial environment observation and state

        env_step : An environment step function that takes a PRNGKey, state, and action and
            returns the next environment observation, state, reward, done, and info

        reset_env : Whether this env loop class will reset environments when done = True.
            If False, when done = True, the loop only resets recorded episode returns and length.
            The environment itself should have some auto reset functionality.

        rollout_callback : A callback function that takes action, env_obs, reward,
            ep_ret, ep_len, next_env_obs, done, info, aux as input
            note that aux is the auxilliary output of the rolled out policy apply_fn when calling rollout().

            The output of this function is used to create the rollout/replay buffer.

            If this function is set to None, the default generated buffer will contain
            env_obs, action, reward, ep_ret, ep_len, done
    """

    def __init__(
        self,
        env_reset: Callable[[PRNGKey], Tuple[EnvObs, EnvState]],
        env_step: Callable[
            [PRNGKey, EnvState, EnvAction],
            Tuple[EnvObs, EnvState, float, bool, Any],
        ],
        num_envs: int = 1,
        rollout_callback: Callable = None,
    ) -> None:
        self.env_reset = env_reset
        self.env_step = env_step
        self.rollout_callback = rollout_callback

        def default_rollout_cb(action, env_obs, reward, ep_ret, ep_len, next_env_obs, terminated, truncated, info, aux):
            return DefaultTimeStep(
                env_obs=env_obs,
                action=action,
                reward=reward,
                ep_ret=ep_ret,
                ep_len=ep_len,
                next_env_obs=next_env_obs,
                terminated=terminated,
                truncated=truncated,
            )

        if self.rollout_callback is None:
            self.rollout_callback = default_rollout_cb
        self.num_envs = num_envs
        super().__init__(num_envs=num_envs)

    @partial(jax.jit, static_argnames=["self"])
    def reset_loop(self, rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, self.num_envs + 1)
        rng_key = rng_keys[0]
        obs, states, _ = jax.jit(jax.vmap(self.env_reset))(rng_keys[1:])
        return EnvLoopState(
            env_obs=obs,
            env_state=states,
            ep_ret=jnp.zeros((self.num_envs,), dtype=float),
            ep_len=jnp.zeros((self.num_envs,), dtype=int),
        )

    @partial(jax.jit, static_argnames=["self", "steps", "apply_fn"])
    def _rollout_single_env(
        self,
        rng_key: PRNGKey,
        loop_state: Union[EnvLoopState, None],
        params: Any,
        apply_fn: Callable[[PRNGKey, Any, EnvObs], Tuple[EnvAction, Any]],
        steps: int,
    ) -> Tuple[Any, EnvLoopState]:
        """
        Rollout on a single env and return the rollout_callback outputs and the last environment observation and state. Expects the env to have auto-reset turned on

        Args :
            rng_key : initial PRNGKey to use for any randomness

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps : number of steps to rollout

            init_env_states : Initial environment observation and state to step forward from. If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state

        """
        if loop_state is None:
            # if no loop state given, reset this singular env
            rng_key, reset_rng_key = jax.random.split(rng_key)
            env_obs, env_state, _ = self.env_reset(reset_rng_key)
            ep_ret, ep_len = jnp.zeros((1,), dtype=float), jnp.zeros((1,), dtype=int)
            loop_state = EnvLoopState(env_state=env_state, env_obs=env_obs, ep_ret=ep_ret, ep_len=ep_len)

        def step_fn(data: Tuple[PRNGKey, EnvLoopState], _):
            rng_key, loop_state = data
            rng_key, rng_step, rng_fn = jax.random.split(rng_key, 3)
            action, aux = apply_fn(rng_fn, params, loop_state.env_obs)
            (
                next_env_obs,
                next_env_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_step(rng_step, loop_state.env_state, action)
            done = terminated | truncated
            next_ep_ret, next_ep_len = loop_state.ep_ret + reward, loop_state.ep_len + 1
            true_next_env_obs = next_env_obs
            if "final_observation" in info:
                # In Jax final_observation should always exist so just use it
                true_next_env_obs = info["final_observation"]
            rb = self.rollout_callback(
                action=action,
                env_obs=loop_state.env_obs,
                reward=reward,
                ep_ret=next_ep_ret,
                ep_len=next_ep_len,
                next_env_obs=true_next_env_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
                aux=aux,
            )
            next_ep_ret = next_ep_ret * (~done)
            next_ep_len = next_ep_len * (~done)

            return (
                rng_key,
                EnvLoopState(env_obs=next_env_obs, env_state=next_env_state, ep_ret=next_ep_ret, ep_len=next_ep_len),
            ), rb

        step_init = (rng_key, loop_state)
        (_, loop_state), rollout_data = jax.lax.scan(step_fn, step_init, (), steps)
        # rollout_data - (steps, D)
        # loop_state - (D)
        # aux = EnvLoopState(
        #     env_obs=final_env_obs,
        #     env_state=final_env_state,
        #     ep_ret=final_ep_returns,
        #     ep_len=final_ep_lengths,
        # )
        # add batch dimension so it plays nice with vmap in the rollout function
        # aux = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), aux)

        return rollout_data, loop_state

    @partial(
        jax.jit,
        static_argnames=["self", "steps_per_env", "apply_fn"],
    )
    def rollout(
        self,
        rng_keys: List[PRNGKey],
        loop_state: Union[EnvLoopState, None],
        params: Any,
        apply_fn: Callable[[PRNGKey, Any, EnvObs], Tuple[EnvAction, Any]],
        steps_per_env: int,
        progress_bar: bool = False,  # unused
    ) -> Tuple[Any, EnvLoopState]:
        """
        Rollout across N parallelized environments with an actor function apply_fn and
        return the rollout buffer as well as final environment observations and states

        Args :
            rng_keys : initial PRNGKeys to use for any randomness. len(rng_keys) is
            the number of parallel environments that will be run

            params : any function parameters passed to apply_fn

            apply_fn : a function that takes as input a PRNGKey, params, and an environment observation
                and returns a tuple with the action and any auxilliary data

            steps_per_env : number of steps to rollout per parallel environment

            max_episode_length : max number of steps before we truncate the current episode. If -1, we will not truncate any environments

            init_env_states : Initial environment observation and state to step forward from.
                If None, this calls the given self.env_reset function
                to obtain the initial environment observation and state


        This rollout style vmaps rollouts on each parallel env. Once an episode is done, the next immediately starts

        Note: This is faster than only jitting an episode rollout and using a valid mask to remove
        time steps in rollouts that occur after the environment is done.
        The speed increase is noticeable for low number of parallel envs

        The downside here is that the first run will always take extra time but this is
        generally quite minimal overhead over the long term.

        """
        batch_rollout = jax.vmap(self._rollout_single_env, in_axes=(0, 0, None, None, None))
        rollout_data, loop_state = batch_rollout(
            jnp.stack(rng_keys),
            loop_state,
            params,
            apply_fn,
            steps_per_env,
        )
        # rollout_data - (num_envs, steps, D)
        # loop_state - (num_envs, D)
        return rollout_data, loop_state
