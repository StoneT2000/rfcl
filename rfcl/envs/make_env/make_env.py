import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Union

import gymnasium
import numpy as np
from chex import Array
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import TimeLimit
from omegaconf import OmegaConf

from rfcl.envs.make_env.is_env import is_gymnasium_robotics_env, is_dmc_env, is_meta_world_env
import rfcl.envs.make_env._mani_skill2 as _mani_skill2
from rfcl.envs.wrappers.common import (
    ContinuousTaskWrapper,
    EpisodeStatsWrapper,
    SparseRewardWrapper,
)


THIS_FILE = "rfcl/envs/make_env/make_env.py"


@dataclass
class EnvConfig:
    env_id: str
    jax_env: bool
    max_episode_steps: int
    num_envs: int
    env_kwargs: Dict
    action_scale: Union[Optional[np.ndarray], Optional[List[float]]]


@dataclass
class EnvMeta:
    sample_obs: Array
    sample_acts: Array
    obs_space: spaces.Space
    act_space: spaces.Space
    env_suite: str


def wrap_mujoco_env(env, idx=0, record_video_path=None, wrappers=[], record_episode_kwargs=dict()):
    from rfcl.envs.wrappers._adroit import RecordEpisodeWrapper

    for wrapper in wrappers:
        env = wrapper(env)
    if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
        env = RecordEpisodeWrapper(
            env,
            record_video_path,
            trajectory_name=f"trajectory_{idx}",
            save_video=record_episode_kwargs["save_video"],
            save_trajectory=record_episode_kwargs["save_trajectory"],
            info_on_video=record_episode_kwargs["info_on_video"],
        )
    return env


def make_env_from_cfg(cfg: EnvConfig, seed: int = None, video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    if not isinstance(cfg.env_kwargs, dict):
        cfg.env_kwargs = OmegaConf.to_container(cfg.env_kwargs)
    return make_env(
        env_id=cfg.env_id,
        jax_env=cfg.jax_env,
        max_episode_steps=cfg.max_episode_steps,
        num_envs=cfg.num_envs,
        seed=seed,
        record_video_path=video_path,
        env_kwargs=cfg.env_kwargs,
        action_scale=cfg.action_scale,
        wrappers=wrappers,
        record_episode_kwargs=record_episode_kwargs,
    )


def make_env(
    env_id: str,
    jax_env: bool,
    max_episode_steps: int,
    num_envs: Optional[int] = 1,
    seed: Optional[int] = 0,
    record_video_path: str = None,
    env_kwargs=dict(),
    action_scale: np.ndarray = None,
    wrappers=[],
    record_episode_kwargs=dict(),
):
    """
    Utility function to create a jax/non-jax based environment given an env_id
    """
    default_record_episode_kwargs = dict(save_video=True, save_trajectory=False, record_single=True, info_on_video=True)
    record_episode_kwargs = {**default_record_episode_kwargs, **record_episode_kwargs}
    if jax_env:
        raise NotImplementedError()
    else:
        context = "fork"
        env_action_scale = 1
        if action_scale is not None:
            action_scale = np.array(action_scale)
            env_action_scale = action_scale
        rescale_action_wrapper = lambda x: gymnasium.wrappers.RescaleAction(x, -env_action_scale, env_action_scale)
        clip_wrapper = lambda x: gymnasium.wrappers.ClipAction(x)
        wrappers = [ContinuousTaskWrapper, SparseRewardWrapper, EpisodeStatsWrapper, rescale_action_wrapper, clip_wrapper, *wrappers]

        if _mani_skill2.is_mani_skill2_env(env_id):
            env_factory = _mani_skill2.env_factory

            context = "forkserver"  # currently ms2 does not work with fork
        elif is_gymnasium_robotics_env(env_id):
            from rfcl.envs.maze.test_maze import PointMazeTestEnv

            def env_factory(env_id, idx, record_video_path, env_kwargs, wrappers=[], record_episode_kwargs=dict()):
                def _init():
                    env = gymnasium.make(env_id, disable_env_checker=True, **env_kwargs)
                    return wrap_mujoco_env(env, idx=idx, record_video_path=record_video_path, wrappers=wrappers, record_episode_kwargs=record_episode_kwargs)

                return _init

        elif is_meta_world_env(env_id):

            def env_factory(env_id, idx, record_video_path, env_kwargs, wrappers=[], record_episode_kwargs=dict()):
                def _init():
                    from gymnasium.envs.registration import EnvSpec
                    from metaworld.envs import (
                        ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
                        ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                    )

                    from rfcl.envs.wrappers._meta_world import MetaWorldEnv

                    env = MetaWorldEnv(env_id, **env_kwargs)
                    env.spec = EnvSpec(id=env_id, max_episode_steps=max_episode_steps)
                    return wrap_mujoco_env(env, idx=idx, record_video_path=record_video_path, wrappers=wrappers, record_episode_kwargs=record_episode_kwargs)

                return _init

        elif is_dmc_env(env_id):
            def env_factory(env_id, idx, record_video_path, env_kwargs, wrappers=[], record_episode_kwargs=dict()):
                def _init():
                    from rfcl.envs.wrappers._dmc import DMCGymEnv
                    env = DMCGymEnv(env_id, **env_kwargs)
                    return wrap_mujoco_env(env, idx=idx, record_video_path=record_video_path, wrappers=wrappers, record_episode_kwargs=record_episode_kwargs)

                return _init
        else:
            raise NotImplementedError()

        wrappers = [
            (lambda x: TimeLimit(x, max_episode_steps=max_episode_steps)),
            *wrappers,
        ]
        # create a vector env parallelized across CPUs with the given timelimit and auto-reset
        vector_env_cls = partial(AsyncVectorEnv, context=context)
        if num_envs == 1:
            vector_env_cls = SyncVectorEnv
        env: VectorEnv = vector_env_cls(
            [
                env_factory(
                    env_id,
                    idx,
                    env_kwargs=env_kwargs,
                    record_video_path=record_video_path,
                    wrappers=wrappers,
                    record_episode_kwargs=record_episode_kwargs,
                )
                for idx in range(num_envs)
            ]
        )
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        env.reset(seed=seed)
        sample_obs = obs_space.sample()
        sample_acts = act_space.sample()

    return env, EnvMeta(
        obs_space=obs_space,
        act_space=act_space,
        sample_obs=sample_obs,
        sample_acts=sample_acts,
        env_suite=get_env_suite(env_id),
    )


def get_env_suite(env_id):
    """
    given env_id return the name of the suite the env is from
    """
    if _mani_skill2.is_mani_skill2_env(env_id):
        return "mani_skill2"
    elif "PointMazeTest" in env_id.split("-") or is_gymnasium_robotics_env(env_id):
        return "gymnasium_robotics"
    elif is_meta_world_env(env_id):
        return "meta_world"
    elif is_dmc_env(env_id):
        return "dm_control"
    else:
        warnings.warn(
            f"Unknown environment suite for env {env_id}. You can safely ignore this. If this is a new environment we recommend updating the get_env_suite function in {THIS_FILE} file with the right details."
        )
        return "unknown"


def get_initial_state_wrapper(env_id):
    """
    given env_id return the InitialStateWrapper that is compatible with that env, allowing resetting to various initial states instead of the
    randomized initial state created with env.reset
    """
    if _mani_skill2.is_mani_skill2_env(env_id):
        from rfcl.envs.wrappers._maniskill2 import ManiSkill2InitialStateWrapper

        return ManiSkill2InitialStateWrapper
    elif is_gymnasium_robotics_env(env_id):
        from rfcl.envs.wrappers._adroit import AdroitInitialStateWrapper

        return AdroitInitialStateWrapper
    elif is_meta_world_env(env_id):
        from rfcl.envs.wrappers._meta_world import MetaWorldInitialStateWrapper

        return MetaWorldInitialStateWrapper
    elif is_dmc_env(env_id):
        from rfcl.envs.wrappers._dmc import DMCInitialStateWrapper
        return DMCInitialStateWrapper
    else:
        raise NotImplementedError(
            f"Need to add the initial state wrapper for {env_id}. Add it to rfcl/envs/wrappers/_<env_suite_name>.py and import it and return it here"
        )
