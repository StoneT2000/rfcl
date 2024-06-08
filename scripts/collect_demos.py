import os
import os.path as osp
import sys
import warnings
from typing import Any, Optional

import flax
import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from rfcl.agents.sac import SAC, ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.envs.make_env import EnvConfig, make_env_from_cfg
from rfcl.logger import LoggerConfig
from rfcl.models import NetworkConfig, build_network_from_cfg
from rfcl.utils.io_utils import merge_h5
from rfcl.utils.parse import parse_cfg
from rfcl.utils.spaces import get_action_dim


warnings.simplefilter(action="ignore", category=FutureWarning)
from dataclasses import dataclass

from rfcl.data.dataset import ReplayDataset, get_states_dataset


@dataclass
class TrainConfig:
    actor_lr: float
    critic_lr: float
    dataset_path: str
    shuffle_demos: bool
    num_demos: int

    data_action_scale: Any


@dataclass
class SACNetworkConfig:
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: Optional[LoggerConfig]
    verbose: int
    algo: str = "sac"
    model_path: Any = None

    num_envs: int = 8
    num_episodes: int = 1000
    out: str = None  # where to save videos and trajectories


from dacite import from_dict


def main(cfg: SACExperiment, model_path: str):
    np.random.seed(cfg.seed)
    # Setup training and evaluation environment configs
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg_data = OmegaConf.to_container(cfg)
    cfg_data["logger"]["cfg"] = dict()
    del cfg_data["logger"]["best_stats_cfg"]
    del cfg_data["logger"]["save_fn"]

    cfg = from_dict(data_class=SACExperiment, data=cfg_data)
    env_cfg = cfg.env
    cfg.env.num_envs = cfg.num_envs
    cfg.sac.num_envs = cfg.env.num_envs
    video_path = cfg.out if cfg.out is not None else osp.join(osp.dirname(model_path), "../", "eval_videos")

    env, env_meta = make_env_from_cfg(
        env_cfg,
        seed=cfg.seed,
        wrappers=[],
        video_path=video_path,
        record_episode_kwargs=dict(save_video=True, save_trajectory=True, record_single=False, info_on_video=False),
    )
    np.save(osp.join(video_path, "action_scale.npy"), env_cfg.action_scale)
    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)

    def create_ac_model():
        actor = DiagGaussianActor(
            feature_extractor=build_network_from_cfg(cfg.network.actor),
            act_dims=act_dims,
            state_dependent_std=True,
        )
        ac = ActorCritic.create(
            jax.random.PRNGKey(cfg.seed),
            actor=actor,
            critic_feature_extractor=build_network_from_cfg(cfg.network.critic),
            sample_obs=sample_obs,
            sample_acts=sample_acts,
            initial_temperature=cfg.sac.initial_temperature,
            actor_optim=optax.adam(learning_rate=cfg.train.actor_lr),
            critic_optim=optax.adam(learning_rate=cfg.train.critic_lr),
        )
        return ac

    # create our algorithm
    ac = create_ac_model()
    algo = SAC(
        env=env,
        eval_env=None,
        env_type=cfg.env.env_type,
        ac=ac,
        logger_cfg=None,  # set none so we don't create a new logger
        cfg=cfg.sac,
    )
    # use a pretrained model
    import pickle

    with open(model_path, "rb") as f:
        state_dict = pickle.load(f)
    previous_ac = flax.serialization.from_bytes(algo.state.ac, state_dict["train_state"].ac)
    ac = ac.load(previous_ac.state_dict())
    algo.state = algo.state.replace(ac=ac)

    print(f"Collecting {cfg.num_episodes} demonstrations using model ckpt at {model_path}, saving to {video_path}")
    print(f"Running {cfg.num_envs} parallel environments")
    rng_key, eval_rng_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)
    results = algo.evaluate(
        eval_rng_key,
        env_cfg.num_envs,
        cfg.env.max_episode_steps * cfg.num_episodes // env_cfg.num_envs,
        algo.loop,
        algo.state.ac.actor,
        algo.state.ac.act,
        progress_bar=True,
    )
    env.close()
    del env
    success_rate = np.mean(results["stats"]["success_at_end"])
    ep_ret = np.mean(results["eval_ep_rets"])
    print("Average Return", ep_ret, "Success Rate", success_rate)
    h5_files = [osp.join(video_path, f"trajectory_{i}.h5") for i in range(cfg.num_envs)]
    merge_h5(osp.join(video_path, f"trajectory.h5"), h5_files)
    for file in h5_files:
        os.remove(file)
        os.remove(file.replace(".h5", ".json"))


if __name__ == "__main__":
    # infer config file from model path
    model_path = sys.argv[1]
    cfg_path = osp.join(osp.dirname(model_path), "../", "config.yml")
    cfg = parse_cfg(default_cfg_path=cfg_path)
    main(cfg, model_path=model_path)
