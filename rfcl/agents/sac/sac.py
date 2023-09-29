import os
import pickle
import time
from collections import defaultdict
from functools import partial
from typing import Any, Tuple

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from flax import struct
from rfcl.agents.base import BasePolicy
from rfcl.data.buffer import GenericBuffer
from rfcl.data.loop import DefaultTimeStep, EnvLoopState
from rfcl.logger import LoggerConfig
from rfcl.utils import tools
from tqdm import tqdm

from rfcl.agents.sac import loss
from rfcl.agents.sac.config import SACConfig, TimeStep
from rfcl.agents.sac.networks import ActorCritic, DiagGaussianActor

@struct.dataclass
class TrainStepMetrics:
    train_stats: Any
    train: Any
    update: Any
    time: Any


@struct.dataclass
class SACTrainState:
    # model states
    ac: ActorCritic

    loop_state: EnvLoopState
    # rng
    rng_key: PRNGKey

    # monitoring
    total_env_steps: int
    """
    Total env steps sampled so far
    """
    training_steps: int
    """
    Total training steps so far
    """
    initialized: bool
    """
    When False, will automatically reset the loop state. This is usually false when starting training. When resuming training
    it will try to proceed from the previous loop state
    """


class SAC(BasePolicy):
    def __init__(
        self,
        jax_env: bool,
        ac: ActorCritic,
        env,
        eval_env=None,
        logger_cfg: LoggerConfig = None,
        cfg: SACConfig = {},
        offline_buffer = None,
    ):
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg
        super().__init__(jax_env, env, eval_env, cfg.num_envs, cfg.num_eval_envs, logger_cfg)
        self.offline_buffer = offline_buffer
        self.state: SACTrainState = SACTrainState(
            ac=ac,
            loop_state=EnvLoopState(),
            total_env_steps=0,
            training_steps=0,
            rng_key=None,
            initialized=False,
        )

        if jax_env:
            def seed_sampler(rng_key):
                return env.action_space().sample(rng_key)
        else:
            def seed_sampler(rng_key):
                return jax.random.uniform(
                    rng_key,
                    shape=(cfg.num_envs, *env.single_action_space.shape),
                    minval=-1.0,
                    maxval=1.0,
                    dtype=float,
                )
        self.seed_sampler = seed_sampler

        # Define our buffer
        buffer_config = dict(
            action=((self.action_dim,), self.action_space.dtype),
            reward=((), np.float32),
            mask=((), float),
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["env_obs"] = (
                self.obs_shape,
                {k: self.observation_space[k].dtype for k in self.observation_space},
            )
        else:
            buffer_config["env_obs"] = (self.obs_shape, np.float32)
        buffer_config["next_env_obs"] = buffer_config["env_obs"]

        self.replay_buffer = GenericBuffer(
            buffer_size=self.cfg.replay_buffer_capacity,
            num_envs=self.cfg.num_envs,
            config=buffer_config,
        )

        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

        if self.jax_env:
            self._env_step = jax.jit(self._env_step, static_argnames=["seed"])

    @partial(jax.jit, static_argnames=["self", "seed"])
    def _sample_action(self, rng_key, actor: DiagGaussianActor, env_obs, seed=False):
        if seed:
            a = self.seed_sampler(rng_key)
        else:
            dist: distrax.Distribution = actor(env_obs)
            a = dist.sample(seed=rng_key)
        return a, {}

    def _env_step(self, rng_key: PRNGKey, loop_state: EnvLoopState, actor: DiagGaussianActor, seed=False):
        if self.jax_env:
            rng_key, *env_rng_keys = jax.random.split(rng_key, self.cfg.num_envs + 1)
            data, loop_state = self.loop.rollout(jnp.stack(env_rng_keys), loop_state, actor, partial(self._sample_action, seed=seed), 1)
        else:
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            data, loop_state = self.loop.rollout([env_rng_key], loop_state, actor, partial(self._sample_action, seed=seed), 1)
        return loop_state, data

    def train(self, rng_key: PRNGKey, steps: int, callback_fn = None, verbose=1):
        """
        Args :
            rng_key: PRNGKey,
                Random key to seed the training with. It is only used if train() was never called before, otherwise the code uses self.state.rng_key
            steps : int
                Max number of environment samples before training is stopped.
        """
        train_start_time = time.time()

        rng_key, reset_rng_key = jax.random.split(rng_key, 2)

        # if env_obs is None, then this is the first time calling train and we prepare the environment
        if not self.state.initialized:
            loop_state = self.loop.reset_loop(reset_rng_key)
            self.state = self.state.replace(
                loop_state=loop_state,
                rng_key=rng_key,
                initialized=True,
            )

        start_step = self.state.total_env_steps

        if verbose:
            pbar = tqdm(total=steps + self.state.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        while self.state.total_env_steps < start_step + steps:
            rng_key, train_rng_key = jax.random.split(self.state.rng_key, 2)
            self.state, train_step_metrics = self.train_step(train_rng_key, self.state)
            self.state = self.state.replace(rng_key=rng_key)

            # evaluate the current trained actor periodically
            if (
                self.eval_loop is not None
                and tools.reached_freq(self.state.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size)
                and self.state.total_env_steps > self.cfg.num_seed_steps
            ):
                rng_key, eval_rng_key = jax.random.split(rng_key, 2)
                eval_results = self.evaluate(
                    eval_rng_key,
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    params=self.state.ac.actor,
                    apply_fn=self.state.ac.act,
                )
                self.logger.store(
                    tag="test",
                    ep_ret=eval_results["eval_ep_rets"],
                    ep_len=eval_results["eval_ep_lens"],
                )
                self.logger.store(tag="test_stats", **eval_results["stats"])
                self.logger.log(self.state.total_env_steps)
                self.logger.reset()

            self.logger.store(tag="train", **train_step_metrics.train)
            self.logger.store(tag="train_stats", **train_step_metrics.train_stats)
            
            ### Log Metrics ###
            if verbose: pbar.update(n=env_rollout_size)
            total_time = time.time() - train_start_time
            if tools.reached_freq(self.state.total_env_steps, self.cfg.log_freq, step_size=env_rollout_size):
                update_aux = tools.flatten_struct_to_dict(train_step_metrics.update)
                self.logger.store(tag="train", training_steps=self.state.training_steps, **update_aux)
                self.logger.store(
                    tag="time",
                    total=total_time,
                    SPS=self.state.total_env_steps / total_time,
                    total_env_steps=self.state.total_env_steps,
                    **train_step_metrics.time
                )
            # log and export the metrics
            self.logger.log(self.state.total_env_steps)
            self.logger.reset()

            # save checkpoints. Note that the logger auto saves upon metric improvements
            if tools.reached_freq(self.state.total_env_steps, self.cfg.save_freq, env_rollout_size):
                self.save(
                    os.path.join(self.logger.model_path, f"ckpt_{self.state.total_env_steps}.jx"),
                    with_buffer=self.cfg.save_buffer_in_checkpoints,
                )

            if callback_fn is not None:
                stop = callback_fn(locals())
                if stop:
                    print(f"Early stopping at {self.state.total_env_steps} env steps")
                    break
    def train_step(self, rng_key: PRNGKey, state: SACTrainState) -> Tuple[SACTrainState, TrainStepMetrics]:
        """
        Perform a single training step

        In SAC this is composed of collecting cfg.steps_per_env * cfg.num_envs of interaction data with a random sample or policy (depending on cfg.num_seed_steps)
        then performing gradient updates

        TODO: If a jax-env is used, this step is jitted
        """

        ac = state.ac
        loop_state = state.loop_state
        total_env_steps = state.total_env_steps
        training_steps = state.training_steps

        train_custom_stats = defaultdict(list)
        train_metrics = defaultdict(list)
        time_metrics = dict()

        # perform a rollout
        # TODO make this buffer collection jittable
        rollout_time_start = time.time()
        for _ in range(self.cfg.steps_per_env):
            rng_key, env_rng_key = jax.random.split(rng_key, 2)
            (next_loop_state, data) = self._env_step(
                env_rng_key,
                loop_state,
                ac.actor,
                seed=(total_env_steps <= self.cfg.num_seed_steps and not (self.cfg.seed_with_policy)),
            )
            if not self.jax_env:
                final_infos = data[
                    "final_info"
                ]  # in gym loop this is just a list. in jax loop it should be a pytree with leaf shape (B, ) and a corresponding mask
                del data["final_info"]
                data = DefaultTimeStep(**data)
            else:
                final_infos = None  # TODO handle final infos in jax envs

            # move data to numpy
            data: DefaultTimeStep = jax.tree_map(lambda x: np.array(x)[:, 0], data)
            terminations = data.terminated
            truncations = data.truncated
            dones = terminations | truncations
            masks = ((~dones) | (truncations)).astype(float)
            if dones.any():
                # note for continuous task wrapped envs where there is no early done, all envs finish at the same time unless
                # they are staggered. So masks is never false.
                # if you want to always value bootstrap set masks to true.
                train_metrics["ep_ret"].append(data.ep_ret[dones])
                train_metrics["ep_len"].append(data.ep_len[dones])
                if not self.jax_env:  # TODO fix for jax envs
                    for i, final_info in enumerate(final_infos):
                        if final_info is not None:
                            if "stats" in final_info:
                                for k in final_info["stats"]:
                                    train_custom_stats[k].append(final_info["stats"][k])
            self.replay_buffer.store(
                env_obs=data.env_obs,
                reward=data.reward,
                action=data.action,
                mask=masks,
                next_env_obs=data.next_env_obs,
            )
            loop_state = next_loop_state

        # log time metrics
        
        for k in train_metrics:
            train_metrics[k] = np.concatenate(train_metrics[k]).flatten()
        rollout_time = time.time() - rollout_time_start
        time_metrics["rollout_time"] = rollout_time
        time_metrics["rollout_fps"] = self.cfg.num_envs * self.cfg.steps_per_env / rollout_time
        state = state.replace(
            loop_state=loop_state,
            total_env_steps=total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env,
        )
        # update policy
        update_aux = dict()
        if self.state.total_env_steps >= self.cfg.num_seed_steps:
            update_time_start = time.time()
            rng_key, update_rng_key, online_sample_key, offline_sample_key = jax.random.split(rng_key, 4)
            if self.offline_buffer is not None:
                batch = self.replay_buffer.sample_random_batch(online_sample_key, self.cfg.batch_size * self.cfg.grad_updates_per_step // 2)
                offline_batch = self.offline_buffer.sample_random_batch(offline_sample_key, self.cfg.batch_size * self.cfg.grad_updates_per_step // 2)
                batch = tools.combine(batch, offline_batch)
            else:
                batch = self.replay_buffer.sample_random_batch(online_sample_key, self.cfg.batch_size * self.cfg.grad_updates_per_step)
            
            batch = TimeStep(**batch)
            ac, update_aux = self.update_parameters(
                update_rng_key,
                ac,
                batch,
            )
            state = state.replace(
                ac=ac,
                training_steps=training_steps + self.cfg.grad_updates_per_step,
            )
            update_time = time.time() - update_time_start
            time_metrics["update_time"] = update_time

        return state, TrainStepMetrics(time=time_metrics, train=train_metrics, update=update_aux, train_stats=train_custom_stats)

    @partial(jax.jit, static_argnames=["self"])
    def update_parameters(
        self,
        rng_key: PRNGKey,
        ac: ActorCritic,
        batch: TimeStep,
    ) -> Tuple[ActorCritic, Any]:
        """
        Update actor critic parameters using the given batch
        """
        # init dummy values
        critic_update_aux = loss.CriticUpdateAux(q=0, critic_loss=0)
        new_actor, actor_update_aux = ac.actor, loss.ActorUpdateAux(actor_loss=0, entropy=0)
        new_temp, temp_update_aux = ac.temp, loss.TempUpdateAux(temp=ac.temp(), temp_loss=0)
        mini_batch_size = self.cfg.batch_size
        assert mini_batch_size * self.cfg.grad_updates_per_step == batch.action.shape[0]
        assert self.cfg.grad_updates_per_step % self.cfg.actor_update_freq == 0
        update_rounds = self.cfg.grad_updates_per_step / self.cfg.actor_update_freq
        grad_updates_per_round = self.cfg.grad_updates_per_step / update_rounds
        
        # jitted code to update critics and allow delayed actor updates. No for loops for fast compilation
        def _critic_updates(data, batch):
            (ac, critic_update_aux, rng_key) = data
            rng_key, critic_update_rng_key = jax.random.split(rng_key, 2)
            new_critic, critic_update_aux = loss.update_critic(
                critic_update_rng_key,
                ac,
                batch,
                self.cfg.discount,
                self.cfg.backup_entropy,
                self.cfg.num_min_qs,
                self.cfg.num_qs
            )
            ac = ac.replace(critic=new_critic)
            new_target = loss.update_target(ac.critic, ac.target_critic, self.cfg.tau)
            ac = ac.replace(target_critic=new_target)
            return (ac, critic_update_aux, rng_key), None
        
        def _update(data, batch):
            # for each update, we perform a number of critic updates followed by an actor update depending on actor update frequency
            (ac, critic_update_aux, actor_update_aux, temp_update_aux, rng_key) = data
            
            mini_batches = jax.tree_util.tree_map(lambda x : jnp.array(jnp.split(x, grad_updates_per_round)), batch)
            (ac, critic_update_aux, rng_key), _ = jax.lax.scan(_critic_updates, (ac, critic_update_aux, rng_key), mini_batches)
            
            rng_key, actor_update_rng_key = jax.random.split(rng_key, 2)
            mini_batch = mini_batch = jax.tree_util.tree_map(lambda x : x[-mini_batch_size:], batch)
            new_actor, actor_update_aux = loss.update_actor(actor_update_rng_key, ac, mini_batch)
            if self.cfg.learnable_temp:
                new_temp, temp_update_aux = loss.update_temp(ac.temp, actor_update_aux.entropy, self.cfg.target_entropy)
            ac = ac.replace(actor=new_actor, temp=new_temp)
            return (ac, critic_update_aux, actor_update_aux, temp_update_aux, rng_key), None
        
        # TODO this step may use extra GPU memory.
        mini_batches = jax.tree_util.tree_map(lambda x : jnp.array(jnp.split(x, update_rounds)), batch)
        init_vals = (ac, critic_update_aux, actor_update_aux, temp_update_aux, rng_key)
        (ac, critic_update_aux, actor_update_aux, temp_update_aux, rng_key), _ = jax.lax.scan(_update, init_vals, mini_batches)

        return (
            ac,
            loss.UpdateMetrics(
                critic=critic_update_aux,
                actor=actor_update_aux,
                temp=temp_update_aux,
            ),
        )

    def state_dict(self, with_buffer=False):
        ac = flax.serialization.to_bytes(self.state.ac)
        state_dict = dict(
            train_state=self.state.replace(ac=ac),
            logger=self.logger.state_dict(),
        )
        if with_buffer:
            state_dict["replay_buffer"] = self.replay_buffer
        return state_dict

    def save(self, save_path: str, with_buffer=False):
        stime = time.time()
        state_dict = self.state_dict(with_buffer=with_buffer)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            # TODO replace pickle with something more efficient for replay buffers?
        print(f"Saving Checkpoint {save_path}.", "Time:", time.time() - stime)

    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Checkpoint {load_path}", state_dict["logger"])
        self.load(state_dict)

    def load(self, data):
        ac = flax.serialization.from_bytes(self.state.ac, data["train_state"].ac)
        # use serialized ac model
        self.state: SACTrainState = data["train_state"].replace(ac=ac)
        # set initialized to False so previous env data is reset if it's not a jax env with env states we can start from
        if not self.jax_env:
            self.state = self.state.replace(initialized=False)
        if self.logger is not None:
            self.logger.load(data["logger"])
        else:
            print("Skip loading logger. No log data will be overwritten/saved")
        if "replay_buffer" in data:
            replay_buffer: GenericBuffer = data["replay_buffer"]
            print(f"Loading replay buffer which contains {replay_buffer.size() * replay_buffer.num_envs} interactions")
            self.replay_buffer = replay_buffer

    def load_policy_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Checkpoint {load_path}")
        return self.load_policy(state_dict)

    def load_policy(self, data) -> ActorCritic:
        ac = flax.serialization.from_bytes(self.state.ac, data["train_state"].ac)
        return ac
