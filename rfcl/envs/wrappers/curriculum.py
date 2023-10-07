"""
Code implementing the reverse curriculum wrapper
"""

import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnvWrapper
from gymnasium.vector.vector_env import VectorEnv


def create_filled_deque(maxlen, fill_value):
    return deque([fill_value] * maxlen, maxlen=maxlen)


@dataclass
class DemoCurriculumMetadata:
    start_step: int = None  # t_i
    total_steps: int = None  # T_i
    success_rate_buffer = deque(maxlen=2)  # size of this is m
    episode_steps_back = deque(maxlen=2)
    solved: bool = False  # whether we have reverse solved this demo


@dataclass
class InitialStateMetadata:
    start_steps: np.ndarray = None
    start_steps_density: np.ndarray = None
    total_steps: int = None


@dataclass
class EpisodeMetadata:
    start_step: int
    demo_id: int


class InitialStateWrapper(gymnasium.Wrapper):
    """
    Wrapper that allows to keep a dataset of demonstrations with states and initialize to some time t in those demonstrations
    """

    def __init__(
        self,
        env,
        states_dataset,
        demo_horizon_to_max_steps_ratio: int = 3,
    ):
        super().__init__(env)
        self.states_dataset = states_dataset

        self._state_seed: int = None
        self._state_rng: np.random.RandomState = np.random.RandomState(np.random.randint(2**32))

        # the current episode info
        self.current_episode_metadata = EpisodeMetadata(start_step=None, demo_id=None)

        self.demo_states: np.ndarray = None
        self.demo_metadata = defaultdict(InitialStateMetadata)
        self.demo_ids = list(states_dataset.keys())
        self.demo_id_density = np.zeros(len(self.demo_ids))
        for demo_id in states_dataset:
            self.demo_metadata[demo_id] = InitialStateMetadata(
                total_steps=len(self.states_dataset[demo_id]["state"]),
                start_steps=np.array([0]),
                start_steps_density=np.array([1]),
            )
        for i in range(len(self.demo_ids)):
            self.demo_id_density[i] = 1 / len(self.demo_ids)

        self.demo_horizon_to_max_steps_ratio = demo_horizon_to_max_steps_ratio

        # look for all observation wrappers so we can apply them one by one in correct order later
        obs_fns = []
        curr_env = self.env
        while True:
            try:
                obs_fns.append(curr_env.get_wrapper_attr("observation"))
            except AttributeError:
                pass
            if hasattr(curr_env, "env"):
                curr_env = curr_env.env
            else:
                break
        self.obs_fns = list(dict.fromkeys(obs_fns))[::-1]

    def set_env_state(self, state):
        raise NotImplementedError()

    def get_env_obs(self):
        raise NotImplementedError()

    def set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        if seed is None:
            # only set state seed if we haven't before
            self._state_seed = self._state_rng.randint(2**32)
        else:
            self._state_seed = seed
        self._state_rng = np.random.RandomState(self._state_seed)

    def set_demo_start_steps(self, t_is, id_to_start_steps, id_to_start_steps_density):
        """Set the distribution of start steps for each demo"""
        for demo_id in id_to_start_steps:
            self.demo_metadata[demo_id].start_steps = id_to_start_steps[demo_id]
            self.demo_metadata[demo_id].start_steps_density = id_to_start_steps_density[demo_id]
        for i, demo_id in enumerate(self.demo_ids):
            t_i = t_is[demo_id]
            self.demo_id_density[i] = t_i / self.demo_metadata[demo_id].total_steps
            if t_i == 0:
                self.demo_id_density[i] = 1e-6
        self.demo_id_density = self.demo_id_density / self.demo_id_density.sum()

    def reset(self, *, seed=None, options=None):
        self.set_episode_rng(seed)

        # sample a demo with states and reset the environment
        demo_id = self._state_rng.choice(self.demo_ids, p=self.demo_id_density)

        state_info = self.states_dataset[demo_id]
        obs, info = self.env.reset(**copy.deepcopy(state_info["reset_kwargs"]))
        self.demo_states = state_info["state"]

        # sample a start step
        metadata = self.demo_metadata[demo_id]
        start_step = self._state_rng.choice(metadata.start_steps, p=metadata.start_steps_density)
        self.set_env_state(self.demo_states[start_step])

        # retrieve the actual new observation which we reset to
        obs = self.get_env_obs()
        # apply any wrappers if they exist
        if len(self.obs_fns) > 0:
            obs = self.apply_observation_wrappers(obs)

        self.current_episode_metadata = EpisodeMetadata(start_step=start_step, demo_id=demo_id)
        self.step_count = 0  # TODO remove in favor of info["eps_len"]
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1

        demo_id = self.current_episode_metadata.demo_id

        info["demo_id"] = demo_id
        metadata = self.demo_metadata[demo_id]
        info["stats"]["steps_back"] = metadata.total_steps - self.current_episode_metadata.start_step
        info["steps_back"] = info["stats"]["steps_back"]
        info["stats"]["sampled_start_step_frac"] = self.current_episode_metadata.start_step / (metadata.total_steps - 1)
        info["sampled_start_step_frac"] = info["stats"]["sampled_start_step_frac"]

        # handle per-demo timelimit here
        if self.demo_horizon_to_max_steps_ratio > 0:
            dynamic_timelimit = 16 + (metadata.total_steps - self.current_episode_metadata.start_step) // self.demo_horizon_to_max_steps_ratio
            if self.step_count >= dynamic_timelimit:
                truncated = True

        return observation, reward, terminated, truncated, info

    def apply_observation_wrappers(self, obs):
        for obs_fn in self.obs_fns:
            obs = obs_fn(obs)
        return obs


GLOBAL_SUCCESS_RATE_THRESHOLD = 0.9
GLOBAL_SUCCESS_HISTORY_BUFFER_SIZE = 100


class ReverseCurriculumWrapper(VectorEnvWrapper):
    """
    Reverse Curriculum Wrapper

    Parameters:

    env: Vectorized environment to apply the wrapper to

    states_dataset: A map from demo id to a dict with keys reset_kwargs and state. reset_kwargs is used to reset the environment
        to the same state that demonstration started from (e.g. defining object shapes that are not parameterized by state).
        state is the demo environment states that we can reset the environment directly to

    reverse_step_size (δ): int - value equal to δ in RFCL paper. Represents the distance between each reverse curriculum stage. Higher value means
        allows for faster, but less stable, progression of the reverse curriculum.

    per_demo_buffer_size: int - Buffer size equal to m in RFCL paper. Number of sequential successes at end of episodes before we consider advancing curriculum
        for a demo.

    start_step_sampler (K): str - Can be "geometric" or "fixed_point", and sets the distribution K in the RFCL paper.

    link_envs: list of other envs wrapped with ReverseCurriculumWrapper to share the curriculum settings with. Whenever the current
        curriculum is changed, all other envs linked here are given the new settings

    eval_mode: bool - whether to use this wrapper in evaluation mode or not. In evaluation mode, only stats are collected, the curriculum is
        not modified or executed

    verbose: int - Defaults to 0. If > 0, then prints additional logs to terminal
    """

    def __init__(
        self,
        env: VectorEnv,
        states_dataset,
        curriculum_method: str = "per_demo",
        reverse_step_size: int = 8,
        per_demo_buffer_size: int = 3,
        start_step_sampler: str = "geometric",
        link_envs: List["ReverseCurriculumWrapper"] = [],
        eval_mode: bool = False,
        verbose=0,
    ):
        super().__init__(env)

        self.link_envs = link_envs
        self.verbose = verbose
        self.eval_mode = eval_mode
        self.states_dataset = states_dataset
        self.reverse_step_size = reverse_step_size
        self.per_demo_buffer_size = per_demo_buffer_size

        self.global_success_rate_history = create_filled_deque(per_demo_buffer_size * len(states_dataset), 0)  # only used for global curriculum

        self.demo_metadata = defaultdict(DemoCurriculumMetadata)
        for demo_id in states_dataset:
            self.demo_metadata[demo_id].start_step = max(len(self.states_dataset[demo_id]["state"]) - 1, 0)
            self.demo_metadata[demo_id].total_steps = len(self.states_dataset[demo_id]["state"])
            self.demo_metadata[demo_id].success_rate_buffer = create_filled_deque(per_demo_buffer_size, 0)
            self.demo_metadata[demo_id].episode_steps_back = create_filled_deque(per_demo_buffer_size, -1)

        self.curriculum_method = curriculum_method
        assert self.curriculum_method in ["global", "per_demo"]

        self.start_step_sampler = start_step_sampler
        assert self.start_step_sampler in ["fixed_point", "geometric"]

        self.sync_envs()

        self.reset_to_states_enabled = True

    def sync_envs(self):
        """
        Sync the demo metadata for initial start state wrappers.
        Call this whenever self.demo_metadata changes
        """
        if self.verbose > 0:
            print("Syncing Metadata")
        t_is = {}
        for x in self.demo_metadata:
            t_is[x] = self.demo_metadata[x].start_step
        if self.start_step_sampler == "geometric":
            start_steps = {}
            start_steps_densities = {}
            for x in self.demo_metadata:
                metadata = self.demo_metadata[x]
                x_start_steps_list = []
                x_start_steps_density_list = [0.5, 0.25, 0.125, 0.125 / 2, 0.125 / 2]
                for i in range(5):
                    x_start_steps_list.append(min(metadata.start_step + i, metadata.total_steps - 1))
                start_steps[x] = np.array(x_start_steps_list)
                start_steps_densities[x] = np.array(x_start_steps_density_list)
        elif self.start_step_sampler == "fixed_point":
            start_steps = {x: np.array([self.demo_metadata[x].start_step]) for x in self.demo_metadata}
            start_steps_densities = {x: np.array([1]) for x in self.demo_metadata}

        self.env.unwrapped.call("set_demo_start_steps", t_is, start_steps, start_steps_densities)  # sync curriculum settings

        # sync metadata and configs across linked reverse curriculum wrapper envs
        for link_env in self.link_envs:
            link_env.demo_metadata = copy.deepcopy(self.demo_metadata)
            link_env.reverse_step_size = self.reverse_step_size
            link_env.global_success_rate_history = create_filled_deque(self.per_demo_buffer_size * len(self.states_dataset), 0)
            link_env.sync_envs()

    def step_wait(self):
        observation, reward, terminated, truncated, info = super().step_wait()
        if self.reset_to_states_enabled:
            if terminated.any() or truncated.any():
                for final_info, exists in zip(info["final_info"], info["_final_info"]):
                    if not exists:
                        continue
                    demo_id = final_info["demo_id"]
                    success = final_info["success"]
                    metadata = self.demo_metadata[demo_id]

                    # record successes only when steps back is equal to the current frontier / start step t_i assigned to demo tau_i
                    if final_info["steps_back"] == metadata.total_steps - metadata.start_step:
                        metadata.success_rate_buffer.append(int(success))
                        metadata.episode_steps_back.append(final_info["steps_back"])
                        self.global_success_rate_history.append(int(success))
                if not self.eval_mode:
                    self.step_curriculum()
        return observation, reward, terminated, truncated, info

    def step_curriculum(self):
        if self.curriculum_method == "per_demo":
            change = False
            for demo_id in self.states_dataset:
                metadata = self.demo_metadata[demo_id]
                running_success_rate_mean = np.mean(metadata.success_rate_buffer)
                if running_success_rate_mean >= 1.0:
                    metadata.success_rate_buffer = create_filled_deque(self.per_demo_buffer_size, 0)
                    metadata.episode_steps_back = create_filled_deque(self.per_demo_buffer_size, -1)
                    if metadata.start_step > 0:
                        metadata.start_step = max(metadata.start_step - self.reverse_step_size, 0)
                        if self.verbose > 0:
                            print(f"Demo {demo_id} stepping back to {metadata.start_step}")
                        change = True
                    else:
                        if not metadata.solved:
                            if self.verbose > 0:
                                print(f"Demo {demo_id} is reverse solved!")
                            metadata.solved = True
                            change = True
            if change:
                self.sync_envs()
        elif self.curriculum_method == "global":
            change = False
            if np.mean(self.global_success_rate_history) >= GLOBAL_SUCCESS_RATE_THRESHOLD:
                self.global_success_rate_history = create_filled_deque(self.per_demo_buffer_size * len(self.states_dataset), 0)
                for demo_id in self.states_dataset:
                    metadata = self.demo_metadata[demo_id]
                    if metadata.start_step > 0:
                        metadata.start_step = max(metadata.start_step - self.reverse_step_size, 0)
                        if self.verbose > 0:
                            print(f"Demo {demo_id} stepping back to {metadata.start_step}")
                        change = True
                    else:
                        if not metadata.solved:
                            if self.verbose > 0:
                                print(f"Demo {demo_id} is reverse solved!")
                            metadata.solved = True
                            change = True
            if change:
                self.sync_envs()
        elif self.curriculum_method == "adaptive":
            raise NotImplementedError()
