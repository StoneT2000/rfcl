"""
Assumes the dataset is of the ManiSkill2 format (a .h5 file and corresponding .json file)
"""

import json
from collections import defaultdict
from typing import Callable

import h5py
import numpy as np
import jax

def get_states_dataset(demo_dataset_path, skip_failed=True, num_demos: int = -1, shuffle: bool = False):
    states_dataset = defaultdict(dict)

    demo_dataset = h5py.File(demo_dataset_path)
    with open(demo_dataset_path.replace(".h5", ".json"), "r") as f:
        demo_dataset_meta = json.load(f)
    if num_demos == -1:
        num_demos = len(demo_dataset_meta["episodes"])
    load_count = 0
    if shuffle:
        np.random.shuffle(demo_dataset_meta["episodes"])
    for episode in demo_dataset_meta["episodes"]:
        if not episode["info"]["success"] and skip_failed:
            continue
        demo_id = episode["episode_id"]
        demo = demo_dataset[f"traj_{demo_id}"]
        reset_kwargs = episode["reset_kwargs"]

        # this is specifically for adroit envs that use options
        if "initial_state_dict" in reset_kwargs["options"]:
            for k in reset_kwargs["options"]["initial_state_dict"]:
                reset_kwargs["options"]["initial_state_dict"][k] = np.array(reset_kwargs["options"]["initial_state_dict"][k])

        # handle both dict like env states and vector env states
        if isinstance(demo["env_states"], h5py.Dataset):
            env_states = np.array(demo["env_states"])
        else:
            env_states = [dict(zip(demo["env_states"], t)) for t in zip(*demo["env_states"].values())]
        seed = None
        if "episode_seed" in episode:
            seed = episode["episode_seed"]
        elif "seed" in reset_kwargs:
            seed = reset_kwargs["seed"]
        states_dataset[demo_id] = dict(state=env_states, seed=seed, reset_kwargs=reset_kwargs, demo_id=demo_id)

        load_count += 1
        if load_count >= num_demos:
            break
    print(f"Converted {demo_dataset_path}, skip_failed={skip_failed}, shuffle={shuffle}, loaded {load_count} demos")
    demo_dataset.close()
    return states_dataset


class ReplayDataset:
    """
    Loads a ManiSkill2 format dataset

    demo_dataset_path : str - path to the demonstration dataset .h5 file (which should have a corresponding .json file next to it)

    shuffle : bool - whether to shuffle the order of demonstrations in the dataset

    skip_failed : bool - whether to ignore demonstrations that do not end in success

    num_demos : int - the number of demonstrations to load

    reward_mode : str - the reward mode to use. If "sparse", will relabel rewards in the demonstration frames based on 
        whether the frame is a success state (+1) or not (0). If "negative_sparse", will also relabel rewards but with 
        0 for success states and -1 otherwise
    
    eps_ids : list of episode/demonstration ids to load from the dataset as opposed to sampling with the code in here.

    data_action_scale : scales the magnitude of each action dimension of all demonstration actions and saves the 
        corresponding action scale to be used in an action rescale wrapper. Makes problem easier by reducing the 
        action space size and constraining it to a factor of what the demonstrations use

    action_scale : pass in a hardcoded action scale. If none, uses data_action_scale instead to generate it
    """

    def __init__(
        self,
        demo_dataset_path,
        shuffle=False,
        skip_failed=False,
        num_demos=-1,
        reward_mode="sparse",
        eps_ids=None,
        action_scale=None,
        data_action_scale=None,
    ) -> None:
        self.demo_dataset_path = demo_dataset_path
        demo_dataset = h5py.File(demo_dataset_path)
        # assert reward_mode in ["sparse", "negative_sparse"]

        all_observations = []
        all_next_observations = []
        all_actions = []
        all_rewards = []

        self.eps_ids = []

        with open(demo_dataset_path.replace(".h5", ".json"), "r") as f:
            demo_dataset_meta = json.load(f)
        if num_demos == -1:
            num_demos = len(demo_dataset_meta["episodes"])
        load_count = 0
        if shuffle:
            np.random.shuffle(demo_dataset_meta["episodes"])

        total_frames = 0
        if eps_ids is None:
            for episode in demo_dataset_meta["episodes"]:
                if not episode["info"]["success"] and skip_failed:
                    continue
                self.eps_ids.append(episode["episode_id"])
                load_count += 1
                if load_count >= num_demos:
                    break
        else:
            self.eps_ids = list(eps_ids)
            load_count = len(eps_ids)
        for eps_id in self.eps_ids:
            demo = demo_dataset[f"traj_{eps_id}"]
            actions = np.array(demo["actions"])

            if reward_mode == "sparse":
                rewards = np.array(demo["success"]).astype(np.float_)
            elif reward_mode == "negative_sparse":
                rewards = rewards - 1
            else:
                rewards = np.array(demo["rewards"])
            all_rewards.append(rewards)
            obs = np.array(demo["obs"])
            total_frames += len(rewards)

            all_observations.append(obs[:-1])
            all_next_observations.append(obs[1:])
            all_actions.append(actions)

        self.data = dict(
            env_obs=np.concatenate(all_observations),
            next_env_obs=np.concatenate(all_next_observations),
            reward=np.concatenate(all_rewards),
            action=np.concatenate(all_actions),
            mask=np.ones(total_frames),
        )
        self.action_scale = None
        act_max, act_min = self.data["action"].max(0), self.data["action"].min(0)

        # determine a natural data action scale based on demonstration data.
        if data_action_scale is not None and action_scale is None:
            print(f"Scaling actions in dataset with data_action_scale={data_action_scale}")
            print(f"act_max: {act_max} - act_min: {act_min}")
            if (np.abs(act_max) > 1).any() or (np.abs(act_min) > 1).any():
                raise ValueError("Dataset actions are not in the range of [-1, 1]. Scale them first!")
            self.action_magnitudes = np.max([np.abs(act_max), np.abs(act_min)], 0)
            self.action_scale = (1 / (self.action_magnitudes)) * (1 / data_action_scale)
            self.action_scale[(act_max == 1) | (act_min == -1)] = 1.0
            self.action_scale[(act_max == 0) & (act_min == 0)] = 1.0
        if action_scale is not None:
            print("Using given action scale", action_scale)
            self.action_scale = action_scale
            self.data["action"] = self.data["action"] * self.action_scale
            act_max, act_min = self.data["action"].max(0), self.data["action"].min(0)
            print(f"new act_max: {act_max} - act_min: {act_min}")
            print("Action scale", self.action_scale)
        print(f"Loaded {load_count} demos, total {len(self.data['reward'])} frames. Loaded {self.eps_ids}")
        self.size = len(self.data["env_obs"])

    def sample_random_batch(self, rng_key: jax.random.PRNGKey, batch_size: int):
        """
        Sample a batch of data
        """
        batch_ids = np.random.randint(self.size, size=batch_size)

        batch_data = dict()
        for k in self.data:
            batch_data[k] = self.data[k][batch_ids]
        return batch_data
