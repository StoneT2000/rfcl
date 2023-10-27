"""
InitialStateWrapper for Adroit along with useful video recording tools.
"""

import copy
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from gymnasium import spaces
from gymnasium_robotics.envs.adroit_hand.adroit_door import AdroitHandDoorEnv
from gymnasium_robotics.envs.adroit_hand.adroit_relocate import AdroitHandRelocateEnv

from rfcl.envs.wrappers.curriculum import InitialStateWrapper
from rfcl.utils.io_utils import dump_json
from rfcl.utils.visualization import (
    extract_scalars_from_info,
    images_to_video,
    put_info_on_image,
)


class AdroitInitialStateWrapper(InitialStateWrapper):
    def set_env_state(self, state):
        env: AdroitHandDoorEnv = self.env.unwrapped
        if isinstance(env, AdroitHandRelocateEnv):
            # fix bug where Mujoco env cannot set state correctly in Relocate when there is close contact with ball
            # very strange error but I believe it may be due to the lower fidelity simulation Adroit envs have
            qp = state["qpos"]
            qv = state["qvel"]
            env.model.body_pos[env.obj_body_id] = state["obj_pos"]
            env.model.site_pos[env.target_obj_site_id] = state["target_pos"]
            env.set_state(qp, qv)  # call this and let Mujoco step forward once
            diff = env.model.body_pos[env.obj_body_id] - env.data.xpos[env.obj_body_id]
            env.model.body_pos[env.obj_body_id] = state["obj_pos"] + diff
            return env.set_state(qp, qv)
        return env.set_env_state(state)

    def get_env_obs(self):
        env: AdroitHandDoorEnv = self.env.unwrapped
        return env._get_obs()


def clean_trajectories(h5_file: h5py.File, json_dict: dict, prune_empty_action=True):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    assert len(h5_file) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            del h5_file[traj_id]
            continue

        if new_traj_id != traj_id:
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes


class RecordEpisodeWrapper(gym.Wrapper):
    """
    record trajectories for mujoco envs in the MS2 format
    """

    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        save_on_reset=True,
        clean_on_close=True,
        info_on_video=True,
        name_prefix="",
    ):
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_on_reset = save_on_reset
        self.name_prefix = name_prefix

        self._elapsed_steps = 0
        self._eps_ret = 0
        self._episode_id = -1
        self._episode_data = []
        self._episode_info = {}

        self.save_trajectory = save_trajectory
        self.clean_on_close = clean_on_close
        self.save_video = save_video
        self.info_on_video = info_on_video
        self._render_images = []
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")

            # Use a separate json to store non-array data
            self._json_path = self._h5_file.filename.replace(".h5", ".json")

            self._json_data = dict(
                env_info=dict(
                    env_id=self.env.spec.id,
                    max_episode_steps=self.env.spec.max_episode_steps,
                    env_kwargs=self.env.spec.kwargs,
                ),
                episodes=[],
            )

    def reset(self, *args, **kwargs):
        if self.save_on_reset and self._episode_id >= 0:
            if self._elapsed_steps == 0:
                self._episode_id -= 1
            self.flush_trajectory(ignore_empty_transition=True)
            self.flush_video(ignore_empty_transition=True)

        # Clear cache
        self._elapsed_steps = 0
        self._eps_ret = 0
        self._episode_id += 1
        self._episode_data = []
        self._episode_info = {}
        self._render_images = []

        reset_kwargs = copy.deepcopy(kwargs)
        obs, info = super().reset(*args, **kwargs)

        if self.save_trajectory:
            env: AdroitHandDoorEnv = self.env.unwrapped
            state = env.get_env_state()
            if isinstance(env, AdroitHandRelocateEnv):
                del state["hand_qpos"]
                del state["palm_pos"]
            data = dict(s=state, o=obs, a=None, r=None, done=None, info=None)
            self._episode_data.append(data)
            self._episode_info.update(
                episode_id=self._episode_id,
                reset_kwargs=reset_kwargs,
                elapsed_steps=0,
            )

        if self.save_video:
            self._render_images.append(self.env.render())

        return obs, info

    def save_data(self, state, obs, action, rew, done, info):
        if self.save_trajectory:
            if isinstance(self.env.unwrapped, AdroitHandRelocateEnv):
                # fixes a bug in the state representation of Adroit Relocate in gymnasium
                del state["hand_qpos"]
                del state["palm_pos"]
            data = dict(s=state, o=obs, a=action, r=rew, done=done, info=info)
            self._episode_data.append(data)
            self._episode_info["elapsed_steps"] += 1
            self._episode_info["info"] = info

        if self.save_video:
            image = self.env.render()
            if self.info_on_video:
                scalar_info = extract_scalars_from_info(info)
                extra_texts = [
                    f"return: {self._eps_ret:.3f}",
                    f"reward: {rew:.3f}",
                    "action: {}".format(",".join([f"{x:.2f}" for x in action])),
                ]
                image = put_info_on_image(image, scalar_info, extras=extra_texts)
            self._render_images.append(image)

    def step(self, action):
        next_obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1
        self._eps_ret += rew

        self.save_data(self.env.unwrapped.get_env_state(), next_obs, action, rew, terminated or truncated, info)

        return next_obs, rew, terminated, truncated, info

    def flush_trajectory(self, verbose=False, ignore_empty_transition=False):
        if not self.save_trajectory or len(self._episode_data) == 0:
            return
        if ignore_empty_transition and len(self._episode_data) == 1:
            return

        traj_id = "traj_{}".format(self._episode_id)
        group = self._h5_file.create_group(traj_id, track_order=True)

        # Observations need special processing
        obs = [x["o"] for x in self._episode_data]
        if isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
            group.create_dataset("obs", data=obs, dtype=obs.dtype)
        else:
            raise NotImplementedError(type(obs[0]))

        if len(self._episode_data) == 1:
            action_space = self.env.action_space
            assert isinstance(action_space, spaces.Box), action_space
            actions = np.empty(
                shape=(0,) + action_space.shape,
                dtype=action_space.dtype,
            )
            dones = np.empty(shape=(0,), dtype=bool)
            rewards = np.empty(shape=(0,), dtype=np.float64)
        else:
            actions = np.stack([x["a"] for x in self._episode_data[1:]])
            dones = np.stack([bool(x["info"]["success"]) for x in self._episode_data[1:]])
            rewards = np.stack([x["r"] for x in self._episode_data[1:]])

        # Only support array like states now
        env_states_raw = [x["s"] for x in self._episode_data]
        if isinstance(env_states_raw[0], dict):
            env_state_parts = dict()
            for env_state_raw in env_states_raw:
                for k in env_state_raw.keys():
                    if k not in env_state_parts:
                        env_state_parts[k] = []
                    env_state_parts[k].append(env_state_raw[k])
            for k in env_states_raw[0].keys():
                env_state_parts[k] = np.stack(env_state_parts[k])
                group.create_dataset(
                    "env_states/" + k,
                    data=env_state_parts[k],
                    dtype=env_state_parts[k].dtype,
                    compression="gzip",
                    compression_opts=5,
                )
        else:
            group.create_dataset("env_states", data=env_states_raw, dtype=np.float32)
        # Dump
        group.create_dataset("actions", data=actions, dtype=np.float32)
        group.create_dataset("success", data=dones, dtype=bool)
        group.create_dataset("rewards", data=rewards, dtype=np.float32)

        # Handle JSON
        self._json_data["episodes"].append(self._episode_info)
        dump_json(self._json_path, self._json_data, indent=2)

        if verbose:
            print("Record the {}-th episode".format(self._episode_id))

    def flush_video(self, suffix="", verbose=False, ignore_empty_transition=False):
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return

        video_name = "{}{}".format(self.name_prefix, self._episode_id)
        if suffix:
            video_name += "_" + suffix
        images_to_video(
            self._render_images,
            str(self.output_dir),
            video_name=video_name,
            fps=20,
            verbose=verbose,
        )

    def close(self) -> None:
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset:
                traj_id = "traj_{}".format(self._episode_id)
                if traj_id in self._h5_file:
                    print(f"{traj_id} exists in h5.")
                else:
                    self.flush_trajectory(ignore_empty_transition=True)
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video(ignore_empty_transition=True)
        return super().close()
