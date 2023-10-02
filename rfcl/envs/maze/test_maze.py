from typing import Dict, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import Env

from .point_maze import PointMazeEnv


G = "g"
R = "r"
C = "c"
test_map_no_close_spawns = [
    [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    [2, 0, 0, 0, 0, 0, 1, R, R, R, R, 1],
    [2, 0, 2, 2, 2, 0, 2, R, 2, R, 1, 1],
    [2, 0, 0, 0, 2, 0, 2, R, 2, 1, 1, 1],
    [2, 2, 2, 0, 2, 0, 0, R, R, R, R, 1],
    [2, 0, 0, 0, 2, 2, 2, R, 2, 1, 1, 1],
    [2, 0, 2, 2, G, 2, R, R, R, R, R, 1],
    [2, 0, 2, 0, 0, 2, R, 2, 1, 1, R, 1],
    [2, 0, 0, 0, 2, R, R, 1, 1, R, R, 1],
    [2, 2, 2, 2, 2, R, 1, 1, 1, R, 1, 1],
    [1, R, R, R, R, R, R, R, 1, R, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
] # 34 spawn locs
test_map = [
    [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    [2, R, R, R, R, R, 1, R, R, R, R, 1],
    [2, R, 2, 2, 2, R, 2, R, 2, R, 1, 1],
    [2, R, R, R, 2, R, 2, R, 2, 1, 1, 1],
    [2, 2, 2, R, 2, R, R, R, R, R, R, 1],
    [2, R, R, R, 2, 2, 2, R, 2, 1, 1, 1],
    [2, R, 2, 2, C, 2, R, R, R, R, R, 1],
    [2, R, 2, R, R, 2, R, 2, 1, 1, R, 1],
    [2, R, R, R, 2, R, R, 1, 1, R, R, 1],
    [2, 2, 2, 2, 2, R, 1, 1, 1, R, 1, 1],
    [1, R, R, R, R, R, R, R, 1, R, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
] # 59 spawn locs


class PointMazeTestEnv(PointMazeEnv):
    def __init__(self, render_mode='rgb_array', remove_close_spawns=False, **kwargs):
        maze_map = test_map
        if remove_close_spawns:
            maze_map = test_map_no_close_spawns
        super().__init__(maze_map=maze_map, render_mode=render_mode, reward_type="sparse")
        self.last_seed = None

        
        # print(len(self.maze.unique_reset_locations))
        default_camera_config = {"distance": 22, "azimuth": 90, "elevation": -90}
        self.point_env.mujoco_renderer.default_cam_config = default_camera_config
        self._elapsed_steps = 0
        self.one_to_one_seed_to_initial_state = True
        self.remove_close_spawns = remove_close_spawns

        # add an additional bottleneck hole to make it hard to by chance get to goal.

    def get_env_state(self):
        return np.concatenate([self.point_env.data.qpos, self.point_env.data.qvel])
    def set_env_state(self, state):
        return self.point_env.set_state(state[:2], state[2:])
    def generate_reset_pos(self) -> np.ndarray:
        assert len(self.maze.unique_reset_locations) > 0
        
        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        if self.last_seed is not None and self.one_to_one_seed_to_initial_state:
            reset_index = self.last_seed % len(self.maze.unique_reset_locations)
        else:
            # reset_index = 7
            reset_index = self.np_random.integers(low=0, high=len(self.maze.unique_reset_locations))
            if reset_index <= 27:
            #     # close to goal!
            #     # lower probability of sampling by default. 
            #     # to mimic difficult problems where it is unlikely to sample "easy" start states.
            #     # point of forward curriculum is to reshape this distribution to sample easy start states more frequently
                if self.np_random.random() > 0.05:
                    reset_index = self.np_random.integers(low=28, high=len(self.maze.unique_reset_locations))
        reset_pos = self.maze.unique_reset_locations[reset_index].copy()
        return reset_pos
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        super().reset(seed=seed)
        
        self.last_seed = seed
        goal = self.generate_target_goal()
        self.goal = goal
        reset_pos = self.generate_reset_pos()
        

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        self.reset_pos = reset_pos

        # Update the position of the target site for visualization
        self.update_target_site_pos()
        self.point_env.init_qpos[:2] = self.reset_pos

        obs, info = self.point_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        self._elapsed_steps += 1
        info["eps_len"] = self._elapsed_steps
        return obs, rew, terminated, truncated, info
    def _get_obs(self, point_obs=None):
        if point_obs is None:
            point_obs, _ = self.point_env._get_obs()
        return super()._get_obs(point_obs)
class PointMazeObsWrapper(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,))
    def observation(self, observation):
        obs = observation["observation"]
        obs = np.concatenate([obs, observation["desired_goal"], [self.env.unwrapped.collided_with_wall_once]])
        return obs

gym.register("PointMazeTest-v0", lambda *args, **kwargs : PointMazeObsWrapper(PointMazeTestEnv(*args, **kwargs)))