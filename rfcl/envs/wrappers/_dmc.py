from dm_control import suite
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
import numpy as np
from dm_env import specs, TimeStep
from gymnasium import spaces
from dm_control.rl.control import Environment

# Some code copied over from https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int32(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype) 

def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCGymEnv(gym.Env):
    def __init__(self, env_name: str, obs_mode="state", render_mode="rgb_array", task_kwargs=dict(), environment_kwargs=dict(), visualize_reward=False, 
        width=84, height=84, frame_skip=1, camera_id=0,
        channels_first=True,
                 **kwargs):
        super().__init__()
        assert 'random' in task_kwargs, 'A seed must be specified via env_kwargs.task_kwargs.random. reset(seed=seed) will not change anything about the environment, it iwll only seed the action space'
        # env_name is parsed as dmc-<domain>-<task>
        assert env_name[:3] == "dmc", "Not a DM Control marked task"
        [_, env_domain, env_task] = env_name.split("-")
        assert (env_domain, env_task) in suite.ALL_TASKS, f"{env_name} is not a valid DM Control domain and task"
        self._env: Environment = suite.load(domain_name=env_domain, task_name=env_task, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs, visualize_reward=visualize_reward)
        self.spec = EnvSpec(id=env_name)
        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )
        self._frame_skip = frame_skip
        self.render_mode = render_mode
        assert obs_mode in ["state", "rgb"]
        self._height = height
        self._width = width
        self._channels_first = channels_first
        self._camera_id = camera_id
        self.obs_mode = obs_mode
        # create observation space
        if obs_mode == "rgb":
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(),
                np.float64
            )

        
    def _get_obs_from_timestep(self, timestep: TimeStep):
        if self.obs_mode == "rgb":
            return self._get_obs()
        else:
            return _flatten_obs(timestep.observation)
    def _get_obs(self):
        if self.obs_mode == "rgb":
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
            return obs
        else:
            return _flatten_obs(self._env._task.get_observation(self._env._physics))
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._norm_action_space
    
    def _convert_action(self, action: np.ndarray):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action
    
    def step(self, action: np.ndarray):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        # extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs()
        # extra['discount'] = time_step.discount
        terminated = False
        truncated = done # TODO check if correct
        info = {}
        info["success"] = False
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        time_step = self._env.reset()
        obs = self._get_obs_from_timestep(time_step)
        return obs, {}
    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_rgb_array()
        return None
    def render_rgb_array(self, height=None, width=None, camera_id=0):
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
    def get_env_state(self):
        return self._env._physics.get_state()
    def set_env_state(self, state):
        return self._env._physics.set_env_state(state)
class DMCInitialStateWrapper(gym.Wrapper):
    # note state methods are already handled in the above gym wrapper
    def get_env_obs(self):
        return self.env.unwrapped._get_obs()
