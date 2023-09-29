import gymnasium as gym

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from shimmy.openai_gym_compatibility import _convert_space
from rfcl.envs.wrappers.curriculum import InitialStateWrapper

class MetaWorldEnv(gym.Env):
    """
    Adapted from https://github.com/facebookresearch/modem/blob/main/tasks/metaworld.py

    Adds additional functions to extract and set env states
    """
    def __init__(self, env_name, render_mode='rgb_array', **kwargs):
        super().__init__()
        env_list = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        if env_name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
            env_list = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
        self.gym_env = env_list[env_name]()
        self.observation_space = _convert_space(self.gym_env.observation_space)
        self.action_space = _convert_space(self.gym_env.action_space)
        self.camera_name = "corner2"
        self.gym_env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.render_mode=render_mode
        self.reward_range = getattr(self.gym_env, "reward_range", None)
        self.spec = getattr(self.gym_env, "spec", None)
        self.gym_env._freeze_rand_vec = False # Turn false in order to ensure resets to new states
        self.gym_env.seeded_rand_vec = True # avoid using pure np random generator.
    def _get_obs(self):
        return self.gym_env._get_obs()
    def step(self, action):
        obs, r, d, info = self.gym_env.step(action)
        info["success"] = bool(info["success"])
        return obs, float(info["success"]), False, False, info
    def get_env_state(self):
        _last_rand_vec = self.gym_env._last_rand_vec
        joint_state, mocap_state = self.gym_env.get_env_state()
        mocap_pos, mocap_quat = mocap_state
        return dict(_last_rand_vec=_last_rand_vec, joint_state=joint_state.flatten(), mocap_pos=mocap_pos, mocap_quat=mocap_quat)
    def set_env_state(self, state):
        self.gym_env._freeze_rand_vec = True
        joint_state, _ = self.gym_env.get_env_state()
        joint_state = joint_state.from_flattened(state["joint_state"], self.gym_env.sim)
        mocap_state = state["mocap_pos"], state["mocap_quat"]
        self.gym_env._last_rand_vec = state["_last_rand_vec"]
        self.gym_env.reset()
        self.gym_env._freeze_rand_vec = False
        return self.gym_env.set_env_state((joint_state, mocap_state))
    def reset(self, *, seed = None, options = None):
        if seed is not None:
            self.gym_env.seed(seed)
        if isinstance(options, dict) and 'initial_state_dict' in options:
            self.set_env_state(options['initial_state_dict'])
        obs = self.gym_env.reset()
        obs, _ ,_ , _ = self.gym_env.step(self.gym_env.action_space.sample()*0)
        return obs, {}
    def render(self, mode="rgb_array", width=512, height=512, camera_id=None):
        return self.gym_env.render(
            offscreen=True, resolution=(width, height), camera_name=self.camera_name
        ).copy()

class MetaWorldInitialStateWrapper(InitialStateWrapper):
    def set_env_state(self, state):
        return self.env.set_env_state(state)
    def get_env_obs(self):
        return self.env.unwrapped._get_obs()