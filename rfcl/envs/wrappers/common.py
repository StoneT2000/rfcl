import gymnasium as gym


class SparseRewardWrapper(gym.Wrapper):
    def step(self, action):
        o, _, terminated, truncated, info = self.env.step(action)
        return o, int(info["success"]), terminated, truncated, info


class ContinuousTaskWrapper(gym.Wrapper):
    """
    Makes a task continuous by disabling any early terminations, allowing episode to only end
    when truncated=True (timelimit reached)
    """

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        return observation, reward, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Adds additional info. Anything that goes in the stats wrapper is logged to tensorboard/wandb under train_stats and test_stats
    """

    def reset(self, *, seed=None, options=None):
        self.eps_seed = seed
        obs, info = super().reset(seed=seed, options=options)
        self.eps_ret = 0
        self.eps_len = 0
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.eps_ret += reward
        self.eps_len += 1
        info["eps_ret"] = self.eps_ret
        info["eps_len"] = self.eps_len
        info["seed"] = self.eps_seed
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            success_at_end=int(
                info["success"]
            ),  # this is the success rate used for comparing algorithm performances, which is more difficult but more realistic
            success_once=self.success_once,
        )
        return observation, reward, terminated, truncated, info
