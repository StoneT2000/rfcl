import gymnasium as gym


try:
    import mani_skill.envs  # NOQA
    from mani_skill.utils.wrappers import RecordEpisode as RecordEpisodeWrapper
    from mani_skill.utils.wrappers.gymnasium import ManiSkillCPUGymWrapper
except ImportError:
    pass


def is_mani_skill3_env(env_id: str):
    try:
        import mani_skill.envs  # NOQA
    except ImportError:
        return False
    from mani_skill.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    def _init():
        env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        env = ManiSkillCPUGymWrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        
        if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
            env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                info_on_video=False,#record_episode_kwargs["info_on_video"],
            )
        return env

    return _init
