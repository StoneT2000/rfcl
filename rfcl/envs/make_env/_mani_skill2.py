import gymnasium as gym


try:
    import mani_skill2.envs  # NOQA
    from mani_skill2.utils.wrappers import RecordEpisode
except ImportError:
    pass


def is_mani_skill2_env(env_id: str):
    try:
        import mani_skill2.envs  # NOQA
    except ImportError:
        return False
    from mani_skill2.utils.registration import REGISTERED_ENVS

    return env_id in REGISTERED_ENVS


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[]):
    def _init():
        env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordEpisode(env, record_video_path, info_on_video=True)
        return env
    return _init
