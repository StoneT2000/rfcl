import gymnasium
from gymnasium.wrappers import RecordVideo

def is_gymnasium_robotics_env(env_id: str):
    try:
        import gymnasium_robotics
        pass
    except ImportError:
        return False
    if env_id not in gymnasium.registry:
        return False
    return env_id == "PointMazeTest" or "gymnasium_robotics" in gymnasium.registry[env_id].entry_point


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[]):
    def _init():
        env = gymnasium.make(env_id, disable_env_checker=True, **env_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and idx == 0:
            env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True)
        return env
    return _init
