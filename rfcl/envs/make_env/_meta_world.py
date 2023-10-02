from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import RecordVideo, TimeLimit


def is_meta_world_env(env_id: str):
    try:
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        )

        import metaworld
    except ImportError:
        return False
    return env_id in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN or env_id in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


def get_env_factory():
    def meta_world_env_factory(env_id, idx, seed, record_video_path, env_kwargs, wrappers=[]):
        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        )

        import metaworld
        from rfcl.envs.wrappers._meta_world import MetaWorldEnv

        def _init():
            env_list = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
            if env_id in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN:
                env_list = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

            env = MetaWorldEnv(env_id)
            env.spec = EnvSpec(id=env_id, max_episode_steps=200)
            env = TimeLimit(env, max_episode_steps=200)

            for wrapper in wrappers:
                env = wrapper(env)
            if record_video_path is not None and idx == 0:
                env = RecordVideo(env, record_video_path, episode_trigger=lambda x: True, disable_logger=True)
            return env

        return _init

    return meta_world_env_factory
