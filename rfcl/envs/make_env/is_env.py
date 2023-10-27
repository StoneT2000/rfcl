"""
Dependency free python code to check if an environment ID is from a particular benchmark suite or not without erroring out about import errors
"""

def is_dmc_env(env_id: str):
    try:
        from dm_control import suite
    except ImportError:
        return False
    # expect env id to be of format "dmc-<domain>-<task>"
    [part, env_domain, env_task] = env_id.split("-")
    return part == "dmc" and (env_domain, env_task) in suite.ALL_TASKS
def is_gymnasium_robotics_env(env_id: str):
    try:
        import gymnasium_robotics
        import gymnasium
        pass
    except ImportError:
        return False
    if env_id not in gymnasium.registry:
        return False
    return env_id == "PointMazeTest" or "gymnasium_robotics" in gymnasium.registry[env_id].entry_point
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