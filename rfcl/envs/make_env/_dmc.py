def is_dmc_env(env_id: str):
    try:
        from dm_control import suite
    except ImportError:
        return False
    # expect env id to be of format "dmc-<domain>-<task>"
    [part, env_domain, env_task] = env_id.split("-")
    return part == "dmc" and (env_domain, env_task) in suite.ALL_TASKS