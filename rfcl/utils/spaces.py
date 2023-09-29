"""
Get observation/action spaces shapes.
"""

from typing import Dict, Tuple, Union

import numpy as np
from gymnasium import spaces
from gymnax.environments import spaces as jax_spaces


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).
    :param observation_space:
    :return:
    """
    if isinstance(observation_space, int):
        return observation_space
    if isinstance(observation_space, spaces.Box) or isinstance(observation_space, jax_spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete) or isinstance(observation_space, jax_spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # TODO add gymnax space later
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Dict) or isinstance(observation_space, jax_spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}


# TODO, handle dicts somehow. why would a env want action dictionaries? because theyre complex idk
def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    :param action_space:
    :return:
    """
    if isinstance(action_space, int):
        return action_space
    elif isinstance(action_space, spaces.Box) or isinstance(action_space, jax_spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete) or isinstance(action_space, jax_spaces.Discrete):
        # Action is an int
        return action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def is_discrete_action_space(action_space: spaces.Space) -> bool:
    if isinstance(action_space, spaces.Discrete):
        return True
    # TODO handle our env types
    return False
