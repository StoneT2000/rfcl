import jax
import numpy as np
from chex import Array


def any_to_np(x: Array):
    return np.array(x)


def is_jax_arr(x: Array):
    return isinstance(x, jax.numpy.ndarray)


def copy_arr(x: Array):
    return x.copy()


def combine(one_dict, other_dict):
    # Code from https://github.com/ikostrikov/rlpd/
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty((v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype)
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def reached_freq(t, freq, step_size=1):
    """
    Returns True if `freq > 0` and time `t` has reached the frequency. Gives a leeway of size `step_size - 1`. This assumes `t` only ever increments
    by size `step_size` and allows any `t` within `step_size` of `freq` to return True.

    Returns False otherwise

    `step_size=1` is equivalent to checking if `t % freq == 0`.

    For e.g. `step_size=256, freq=1000`,
    - At `t=0`, return True
    - At `t=256, 512, 768` return False.
    - At `t=1024` return True.

    """
    if freq > 0 and (t - step_size) // freq < t // freq:
        return True
    return False


def flatten_struct_to_dict(tree):
    """
    Takes any PyTree and flattens it to return a Python dictionary with the same values. Flexibly handle keys that are strings or attributes
    """
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)
    out_dict = dict()
    for key_path, value in flattened:
        key_path_str = []
        for part in key_path:
            if isinstance(part, jax.tree_util.DictKey):
                key_path_str.append(part.key)
            else:
                # if not DictKey, then this is of type jax.tree_util.GetAttrKey instead
                key_path_str.append(part.name)
        key_path_str = "/".join(key_path_str)
        out_dict[key_path_str] = value
    return out_dict
