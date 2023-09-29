"""
Code adapted from ManiSkill2
"""
import gzip
import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np


class CustomJsonEncoder(json.JSONEncoder):
    """Custom json encoder to support more types, like numpy and Path."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def dump_json(filename: Union[str, Path], obj, **kwargs):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "wt")
    elif filename.endswith(".json"):
        f = open(filename, "wt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    json.dump(obj, f, cls=CustomJsonEncoder, **kwargs)
    f.close()


def write_txt(filename: Union[str, Path], content: Union[str, Sequence[str]]):
    with open(filename, "w") as f:
        if not isinstance(content, str):
            content = "\n".join(content)
        f.write(content)

import h5py

def merge_h5(output_path: str, traj_paths, recompute_id=True):
    print("Merge to", output_path)

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"env_info": {}, "episodes": []}
    _env_info = None
    cnt = 0

    for traj_path in traj_paths:
        traj_path = str(traj_path)
        print("Merging", traj_path)

        h5_file = h5py.File(traj_path, "r")
        json_path = traj_path.replace(".h5", ".json")
        json_data = load_json(json_path)

        # Check env info
        env_info = json_data["env_info"]
        if _env_info is None:
            _env_info = env_info
            merged_json_data["env_info"] = _env_info
        else:
            assert str(env_info) == str(_env_info), traj_path

        # Merge
        for ep in json_data["episodes"]:
            episode_id = ep["episode_id"]
            traj_id = f"traj_{episode_id}"

            # Copy h5 data
            if recompute_id:
                new_traj_id = f"traj_{cnt}"
            else:
                new_traj_id = traj_id

            assert new_traj_id not in merged_h5_file, new_traj_id
            h5_file.copy(traj_id, merged_h5_file, new_traj_id)

            # Copy json data
            if recompute_id:
                ep["episode_id"] = cnt
            merged_json_data["episodes"].append(ep)

            cnt += 1

        h5_file.close()

    # Ignore commit info
    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)
