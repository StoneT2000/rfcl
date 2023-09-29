import os.path as osp
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
from omegaconf import OmegaConf

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)
import wandb as wb


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


from dataclasses import dataclass


@dataclass
class LoggerConfig:
    workspace: str
    exp_name: Union[str, None] = None
    clear_out: bool = False
    project_name: Union[str, None] = None
    tensorboard: bool = False
    wandb: bool = False
    wandb_cfg: Union[Dict, None] = None
    cfg: Dict = None
    best_stats_cfg: Dict = None
    save_fn: Union[Callable, None] = None


class Logger:
    """
    Logging tool
    """

    def __init__(
        self,
        workspace: str = "default_workspace",
        exp_name: str = "default_exp",
        clear_out: bool = False,
        project_name: str = None,
        tensorboard=True,
        wandb=False,
        wandb_cfg: Union[Dict, None] = None,
        cfg: Union[Dict, OmegaConf, None] = {},
        best_stats_cfg: Union[Dict, None] = {},
        save_fn: Callable = None,
    ) -> None:
        """
        A logger for logging data points as well as summary statistics.

        Stores logs in <workspace>/<exp_name>/logs

        Checkpoints (model weights, training states) usually stored in <workspace>/<exp_name>/models

        Parameters
        ----------
        wandb : bool
            Whether to use Weights and Biases and log to there

        tensorboard : bool
            Whether to log locally to tensorboard

        workspace : str
            A workspace to store all experiments in a group together to

        exp_name : str
            Name of this particular experiment we are logging for

        project_name : str
            Used by wandb only. Defines the project name

        clear_out : bool
            If true, clears out all previous data for this experiment. Otherwise will use the same folders

        best_stats_cfg : dict
            maps stat name to 1 -> higher is better, -1 -> lower is better
            If set, will record the best result for the stat

        save_fn : Callable
            function that saves some relevant models/state. Called whenever a stat is improved based on best_stats_cfg

        cfg : Dict | OmegaConf
            A dict or OmegaConf object containing all configuration details for this experiment

            If wandb_id is given, it will try and continue that wandb experiment.
        """
        self.wandb = wandb
        if wandb_cfg is None:
            wandb_cfg = {}
        if cfg is None:
            cfg = {}
        if best_stats_cfg is None:
            best_stats_cfg = {}
        self.tensorboard = tensorboard
        self.tb_writer = None
        self.wandb_run = None

        self.start_step = 0
        self.last_log_step = 0

        self.exp_path = osp.join(workspace, exp_name)
        self.model_path = osp.join(self.exp_path, "models")
        self.video_path = osp.join(self.exp_path, "videos")
        self.log_path = osp.join(self.exp_path, "logs")
        if clear_out:
            if osp.exists(self.exp_path):
                shutil.rmtree(self.exp_path, ignore_errors=True)

        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        Path(self.video_path).mkdir(parents=True, exist_ok=True)
        # set up external loggers

        if self.wandb:
            if project_name is None:
                project_name = workspace
            if "wandb_id" in cfg:
                wandb_id = cfg["wandb_id"]
                wb.init(
                    project=project_name,
                    name=exp_name,
                    id=wandb_id,
                    resume="allow",
                    **wandb_cfg,
                )
            else:
                wandb_id = wb.util.generate_id()
                self.wandb_run = wb.init(project=project_name, name=exp_name, id=wandb_id, **wandb_cfg)
                cfg["wandb_id"] = wandb_id
        self.save_config(cfg)

        self.data = defaultdict(dict)
        self.data_log_summary = defaultdict(dict)
        self.stats = {}
        self.best_stats = {}
        self.best_stats_cfg = best_stats_cfg
        self.save_fn = save_fn

    @classmethod
    def create_from_cfg(cls, cfg: LoggerConfig):
        return cls(
            workspace=cfg.workspace,
            exp_name=cfg.exp_name,
            clear_out=cfg.clear_out,
            project_name=cfg.project_name,
            tensorboard=cfg.tensorboard,
            wandb=cfg.wandb,
            wandb_cfg=cfg.wandb_cfg,
            cfg=cfg.cfg,
            best_stats_cfg=cfg.best_stats_cfg,
            save_fn=cfg.save_fn,
        )

    def init_tb(self):
        if self.tensorboard and self.tb_writer is None:
            from tensorboardX import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=self.log_path)

    def close(self):
        """
        finishes up experiment logging

        in wandb, finishes the experiment and uploads remaining data
        """
        if self.tensorboard:
            self.tb_writer.close()
        if self.wandb:
            wb.finish()

    def save_config(self, config: Union[Dict, OmegaConf], verbose=2):
        """
        save configuration of experiments to the experiment directory
        """
        if type(config) == type(OmegaConf.create()):
            config = OmegaConf.to_container(config)
        if self.wandb:
            wb.config.update(config, allow_val_change=True)
        config_path = osp.join(self.exp_path, "config.yml")
        # config_json = convert_json(config)
        # output = json.dumps(config_json, indent=2, sort_keys=True)
        if verbose > 1:
            self.print("Saving config:\n", color="cyan", bold=True)
            self.print(config)
        with open(config_path, "w") as out:
            out.write(OmegaConf.to_yaml(config))

    def print(self, msg, file=sys.stdout, color="", bold=False):
        """
        print to terminal, stdout by default. Ensures only the main process ever prints.
        """
        if color == "":
            print(msg, file=file)
        else:
            print(colorize(msg, color, bold=bold), file=file)
        sys.stdout.flush()

    def store(self, tag="default", log_summary=False, **kwargs):
        """
        Stores scalar values or arrays into logger by tag and key to then be logged

        if log_summary is True, logs std, min, and max
        """
        for k, v in kwargs.items():
            self.data[tag][k] = v
            self.data_log_summary[tag][k] = log_summary

    def get_data(self, tag=None):
        if tag is None:
            data_dict = {}
            for tag in self.data.keys():
                for k, v in self.data[tag].items():
                    data_dict[f"{tag}/{k}"] = v
            return data_dict
        return self.data[tag]

    def pretty_print_table(self, data):
        # Code from spinning up
        vals = []
        key_lens = [len(key) for key in data.keys()]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in sorted(data.keys()):
            val = data[key]
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)
        print("-" * n_slashes, flush=True)

    def log(self, step, local_only=False):
        """
        log accumulated data to tensorboard if enabled and to the terminal and locally.

        Statistics are then retrievable as a dict via get_data

        """
        if step < self.last_log_step:
            warnings.warn(
                f"logged at step {step} but previously logged at step {self.last_log_step}",
                RuntimeWarning,
            )
        self.last_log_step = step

        self.init_tb()
        for tag in self.data.keys():
            data_dict = self.data[tag]
            for k, v in data_dict.items():
                key_vals = dict()
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    if len(v) > 0:
                        vals = np.array(v)
                        vals_sum, n = vals.sum(), len(vals)
                        avg = vals_sum / n
                        key_vals = {f"{tag}/{k}_avg": avg}
                        if self.data_log_summary[tag][k]:
                            sum_sq = np.sum((vals - avg) ** 2)
                            std = np.sqrt(sum_sq / n)
                            minv = np.min(vals)
                            maxv = np.max(vals)
                            key_vals = {
                                **key_vals,
                                f"{tag}/{k}_std": std,
                                f"{tag}/{k}_min": minv,
                                f"{tag}/{k}_max": maxv,
                            }
                else:
                    key_vals = {f"{tag}/{k}": v}
                for name, scalar in key_vals.items():
                    if name in self.best_stats_cfg:
                        sort_order = self.best_stats_cfg[name]
                        update_val = False
                        if name not in self.best_stats:
                            update_val = True
                        else:
                            prev_val = self.best_stats[name]["val"]
                            if (sort_order == 1 and prev_val < scalar) or (sort_order == -1 and prev_val > scalar):
                                update_val = True
                        if update_val:
                            self.best_stats[name] = dict(val=scalar, step=step)
                            fmt_name = name.replace("/", "_")
                            self.save_fn(osp.join(self.model_path, f"best_{fmt_name}_ckpt.jx"))
                            print(f"{name} new best at {step}: {scalar}")
                    if self.tensorboard and not local_only:
                        self.tb_writer.add_scalar(name, scalar, self.start_step + step)
                    self.stats[name] = scalar
                if self.wandb and not local_only:
                    self.wandb_run.log(data=key_vals, step=self.start_step + step)

        return self.stats

    def reset(self):
        """
        call this each time after log is called
        """
        self.data = defaultdict(dict)
        self.stats = {}

    def state_dict(self):
        return dict(best_stats=self.best_stats, last_log_step=self.last_log_step)

    def load(self, data):
        self.best_stats = data["best_stats"]
        self.last_log_step = data["last_log_step"]
        return self
