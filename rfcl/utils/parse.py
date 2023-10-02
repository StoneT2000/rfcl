"""
parser code for parsing configuration files
"""
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def parse_cfg(cfg_path: str = None, default_cfg_path: str = None) -> OmegaConf:
    """Parses a config file and returns an OmegaConf object. Priority is CLI configs,
    then provided config, then the default config if it exists"""
    if default_cfg_path is not None:
        base = OmegaConf.load(default_cfg_path)

        if "base_config" in base:
            old_cfg = None
            base_config = base.base_config
            if isinstance(base_config, str):
                base_config = [base_config]
            if not isinstance(base_config, list):
                raise ValueError("base_config must be a string or list of strings")
            for path in base_config:
                new_path = Path(default_cfg_path).parent / Path(path)
                new_cfg = parse_cfg(default_cfg_path=new_path)
                if old_cfg is not None:
                    new_cfg = OmegaConf.merge(old_cfg, new_cfg)
                old_cfg = new_cfg

            base = OmegaConf.merge(old_cfg, base)

    else:
        base = OmegaConf.create()

    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
        base.merge_with(cfg)

    cli = OmegaConf.from_cli()
    for k, v in cli.items():
        if v is None:
            cli[k] = True
    base.merge_with(cli)
    return base
