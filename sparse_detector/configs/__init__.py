"""
Configuration utilities
"""
import os
import yaml
from typing import Any
from pathlib import Path


def load_config(config_file: Path) -> Any:
    return yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)


def load_base_configs(file_path = None):
    if not file_path:
        cur_dir = Path(os.path.dirname(__file__))
        root_dir = cur_dir.parent.parent
        file_path = root_dir / "configs" / "base.yml"
    
    return load_config(file_path)
