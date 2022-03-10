"""
Configuration utilities
"""
import yaml
from typing import Any
from pathlib import Path


def load_config(config_file: Path) -> Any:
    return yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
