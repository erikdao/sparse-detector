"""
Configuration utilities
"""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

from sparse_detector.utils.misc import flatten_dict


def load_config(config_file: Path) -> Any:
    return yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)


def load_base_configs(file_path = None):
    if not file_path:
        cur_dir = Path(os.path.dirname(__file__))
        root_dir = cur_dir.parent.parent
        file_path = root_dir / "configs" / "base.yml"
    
    return load_config(file_path)


def build_detr_config(config_file: Path, params: Optional[Any] = None, device: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create configuration dict necessary to build DETR model.
    The base configuration is loaded from config_file. Then it will be updated with params if presented
    """
    base_configs = load_config(config_file)
    base_configs = flatten_dict(base_configs)
    if params is not None:
        for k, v in params.items():
            if k not in base_configs:
                continue
            base_configs.update({k: v})
    
    # Manual surgery
    base_configs['lr_backbone'] = float(base_configs['lr_backbone'])
    base_configs['device'] = device

    return base_configs
