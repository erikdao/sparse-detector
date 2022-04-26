"""
Configuration utilities
"""
import os
import yaml
from typing import Any, Dict
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


def load_detr_base_configs(file_path = None):
    if not file_path:
        cur_dir = Path(os.path.dirname(__file__))
        root_dir = cur_dir.parent.parent
        file_path = root_dir / "configs" / "detr_baseline.yml"
    
    return load_config(file_path)


def build_detr_config(base_file: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """
    Convert configs loaded from YAML files into format that could be used to create model instance,
    such as those passed to the `build_model` function.

    Args:
    - base_config: a dictionary that has keys including `model`, `loss`, `datasets`, `trainer`
    """
    base_configs = load_detr_base_configs(base_file)
    model_configs = {**base_configs['model'], **base_configs['loss']}
    flat_configs = flatten_dict(model_configs)

    # Remove the `lr` key as it's not needed when building the model
    flat_configs.pop('lr')
    flat_configs['lr_backbone'] = float(flat_configs['lr_backbone'])
    configs = {**flat_configs, **kwargs}

    return configs
