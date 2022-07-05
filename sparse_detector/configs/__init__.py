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
    if 'lr_alpha' in base_configs:
        base_configs['lr_alpha'] = float(base_configs['lr_alpha'])
    base_configs['device'] = device

    return base_configs


def build_matcher_config(config_file: Path) -> Dict[str, Any]:
    base_configs = load_config(config_file)
    base_configs = flatten_dict(base_configs)

    config = dict(
        set_cost_class=base_configs['set_cost_class'],
        set_cost_bbox=base_configs['set_cost_bbox'],
        set_cost_giou=base_configs['set_cost_giou']
    )
    return config


def build_trainer_config(base_configs: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    trainer_config = flatten_dict(base_configs)
    if params is not None:
        for k, v in params.items():
            if k not in trainer_config:
                continue
            trainer_config.update({k: v})
    
    # Manual surgery
    trainer_config['lr'] = float(trainer_config['lr'])
    trainer_config['weight_decay'] = float(trainer_config['weight_decay'])

    return trainer_config


def build_dataset_config(base_configs: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dataset_config = flatten_dict(base_configs)
    if params is not None:
        for k, v in params.items():
            if k not in dataset_config:
                continue
            dataset_config.update({k: v})
    
    return dataset_config
