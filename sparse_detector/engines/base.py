"""
DETR engines
"""
from typing import Any, Optional

import torch.nn as nn
import torch.optim as optim


def build_detr_optims(
    model: nn.Module,
    lr: Optional[float] = None,
    lr_backbone: Optional[float] = None,
    lr_drop: Optional[int] = None,
    weight_decay: Optional[float] = None,
) -> Any:
    """Build optimizer and learning rate scheduler for DETR.
    DETR uses different learning rates for the backbone and the transformer detector
    
    Args:
        model: DETR model
        lr: Learning rate of the transformer detector
        lr_backbone: Learning rate of the backbone
        lr_drop: Number of epochs after which learning rates are dropped
        weight_decay: Optimizer's weight decay
    """
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    
    return optimizer, lr_scheduler
