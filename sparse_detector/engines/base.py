"""
DETR engines
"""
import sys
import math
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from sparse_detector.utils import distributed as dist_utils
from sparse_detector.datasets.coco_eval import CocoEvaluator
from sparse_detector.utils.logging import MetricLogger, SmoothedValue


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
    # Constructing the param dicts. For alpha_entmax, we don't want to decay alpha
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if ("backbone" not in n) and ("pre_alpha" in n) and p.requires_grad],
            "weight_decay": 0.0,
            "lr": lr * 10  # TODO: Fix this! It's only for debugging
        },
        {
            "params": [p for n, p in model.named_parameters() if ("backbone" not in n) and ("pre_alpha" not in n) and p.requires_grad],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)
    
    return optimizer, lr_scheduler


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int = None,
    max_norm: float = 0,
    global_step: Optional[int] = None,
    wandb_run: Optional[Any] = None,
    log_freq: Optional[int] = 50
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", wandb_run=wandb_run)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (samples, targets), g_step in metric_logger.log_every(data_loader, log_freq=log_freq, global_step=global_step, header=header, prefix="train", epoch=epoch):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats, g_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    postprocessors: Dict[str, Any],
    data_loader: Iterable,
    base_ds: Any,
    device: torch.device,
    epoch: int = None,
    wandb_run: Optional[Any] = None
) -> Any:
    """
    Returns:
        stats: CocoEval stats, 12 metrics available in CocoEval
        coco_evaluator: CocoEvaluator object
    """
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ", wandb_run=wandb_run)
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, log_freq=10, header=header, prefix="val", epoch=epoch):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator
