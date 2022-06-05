"""
Compute the Percentage of Attentions inside Bounding Boxes metrics

Run:
python -m scripts.compute_paibb \
    --detr-config-file configs/decoder_sparsemax_baseline.yml \
    --resume-from-checkpoint checkpoints/v2_decoder_sparsemax/checkpoint.pth \
    --decoder-act sparsemax
"""
import os
import sys
import warnings
import itertools
import random

warnings.filterwarnings("ignore")
from pathlib import Path

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
import torch.nn.functional as F

import click
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = "tight"
import matplotlib.patches as patches

import seaborn as sns

sns.set(font_scale=1.4)
sns.set_style(
    "white",
    {
        "axes.edgecolor": "#475569",
        "font.family": ["sans-serif"],
        "font.sans-serif": ["Arial", "Droid Sans", "sans-serif"],
    },
)

from sparse_detector.configs import (
    build_dataset_config,
    build_detr_config,
    build_matcher_config,
    load_base_configs,
    load_config,
)
from sparse_detector.models import build_model
from sparse_detector.models.matcher import build_matcher
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.utils.logging import MetricLogger
import sparse_detector.datasets.transforms as T
from sparse_detector.utils.box_ops import box_cxcywh_to_xyxy
from sparse_detector.visualizations import ColorPalette
from sparse_detector.datasets.coco import CLASSES


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def draw_box_on_ax(box, ax, text=None, color=None):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    color = color or ColorPalette.GREEN
    rect = patches.Rectangle((xmin, ymin), width, height,
                    fill=False, color=color, linewidth=2, zorder=1000, axes=ax)
    ax.add_artist(rect)

    if text:
        ax.text(xmin, ymin, text, fontsize=14, bbox=dict(facecolor=ColorPalette.GREEN, alpha=0.5))


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("--detr-config-file", default="", help="Path to config file")
@click.option("--decoder-act", type=str, default="sparsemax")
@click.option("--batch-size", default=6, type=int, help="Batch size per GPU")
@click.option("--resume-from-checkpoint", default="", help="resume from checkpoint")
@click.option(
    "--detection-threshold",
    default=None,
    type=float,
    help="Threshold to filter detection results",
)
@click.option("--pre-norm/--no-pre-norm", default=True)
@click.pass_context
def main(
    ctx,
    seed,
    detr_config_file,
    decoder_act,
    batch_size,
    resume_from_checkpoint,
    detection_threshold,
    pre_norm,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda")
    base_config = load_base_configs()
    detr_config = build_detr_config(detr_config_file, params=None, device=device)
    detr_config["average_cross_attn_weights"] = False

    matcher_config = build_matcher_config(detr_config_file)
    dataset_config = build_dataset_config(base_config["dataset"], params=ctx.params)

    # Build model
    print("Build model")
    model, criterion, _ = build_model(**detr_config)
    if resume_from_checkpoint:
        print(f"Load model from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    describe_model(model)
    criterion.eval()

    print("Building matcher")
    matcher = build_matcher(
        set_cost_class=matcher_config["set_cost_class"],
        set_cost_bbox=matcher_config["set_cost_bbox"],
        set_cost_giou=matcher_config["set_cost_giou"],
    )

    print("Building dataset")
    data_loader, _ = build_dataloaders(
        "val",
        dataset_config["coco_path"],
        1, # dataset_config["batch_size"],
        False,
        dataset_config["num_workers"],
    )

    metric_logger = MetricLogger(delimiter=" ")

    reverse_normalize = T.Normalize(
        [-0.485/0.229, -0.456/0.224, -0.406/0.225],
        [1/0.229, 1/0.224, 1/0.225]
    )

    batch_images, batch_targets = next(itertools.islice(
        metric_logger.log_every(data_loader, log_freq=10, header=None, prefix="val"), 5, None
    ))
    batch_images = batch_images.to(device)
    batch_targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

    print("Getting predictions")
    conv_features, dec_attn_weights = [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    outputs = model(batch_images.to(device))

    for hook in hooks:
        hook.remove()

    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_boxes = outputs['pred_boxes'].detach().cpu()
    print(f"pred_logits={pred_logits.shape};\t pred_boxes={pred_boxes.shape}")

    conv_features = conv_features[0]
    h, w = conv_features['3'].tensors.shape[-2:]
    attentions = dec_attn_weights[0].detach()  # [B, nh, num_queries, K]
    B, num_heads, num_queries, K = attentions.size()

    print("Matching predictions and groundtruth")
    indices = matcher(outputs, batch_targets)

    # For each image in the batch:
    for i, (pred_indices, gt_indices) in enumerate(indices):
        print(f"i={i}, pred_indices={pred_indices}\t gt_indices={gt_indices}")
        img_attns = attentions[i]  # [num_heads, num_queries, K]
        targets = batch_targets[i]
        img_h, img_w = targets['orig_size']
        img_name = str(targets['image_id'].item()).rjust(12, '0')

        for (pred_idx, gt_idx) in zip(pred_indices, gt_indices):
            attn_maps = img_attns[:, pred_idx, :] # [num_heads, K]
            attn_maps = attn_maps.view(attn_maps.size(0), h, w)  # [num_heads, h, w]
            rescale_maps = F.interpolate(attn_maps.unsqueeze(0), (img_h, img_w), mode='bilinear') # [1, num_heads, img_h, img_w]
            matched_gt_bboxes = targets['boxes'][gt_idx] # [4]
            print(attn_maps.shape, rescale_maps.shape)

            for h_i, attn_map in enumerate(rescale_maps.squeeze(0)):
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(attn_map.detach().cpu(), cmap='viridis')
                fig.savefig(f"temp/{img_name}_head-{h_i}_query-{pred_idx}.png", bbox_inches='tight')
            

if __name__ == "__main__":
    main()
