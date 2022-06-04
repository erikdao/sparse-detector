"""
This script visualizes the Hungarian Matcher results
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
import torchvision.transforms.functional as F

import click
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'
# import seaborn as sns
# sns.set()

from sparse_detector.configs import (
    build_dataset_config,
    build_detr_config,
    build_matcher_config,
    load_base_configs,
)
from sparse_detector.models import build_model
from sparse_detector.models.matcher import build_matcher
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.utils.logging import MetricLogger
import sparse_detector.datasets.transforms as T


@click.command()
@click.option("--seed", type=int, default=42)
@click.option("--detr-config-file", default="configs/decoder_sparsemax_baseline.yml", help="Path to config file")
@click.option("--resume-from-checkpoint", default="", help="resume from checkpoint")
@click.pass_context
def main(ctx, seed, detr_config_file, resume_from_checkpoint):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda")

    base_config = load_base_configs()
    detr_config = build_detr_config(detr_config_file, params=ctx.params, device=device)
    matcher_config = build_matcher_config(detr_config_file)
    dataset_config = build_dataset_config(base_config["dataset"], params=ctx.params)

    detr_config["average_cross_attn_weights"] = False
    print(detr_config)
    print("Matcher")
    print(matcher_config)

    print("Building model with configs")
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
        dataset_config["batch_size"],
        False,
        dataset_config["num_workers"],
    )

    metric_logger = MetricLogger(delimiter=" ")

    reverse_normalize = T.Normalize(
        [-0.485/0.229, -0.456/0.224, -0.406/0.225],
        [1/0.229, 1/0.224, 1/0.225]
    )

    for batch_id, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, log_freq=10, header=None, prefix="val")
    ):
        # fig, axs = plt.subplots(ncols=len(samples.tensors), squeeze=False)
        # rev_samples, _ = reverse_normalize(samples.tensors, targets)

        # print("Visualizing original images")
        # for i, img in enumerate(rev_samples):
        #     img = img.detach()
        #     img = F.to_pil_image(img)
        #     fig, ax = plt.subplots(figsize=(20, 20))
        #     plt.imshow(np.asarray(img))
        #     fig.savefig(f"temp/b-{batch_id}_img-{i}.png")

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        pred_logits = outputs['pred_logits'].detach()  # [B, num_queries, 92]
        pred_boxes = outputs['pred_boxes'].detach()  # [B, num_queries, 4]

        # `indices` is a list of length B (batch size), each item is a tuple
        # containing two tensors, the first tensor contains the indices of predictions
        # the second contains the indices of the corresponding matched ground-truth
        indices = matcher(outputs, targets)
        for i, (pred_indx, gt_indx) in enumerate(indices):
            pass

        import ipdb; ipdb.set_trace()
        break


if __name__ == "__main__":
    main()
