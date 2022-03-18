"""
Run detection model with input image(s) and get the results
"""
import os
import sys
from typing import Any, Tuple

import torch
import torchvision.transforms as pth_transform

import click
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.coco import NORMALIZATION
from sparse_detector.datasets import transforms as T

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("temp/prediction.png", bbox_inches="tight")


@click.command()
@click.argument('image_path', type=click.File('rb'))
@click.argument('checkpoint_path', type=click.File('rb'))
@click.option('--iou-threshold', type=float, default=0.5)
def main(image_path, checkpoint_path, iou_threshold):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    click.echo("Reading input image")
    image = Image.open(image_path)
    orig_w, orig_h = image.size

    # The customer RandomResize outputs both the image and target even if
    # the target is None, so we perform the transform separately and only
    # feed the image to the next standard torch transform pipeline
    resize_transform = T.RandomResize([800], max_size=1333)
    resize_image, _ = resize_transform(image)

    transforms = pth_transform.Compose([
        pth_transform.ToTensor(),
        pth_transform.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
    ])

    input_tensor = transforms(resize_image).unsqueeze(0).to(device)
    click.echo(f"input_tensor: {input_tensor.shape}")

    click.echo("Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    click.echo("Building model")
    model, criterion, postprocessors = build_model(
        "resnet50", 1e-5, False, True, "sine", 256, 6, 6,
        dim_feedforward=2048, dropout=0.1, num_queries=100,
        bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, aux_loss=False,
        set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
        nheads=8, pre_norm=True, dataset_file='coco', device=device
    )

    click.echo("Load model from checkpoint")
    model.eval()
    model.to(device)

    model.load_state_dict(checkpoint['model'])
    describe_model(model)

    outputs = model(input_tensor)
    # keep only predictions with 0.7+ confidence
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_boxes = outputs['pred_boxes'].detach().cpu()

    probas = pred_logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], image.size)
    plot_results(image, probas[keep], bboxes_scaled)


if __name__ == "__main__":
    main()
