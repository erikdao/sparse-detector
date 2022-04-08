"""
Visualizing self-attention of the model
"""
import math
import os
import sys

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
import torchvision.transforms as pth_transform

import click
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.coco import NORMALIZATION, CLASSES
from sparse_detector.datasets import transforms as T


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


def rescale_bboxes_for_ax(bbox, img_size, canvas_size):
    imw, imh = img_size
    cw, ch = canvas_size
    w_scale, h_scale = imw * 1.0 / cw, imh * 1.0 / ch
    xmin, ymin, xmax, ymax = bbox
    b = (xmin * 1.0 / w_scale, ymin * 1.0 / h_scale, xmax * 1.0 / w_scale, ymax * 1.0 / h_scale)
    return b
    

@click.command()
@click.argument('image_path', type=click.File('rb'))
@click.argument('checkpoint_path', type=click.File('rb'))
@click.option('--seed', type=int, default=42)
def main(image_path, checkpoint_path, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    click.echo("Reading input image")
    image = Image.open(image_path)
    image_id = str(image_path.name).split("/")[-1].split(".")[0]

    # The customer RandomResize outputs both the image and target even if
    # the target is None, so we perform the transform separately and only
    # feed the image to the next standard torch transform pipeline
    resize_transform = T.RandomResize([800])
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
        nheads=8, pre_norm=True, dataset_file='coco', device=device,
        decoder_act='sparsemax'
    )

    click.echo("Load model from checkpoint")
    model.eval()
    model.to(device)

    model.load_state_dict(checkpoint['model'])
    describe_model(model)

    # propagate through the model
    outputs = model(input_tensor)
    # keep only predictions with 0.7+ confidence
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_boxes = outputs['pred_boxes'].detach().cpu()

    probas = pred_logits.softmax(-1)[0, :, :-1]
    print(f"probas: {probas.shape}")
    keep = probas.max(-1).values > 0.7
    print(f"keep: {keep.shape}")

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], image.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(input_tensor)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    # Here we get the feature from the last block in the last layer of ResNet backbone
    h, w = conv_features['3'].tensors.shape[-2:]

    queries = keep.nonzero()
    items = []
    for idx, bbox in zip(queries, bboxes_scaled):
        items.append((dec_attn_weights[0, idx].view(h, w).detach().cpu().numpy(), idx))
        items.append(bbox)
    
    ncols = 4
    if len(items) > 12:
        ncols = 6
    nrows = len(items) // ncols
    if (len(items) % ncols) != 0:
        nrows = len(items) // ncols + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))

    for idx, (ax, item) in enumerate(zip(axes.flat, items)):
        if idx % 2 == 0:
            attn, qid = item
            ax.imshow(attn)
            bbox = items[idx+1]
            ax.set_title(f"query {qid.item()}")
        else:
            qid = items[idx-1][1]
            ax.imshow(image)
            bbox = item
            ax.set_title(CLASSES[probas[qid].argmax()])
        ax_h, ax_w = ax.bbox.height, ax.bbox.width
        print(f"ax_h={ax_h}; ax_w={ax_w}\t image size: {image.size}")

        if idx % 2 == 0:
            (xl_min, xl_max) = ax.get_xlim()
            (yl_max, yl_min) = ax.get_ylim()
            w = abs(xl_min) + abs(xl_max)
            h = abs(yl_min) + abs(yl_max)
            bbox = rescale_bboxes_for_ax(bbox, image.size, (w, h))

        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, color='red', linewidth=2, zorder=10000000, axes=ax)
        ax.add_artist(rect)
        # ax.axis('off')

    fig.savefig(f"temp/mha_{image_id}_sparsemax.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
