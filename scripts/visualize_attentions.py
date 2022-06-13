"""
Visualizing self-attention of the model

Run
python -m scripts.visualize_attentions data/COCO/val2017/000000082807.jpg \
    checkpoints/v2_decoder_sparsemax/checkpoint.pth \
    --decoder-act sparsemax \
    --detr-config-file configs/decoder_sparsemax_baseline.yml
"""
import json
import math
import os
import sys
from sparse_detector.configs import build_detr_config

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
from sparse_detector.visualizations import ColorPalette


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
    """
    Rescale the predicted bounding boxes for visualization on the attention maps
    """
    imw, imh = img_size
    cw, ch = canvas_size
    w_scale, h_scale = imw * 1.0 / cw, imh * 1.0 / ch
    xmin, ymin, xmax, ymax = bbox
    b = (xmin * 1.0 / w_scale, ymin * 1.0 / h_scale, xmax * 1.0 / w_scale, ymax * 1.0 / h_scale)
    return b
    

def plot_attn_results(predictions, probabilities, image, image_size=None, groundtruth=None, filename=None):
    imw, imh = image_size
    # Figure out a good layout for the figure
    ncols = 4
    if len(predictions) > 12:
        ncols = 6
    nrows = len(predictions) // ncols
    if (len(predictions) % ncols) != 0:
        nrows = len(predictions) // ncols + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4.5))

    for idx, (ax, item) in enumerate(zip(axes.flat, predictions)):
        qid = None
        if idx % 2 == 0:
            attn, qid = item
            ax.imshow(attn)
            ax.set_title(f"query {qid.item()}")

            # When showing the attention maps, matplotlib create a canvas of different scale compared
            # with the case when an actual image is shown.
            bbox = predictions[idx+1]
            (xl_min, xl_max) = ax.get_xlim()
            (yl_max, yl_min) = ax.get_ylim()
            w = abs(xl_min) + abs(xl_max)
            h = abs(yl_min) + abs(yl_max)
            bbox = rescale_bboxes_for_ax(bbox, (imw, imh), (w, h))
        else:
            qid = predictions[idx-1][1]
            ax.imshow(image)
            bbox = item
            ax.set_title(CLASSES[probabilities[qid].argmax()])
        
        # Draw the groundtruth bounding boxes
        for gt_bbox in groundtruth:
            xmin, ymin, width, height = gt_bbox
            rect = patches.Rectangle((xmin, ymin), width, height,
                            fill=False, color=ColorPalette.GREEN, linewidth=2, zorder=1000, axes=ax)
            ax.add_artist(rect)

        # Draw the bounding box predictions
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, color=ColorPalette.RED, linewidth=2, zorder=1000, axes=ax)
        ax.add_artist(rect)

        # Annotate the class and probability
        if idx % 2 != 0:
            cl = probabilities[qid].argmax()
            prob = probabilities[qid, cl]
            text = f'{CLASSES[cl]}: {prob.item():0.2f}'
            ax.text(xmin, ymin, text, fontsize=10,
                    bbox=dict(facecolor=ColorPalette.YELLOW, alpha=0.5))
        ax.axis('off')

        fig.savefig(filename, bbox_inches="tight")


@click.command()
@click.argument('image_path', type=click.File('rb'))
@click.argument('checkpoint_path', type=click.File('rb'))
@click.option('--seed', type=int, default=42)
@click.option('--decoder-act', type=str, default='sparsemax')
@click.option('--decoder-layer', type=int, default=-1)
@click.option('--detr-config-file', type=str, default=None)
@click.pass_context
def main(ctx, image_path, checkpoint_path, seed, decoder_act, decoder_layer, detr_config_file):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detr_config = build_detr_config(detr_config_file, params=ctx.params, device=device)

    click.echo("Reading input image")
    image = Image.open(image_path)
    image_id = str(image_path.name).split("/")[-1].split(".")[0]

    # Getting groundtruth annotations
    with open("data/COCO/annotations/instances_val2017.json", "r") as f:
        data = json.load(f)
        annotations = data["annotations"]
        img_id = int(image_id.lstrip("0"))
        img_annos = list(filter(lambda x: x['image_id'] == img_id, annotations))
        image_bboxes = [anno['bbox'] for anno in img_annos]

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
    model, criterion, postprocessors = build_model(**detr_config)

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
        model.transformer.decoder.layers[decoder_layer].multihead_attn.register_forward_hook(
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

    plot_attn_results(items, probas, image, image.size, image_bboxes, f"temp/mha_{image_id}_{decoder_act}_dec-layer-{decoder_layer}.png")

    print("Saving attention maps")
    print(f"w={w}, h={h}")
    output = {
        "queries": queries.detach().cpu(),
        "attentions": dec_attn_weights.detach().cpu()
    }
    torch.save(output, f"temp/{decoder_act}_attns_{image_id}.pt")

if __name__ == "__main__":
    main()
