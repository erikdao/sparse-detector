"""
A very ad-hoc script for visualizations for attentions at all decoder layers of four models
for some given images
"""
import json
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
plt.rcParams["savefig.bbox"] = 'tight'
import matplotlib.patches as patches

import seaborn as sns
sns.set(font_scale=1.4)
sns.set_style("white", {
    "axes.edgecolor": '#475569',
    'font.family': ['sans-serif'],
    'font.sans-serif': ['Arial',
        'Droid Sans',
        'sans-serif'
    ],
})

from sparse_detector.models import build_model
from sparse_detector.datasets.coco import NORMALIZATION
from sparse_detector.datasets import transforms as T
from sparse_detector.visualizations import ColorPalette
from sparse_detector.utils.box_ops import rescale_bboxes


def main(image_path, checkpoint_path, detr_config_file, decoder_act=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detr_config = build_detr_config(detr_config_file, device=device)

    click.echo("Reading input image")
    image = Image.open(image_path)
    image_id = str(image_path).split("/")[-1].split(".")[0]

    # Getting groundtruth annotations
    with open("data/COCO/annotations/instances_val2017.json", "r") as f:
        data = json.load(f)
        annotations = data["annotations"]
        categories = data["categories"]
        img_id = int(image_id.lstrip("0"))
        img_annos = list(filter(lambda x: x['image_id'] == img_id, annotations))
        groundtruths = []

        for anno in img_annos:
            groundtruths.append({
                "category_id": anno['category_id'],
                "name": next(x for x in categories if x['id'] == anno['category_id'])["name"],
                "bbox": anno['bbox']
            })


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

    # propagate through the model
    outputs = model(input_tensor)
    # keep only predictions with 0.7+ confidence
    pred_logits = outputs['pred_logits'].detach().cpu()
    pred_boxes = outputs['pred_boxes'].detach().cpu()

    probas = pred_logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # convert boxes from [0; 1] to image scales
    predicted_boxes = rescale_bboxes(pred_boxes[0, keep], image.size)

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights = [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
    ]

    hooks += [
        model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ) for i in range(6)
    ]

    # propagate through the model
    outputs = model(input_tensor)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    # get the feature map shape
    # Here we get the feature from the last block in the last layer of ResNet backbone
    h, w = conv_features['3'].tensors.shape[-2:]

    queries = keep.nonzero()
    for i in range(6):
        visualize_layer_attentions_queries(
            dec_attn_weights[i],
            image,
            img_id,
            i+1,
            predicted_boxes,
            groundtruths,
            probas,
            queries,
            cnn_feat_w=w,
            cnn_feat_h=h
        )
        print(f"Visualized layer: {i+1}")
    # items = []
    # for idx, bbox in zip(queries, bboxes_scaled):
    #     items.append((dec_attn_weights[0, idx].view(h, w).detach().cpu().numpy(), idx))
    #     items.append(bbox)

    # plot_attn_results(items, probas, image, image.size, image_bboxes, f"temp/mha_{image_id}_{decoder_act}_dec-layer-{decoder_layer}.png")


def visualize_layer_attentions_queries(
    attentions, image, image_id, layer, predicted_boxes, groundtruths, predicted_probs, queries, cnn_feat_w, cnn_feat_h):
    """
    Arguments:
        attentions [1, N, dim]
    """

    def get_gt_for_label(label):
        return next(x for x in groundtruths if x['category_id'] == label)

    def plot_input_image(ax, pil_image, groundtruths):
        ax.imshow(pil_image)
        ax.axis('off')
        for gt in groundtruths:
            (xmin, ymin, width, height) =  gt['bbox']
            ax.add_patch(plt.Rectangle((xmin, ymin), width, height,
                                    fill=False, color=ColorPalette.GREEN, linewidth=2))
            text = gt['name']
            # Kill me: this is dirty
            if text == "couch":
                xmin = xmin + 5
            elif text == "dog":
                xmin = xmin + 5
                ymin = ymin + height - 10
            elif text == "sports ball":
                ymin = ymin - 15
            else:
                raise ValueError(f"Uh hm! Weird label {text}")

            ax.text(xmin+2, ymin, text, fontsize=12, bbox=dict(facecolor='white', edgecolor=None, alpha=0.8))

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

    def plot_attention_map(ax, attn_map, bbox, predicted_class, groundtruth, query, ylabel):
        ax.imshow(attn_map, cmap="viridis")

        assert predicted_class == groundtruth['category_id']
        ax.set_title("Query %d (%s)" % (query, groundtruth['name']), pad=10)
    
        # When showing the attention maps, matplotlib create a canvas of different scale compared
        # with the case when an actual image is shown.
        (xl_min, xl_max) = ax.get_xlim()
        (yl_max, yl_min) = ax.get_ylim()
        w = abs(xl_min) + abs(xl_max)
        h = abs(yl_min) + abs(yl_max)
        imw, imh = image.size

        # Draw predicted box
        bbox = rescale_bboxes_for_ax(bbox, (imw, imh), (w, h))
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=ColorPalette.RED, linewidth=2, zorder=1000, axes=ax)
        ax.add_artist(rect)
        if ylabel is not None:
            ax.set_ylabel(ylabel, labelpad=10, fontdict=dict(fontsize=18, fontweight="bold"))
            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_visible(False)
        else:
            ax.axis('off')

    # Setup figure
    nrows = 1
    ncols = len(queries) + 1  # +1 for the input image
    assert ncols == 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5))
    plot_input_image(axes[0], image, groundtruths)

    # plot attention map
    attentions = attentions[0]
    for idx, query in enumerate(queries):
        attention_map = attentions[query].view(cnn_feat_h, cnn_feat_w).detach().cpu().numpy()
        bbox = predicted_boxes[idx]
        predicted_class = predicted_probs[query].argmax().item()
        groundtruth = get_gt_for_label(predicted_class)
        
        ylabel = f"Layer {layer}" if idx == 0 else None
        plot_attention_map(axes[idx+1], attention_map, bbox, predicted_class, groundtruth, query.item(), ylabel)

    # fig.subplots_adjust(wspace=0.02, hspace=0)
    plt.tight_layout()
    filename = "outputs/cherry_pick_attentions/alpha_entmax/%s_layer-%d" % (str(image_id), layer)
    fig.savefig(filename + ".png", bbox_inches="tight")
    fig.savefig(filename + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    image_path = "misc/input_images/000000347930.jpg"
    # checkpoint_path = "checkpoints/v2_baseline_detr/checkpoint.pth"
    # detr_config_file = "configs/detr_baseline.yml"
    checkpoint_path = "checkpoints/v2_decoder_alpha_entmax/checkpoint.pth"
    detr_config_file = "configs/decoder_alpha_entmax.yml"

    main(image_path, checkpoint_path, detr_config_file)
