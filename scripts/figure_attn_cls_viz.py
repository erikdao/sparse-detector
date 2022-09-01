"""
An ad-hoc script to visualize the attention for each object class of all models
"""
import os
import sys
from pathlib import Path

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
torch.manual_seed(42)
device = torch.device('cuda')

import torchvision.transforms as pth_transforms

import numpy as np
np.random.seed(42)
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
from sparse_detector.configs import build_detr_config, build_dataset_config, load_config


def load_model(config_file_name, model_checkpoint_dir):
    detr_config_file = Path(package_root) / "configs" / config_file_name
    base_config = load_config(Path(package_root) / "configs" / "base.yml")
    detr_config = build_detr_config(detr_config_file, params=None, device=device)

    # Surgery for path
    dataset_config = build_dataset_config(base_config["dataset"], params=None)
    dataset_config['coco_path'] = Path(package_root) / "data" / "COCO"

    detr_config["average_cross_attn_weights"] = True

    # Build model
    model, _, _ = build_model(**detr_config)
    resume_from_checkpoint = Path(package_root) / "checkpoints" / model_checkpoint_dir / "checkpoint.pth"
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model, detr_config, dataset_config


def load_image(img_id):
    coco_path = Path(package_root) / "data" / "COCO"
    img_path = coco_path / "val2017" / (f"{img_id}".rjust(12, '0') + '.jpg')
    pil_img = Image.open(img_path)

    resize_transform = T.RandomResize([800], max_size=1333)
    resize_image, _ = resize_transform(pil_img)
    print("resize_image", resize_image.size)

    transforms = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
    ])

    input_tensor = transforms(resize_image).unsqueeze(0).to(device)
    return input_tensor, pil_img


def main(image_id, box_annotations, categories):
    # image_id = 491130
    # box_annotations = [
    #     {'bbox': [100.67, 115.06, 314.97, 362.42], 'category': 'person', 'id': 1, 'image_id': 491130},
    #     {'bbox': [10.07, 222.92, 316.4, 345.17], 'category': 'snowboard', 'id': 36, 'image_id': 491130},
    #     {'bbox': [52.49, 152.53, 62.78, 104.91], 'category': 'person', 'id': 1, 'image_id': 491130}
    # ]
    # categories = {1: "person", 36: "snowboard"}

    input_tensor, image = load_image(image_id)

    model_list = {
        "softmax": ("detr_baseline.yml", "v2_baseline_detr"),
        # "sparsemax": ("decoder_sparsemax_baseline.yml", "v2_decoder_sparsemax"),
        # "entmax15": ("decoder_entmax15_baseline.yml", "v2_decoder_entmax15"),
        # "alpha_entmax": ("decoder_alpha_entmax.yml", "v2_decoder_alpha_entmax"),
    }

    model_results = dict()

    for model_name, model_config in model_list.items():
        config_file, checkpoint_dir = model_config
        model, _, _ = load_model(config_file, checkpoint_dir)
        # use lists to store the outputs via up-values
        conv_features, dec_attn_weights = [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            )
        ]

        # propagate through the model
        outputs = model(input_tensor)
        for hook in hooks:
            hook.remove()

        # keep only predictions with 0.7+ confidence
        pred_logits = outputs['pred_logits'].detach().cpu()
        pred_boxes = outputs['pred_boxes'].detach().cpu()

        probas = pred_logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5
        queries = keep.nonzero()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attentions = dec_attn_weights[0]

        # get the feature map shape
        # Here we get the feature from the last block in the last layer of ResNet backbone
        h, w = conv_features['3'].tensors.shape[-2:]
        print("h=", h, "w=", w)

        output = []

        for idx, query in enumerate(queries):
            print("query", idx, "map", dec_attentions[0, query].shape)
            attn_map = dec_attentions[0, query].view(h, w).detach().cpu()
            sparse = (attn_map == 0.0).type(torch.IntTensor).sum()
            print("sparse", sparse)

            predicted_class = probas[query].argmax().item()
            predicted_boxes = rescale_bboxes(pred_boxes[0, query], image.size)

            item = {
                    "query": query.item(),
                    "attention_map": attn_map,
                    "predicted_class": predicted_class,
                    "predicted_label": categories[predicted_class],
                    "predicted_prob": probas[query, predicted_class].item(),
                    "predicted_boxes": predicted_boxes.squeeze(0),  # pred_boxes[query].squeeze(0)
                }
            output.append(item)

        model_results[model_name] = output
    visualize_model_results(model_results, image, box_annotations)

    # torch.save(model_results, f"temp/{image_id}_result.pt")
        # fig, axes = plt.subplots(nrows=1, ncols=len(output), figsize=(5 * (len(output) - 1), 5))
        # for idx, item in enumerate(output):
        #     ax = axes[idx]
        #     ax.imshow(item['attention_map'], cmap='viridis')
        #     ax.set_title(item['query'])
        #     xlabel = "%s: %.4f" % (item['predicted_label'], item['predicted_prob'])
        #     ax.set_xlabel(xlabel)

        #     (xl_min, xl_max) = ax.get_xlim()
        #     (yl_max, yl_min) = ax.get_ylim()
        #     w = abs(xl_min) + abs(xl_max)
        #     h = abs(yl_min) + abs(yl_max)
        #     imw, imh = image.size

        #     bbox = rescale_bboxes_for_ax(item['predicted_boxes'], (imw, imh), (w, h))
        #     xmin, ymin, xmax, ymax = bbox
        #     rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                         fill=False, color=ColorPalette.YELLOW, linewidth=2, zorder=1000, axes=ax)
        #     ax.add_artist(rect)
        
        # fig.savefig(f"temp/{image_id}_{model_name}.png", bbox_inches="tight")

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

def visualize_model_results(model_results, image, bbox_annotations):
    def plot_input_image(ax, image, annotation):
        COLOR = '#86efac'
        ax.imshow(image)
        ax.axis('off')

        category = annotation['category']
        xmin, ymin, width, height = annotation['bbox']
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=COLOR, fill=False, linewidth=3)
        ax.add_artist(rect)
        ax.text(xmin+6, ymin-18, category, fontsize=18, bbox=dict(facecolor=COLOR, edgecolor=COLOR))
        
        ax.set_ylabel(category, labelpad=5, fontsize=18)
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_visible(False)

    def plot_attention_map(ax, data, groundtruth, model):
        query = data["query"]
        attention_map = data["attention_map"]
        predicted_class = data["predicted_class"]
        predicted_label = data["predicted_label"]
        predicted_prob = data["predicted_prob"]
        predicted_boxes = data["predicted_boxes"]

        ax.imshow(attention_map, cmap="viridis")

        # assert predicted_class == groundtruth, f"predicted_class: {predicted_class}; groundtruth: {groundtruth}"
        ax.set_title("Query %d: %s" % (query, predicted_label), pad=10)
    
        # When showing the attention maps, matplotlib create a canvas of different scale compared
        # with the case when an actual image is shown.
        (xl_min, xl_max) = ax.get_xlim()
        (yl_max, yl_min) = ax.get_ylim()
        w = abs(xl_min) + abs(xl_max)
        h = abs(yl_min) + abs(yl_max)
        imw, imh = image.size

        # Draw predicted box
        # print(predicted_boxes.shape)
        bbox = rescale_bboxes_for_ax(predicted_boxes, (imw, imh), (w, h))
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color=ColorPalette.YELLOW, linewidth=4, zorder=1000, axes=ax)
        ax.add_artist(rect)
        # text = f"{predicted_label}: {predicted_prob:.4f}"
        # ax.text(xmin+0.48, ymin-0.6, text, fontsize=14, bbox=dict(facecolor=ColorPalette.YELLOW, edgecolor=ColorPalette.YELLOW))

        # ax.set_xlabel(model)
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_visible(False)
        ax.axis('off')

    for idx, ann in enumerate(bbox_annotations):
        category = ann['id']
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 12))

        # plot_input_image(axes[0], image, ann)

        for mid, (model_name, result) in enumerate(model_results.items()):
            matched_predictions = [x for x in result if x['predicted_class'] == category]
            if len(matched_predictions) == 0:
                print(model_name, category)
            plot_attention_map(ax, matched_predictions[0], category, model_name)

        fig.subplots_adjust(wspace=0.02)

        image_id = ann['image_id']
        fig.savefig(f"outputs/presentations/{image_id}_{idx}_row.pdf", bbox_inches="tight")
        fig.savefig(f"outputs/presentations/{image_id}_{idx}_row.png", bbox_inches="tight")


if __name__ == "__main__":
    image_id = 291664  # 170474
    # box_annotations = [
    #     {'bbox': [345.92, 34.32, 26.1, 27.45], 'category': 'sports ball', 'id': 37, 'image_id': image_id},
    #     {'bbox': [153.17, 27.02, 304.04, 448.98], 'category': 'person', 'id': 1, 'image_id': image_id},
    #     {'bbox': [64.32, 312.27, 121.65, 150.65], 'category': 'tennis racket', 'id': 43, 'image_id': image_id},
    #     {'bbox': [489.53, 408.47, 109.47, 61.5], 'category': 'chair', 'id': 62, 'image_id': image_id}
    # ]
    box_annotations = [
        {'bbox': [280.66, 169.34, 209.06, 328.8], 'category': 'dog', 'id': 18, 'image_id': image_id},
        {'bbox': [131.38, 65.1, 184.1, 396.37], 'category': 'fire hydrant', 'id': 11, 'image_id': image_id}
    ]
    categories = dict()
    for ann in box_annotations:
        categories[ann['id']] = ann['category']

    main(image_id, box_annotations, categories)
