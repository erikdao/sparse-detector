"""
Run detection model with input image(s) and get the results
"""
import os
import sys

import click
import torch
import torchvision.transforms as pth_transform
from PIL import Image

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.coco import NORMALIZATION
from sparse_detector.datasets import transforms as T


@click.command()
@click.argument('image_path', type=click.File('rb'))
@click.argument('checkpoint_path', type=click.File('rb'))
def main(image_path, checkpoint_path):
    image = Image.open(image_path)

    # The customer RandomResize outputs both the image and target even if
    # the target is None, so we perform the transform separately and only
    # feed the image to the next standard torch transform pipeline
    resize_transform = T.RandomResize([800], max_size=1333)
    resize_image, _ = resize_transform(image)

    transforms = pth_transform.Compose([
        pth_transform.ToTensor(),
        pth_transform.Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
    ])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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


if __name__ == "__main__":
    main()
