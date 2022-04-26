"""
This script is used to compute the Gini score for all decoder layers' attention maps
It's another adhoc script
"""
import json
import math
import os
import sys

from sparse_detector.configs import build_detr_config

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import torchvision.transforms as pth_transform
import numpy as np
from PIL import Image

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.coco import NORMALIZATION, CLASSES
from sparse_detector.datasets import transforms as T
from sparse_detector.utils.metrics import gini


@click.command()
@click.argument('checkpoint_path', type=click.File('rb'))
@click.option('--seed', type=int, default=42)
@click.option('--decoder-act', type=str, default='sparsemax')
def main(checkpoint_path, seed, decoder_act):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    click.echo("Loading configs")
    configs = build_detr_config(device=device)
    if decoder_act:
        configs['decoder_act'] = decoder_act
    print(configs)

    click.echo("Building model with configs")
    # model, criterion, postprocessors = build_model(
    #     backbone="resnet50", lr_backbone=1e-5, dilation=False,
    #     return_interm_layers=True, position_embedding="sine",
    #     hidden_dim=256, enc_layers=6, dec_layers=6,
    #     dim_feedforward=2048, dropout=0.1, num_queries=100,
    #     bbox_loss_coef=5, giou_loss_coef=2, eos_coef=0.1, aux_loss=False,
    #     set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
    #     nheads=8, pre_norm=True, device=device,
    #     decoder_act=decoder_act
    # )
    model, criterion, postprocessors = build_model(**configs)

    click.echo("Load model from checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    describe_model(model)

    criterion.eval()

    # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], image.size)

    # use lists to store the outputs via up-values
    # attentions = []
    # conv_features = []
    # hooks = [
    #     model.backbone[-2].register_forward_hook(
    #         lambda self, input, output: conv_features.append(output)
    #     ),
    # ]

    # for i in range(6):
    #     hooks.append(
    #         model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
    #             lambda self, input, output: attentions.append(output[1])
    #         )
    #     )

    # # propagate through the model
    # outputs = model(input_tensor)

    # # keep only predictions with 0.7+ confidence
    # pred_logits = outputs['pred_logits'].detach().cpu()
    # pred_boxes = outputs['pred_boxes'].detach().cpu()

    # probas = pred_logits.softmax(-1)[0, :, :-1]
    # print(f"probas: {probas.shape}")
    # keep = probas.max(-1).values > 0.7
    # print(f"keep: {keep.shape}")

    # for hook in hooks:
    #     hook.remove()

    # # don't need the list anymore
    # conv_features = conv_features[0]

    # # get the feature map shape
    # # Here we get the feature from the last block in the last layer of ResNet backbone
    # h, w = conv_features['3'].tensors.shape[-2:]

    # queries = keep.nonzero()
    # for idx, layer_attn in enumerate(attentions):  # Loop through the attention maps for each decoder layer
    #     attn_gini = 0.0
    #     for query in queries:
    #         attn = layer_attn[0, query].view(w, h).detach().cpu()
    #         attn_gini += gini(attn)
        
    #     attn_gini /= len(queries)
    #     print(f"Layer {idx}: attn_gini={attn_gini}")


if __name__ == "__main__":
    main()