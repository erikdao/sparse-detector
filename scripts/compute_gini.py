"""
This script is used to compute the Gini score for all decoder layers' attention maps
It's another adhoc script
"""
import json
import math
import os
import sys

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import numpy as np

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.coco import NORMALIZATION, CLASSES
from sparse_detector.datasets import transforms as T
from sparse_detector.utils.metrics import gini
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.configs import build_detr_config
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.utils.logging import MetricLogger, SmoothedValue

@click.command()
@click.option('--seed', type=int, default=42)
@click.option('--decoder-act', type=str, default='sparsemax')
@click.option('--coco-path', type=str, default="./data/COCO")
@click.option('--num-workers', default=12, type=int)
@click.option('--batch-size', default=8, type=int, help="Batch size per GPU")
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--detection-threshold', default=0.7, help='Threshold to filter detection results')
def main(resume_from_checkpoint, seed, decoder_act, coco_path, num_workers, batch_size, dist_url, detection_threshold):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dist_config = dist_utils.init_distributed_mode(dist_url)

    click.echo("Loading configs")
    configs = build_detr_config(device=device)
    if decoder_act:
        configs['decoder_act'] = decoder_act
    print(configs)

    click.echo("Building model with configs")
    model, criterion, postprocessors = build_model(**configs)

    click.echo("Load model from checkpoint")
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    describe_model(model)
    criterion.eval()

    click.echo("Building dataset")
    data_loader, base_ds = build_dataloaders('val', coco_path, batch_size, dist_config.distributed, num_workers)
    
    click.echo("Computing gini")
    for batch_idx, (samples, targets) in enumerate(data_loader):
        attentions = []
        conv_features = []
        hooks = [model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),]

        for i in range(6):
            hooks.append(
                model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
                    lambda self, input, output: attentions.append(output[1])
                )
            )

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        for hook in hooks:
            hook.remove()

        pred_logits = outputs['pred_logits'].detach() # .cpu()
        probas = pred_logits.softmax(-1)[0, :, :-1]
        print(f"probs: {probas.shape}")
        keep = probas.max(-1).values > detection_threshold
        queries = keep.nonzero()
        print(f"#queries: ", queries)
        h, w = conv_features[0]['3'].tensors.shape[-2:]

        print(f"Batch {batch_idx} --------------------")
        if batch_idx > 0:
            break

    # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], image.size)

    # use lists to store the outputs via up-values
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