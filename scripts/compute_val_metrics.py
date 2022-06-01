"""
Computing metrics on COCO validation set

This script computes the Gini Score for self-attention matrices per head per decoder's layer
for all images in the COCO validation set
"""
import os
import sys
import time
import random
import datetime
import warnings
warnings.filterwarnings("ignore")

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import numpy as np
from tqdm import tqdm

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.utils.metrics import gini, gini_alternative, gini_sorted, zeros_ratio
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.configs import build_detr_config, load_base_configs, build_dataset_config
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.utils.logging import MetricLogger


@click.command()
@click.argument('metric', default='gini', type=str)
@click.option('--metric-threshold', type=float, default=None, help='Threshold for metric')
@click.option('--seed', type=int, default=42)
@click.option('--detr-config-file', default='', help="Path to config file")
@click.option('--decoder-act', type=str, default='sparsemax')
@click.option('--coco-path', type=str, default="./data/COCO")
@click.option('--num-workers', default=12, type=int)
@click.option('--batch-size', default=6, type=int, help="Batch size per GPU")
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--detection-threshold', default=0.9, help='Threshold to filter detection results')
@click.option('--pre-norm/--no-pre-norm', default=True)
@click.pass_context
def main(
    ctx, metric, detr_config_file, resume_from_checkpoint, seed,
    decoder_act, pre_norm, coco_path, num_workers, batch_size, detection_threshold,
    metric_threshold
):
    if metric not in ['gini', 'zeros_ratio']:
        raise ValueError(f"Metric {metric} not supported! Quitting!")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    base_config = load_base_configs()
    dist_config = dist_utils.init_distributed_mode(base_config['distributed']['dist_url'])
    detr_config = build_detr_config(detr_config_file, params=ctx.params, device=device)
    dataset_config = build_dataset_config(base_config['dataset'], params=ctx.params)

    # By default the attention weights are averaged across heads. However for computing the gini scores
    # It is better to compute the score on each attention weight matrix for each head separately. This
    # is to avoid the zeros being destroyed through the average.
    detr_config['average_cross_attn_weights'] = False
    print(detr_config)

    # Fix the see for reproducibility
    seed = seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Building model with configs")
    model, criterion, _ = build_model(**detr_config)

    if resume_from_checkpoint:
        print(f"Load model from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    describe_model(model)
    criterion.eval()

    if dist_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_config.gpu])

    print("Building dataset")
    data_loader, _ = build_dataloaders(
        'val', dataset_config['coco_path'], dataset_config['batch_size'],
        dist_config.distributed, dataset_config['num_workers']
    )

    print(f"Computing {metric}")
    metric_logger = MetricLogger(delimiter=" ")
    header = f"{metric}:"

    start_time = time.time()
    dataset_metric = []
    for batch_id, (samples, targets) in enumerate(metric_logger.log_every(data_loader, log_freq=10, header=header, prefix="val")):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        attentions = []
        conv_features = []
        hooks = [model.module.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),]

        for i in range(6):
            hooks.append(
                model.module.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
                    lambda self, input, output: attentions.append(output[1])
                )
            )

        outputs = model(samples)
        for hook in hooks:
            hook.remove()

        h, w = conv_features[0]['3'].tensors.shape[-2:]

        # At this point the `attentions` matrix would have the following properties:
        # - Length: 6 - output from 6 decoder's layers
        # - Each item: [B, num_heads, num_queries, h*w]: each item contains the attentions
        #              from each head for all images in the batch

        pred_logits = outputs['pred_logits'].detach().cpu()  # [B, num_queries, num_classes]
        probas = pred_logits.softmax(-1)[:, :, :-1]  # [B, num_queries, 91]
        # Hack: > 0.0 so it will keep all attentions from all queries
        batch_keep = probas.max(-1).values > 0.0 # detection_threshold # [B, num_queries]

        batch_metric = []
        # For each image in the batch
        for img_idx, keep in enumerate(batch_keep):
            assert keep.shape == (100,)
            queries = keep.nonzero().squeeze(-1)  # [kept_queries,]
            if len(queries) == 0:
                continue
            
            num_heads = 8
            num_queries = len(queries)
            # List of attention maps from all decoder's layer for this particular image
            # attn[img_idx] is of shape [num_heads, num_queries, h*w] (e.g., [8, 100, 1444])
            img_attentions = [attn[img_idx].detach().cpu() for attn in attentions]
            assert len(img_attentions) == 6
            
            image_metric = []
            for layer_idx, layer_attns in enumerate(img_attentions):
                attn_metric = 0.0
                for head in range(8):  # iterate across heads
                    for query in queries:
                        attn_q_h = layer_attns[head][query].view(w, h).detach().cpu()
                        if metric == 'zeros_ratio':
                            attn_metric += zeros_ratio(attn_q_h, threshold=metric_threshold)
                        elif metric == 'gini':
                            attn_metric += gini_sorted(attn_q_h)
                
                attn_metric /= (num_queries * num_heads)
                image_metric.append(attn_metric)
            
            image_metric_t = torch.stack(image_metric)
            batch_metric.append(image_metric_t)
        
        batch_metric_t = torch.stack(batch_metric)
        batch_metric_score = torch.mean(batch_metric_t, 0)

        dataset_metric.extend(batch_metric)
        del attentions
        del conv_features
    
    rank_gini = torch.stack(dataset_metric)
    click.echo(f"Rank: {dist_config.rank}; Mean: {torch.mean(rank_gini, 0)}")

    final_score = dist_utils.all_gather(rank_gini)
    final_score = torch.cat(final_score, dim=0)

    print("Final scores", final_score.shape)
    print(f"Mean: {torch.mean(final_score, 0)}")
    print(f"Std: {torch.std(final_score, 0)}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Time spent: {}".format(total_time_str))


if __name__ == "__main__":
    main()