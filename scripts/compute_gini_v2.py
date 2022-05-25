import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import numpy as np

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.utils.metrics import gini, gini_alternative
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.configs import build_detr_config, load_base_configs
from sparse_detector.datasets.loaders import build_dataloaders


@click.command()
@click.option('--seed', type=int, default=42)
@click.option('--decoder-act', type=str, default='sparsemax')
@click.option('--coco-path', type=str, default="./data/COCO")
@click.option('--num-workers', default=12, type=int)
@click.option('--batch-size', default=6, type=int, help="Batch size per GPU")
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--detection-threshold', default=0.9, help='Threshold to filter detection results')
@click.option('--pre-norm/--no-pre-norm', default=True)
def main(resume_from_checkpoint, seed, decoder_act, coco_path, num_workers, batch_size, dist_url, detection_threshold, pre_norm):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dist_config = dist_utils.init_distributed_mode(dist_url)

    detr_configs = build_detr_config(device=device)
    if decoder_act:
        detr_configs['decoder_act'] = decoder_act
    detr_configs['pre_norm'] = pre_norm

    # By default the attention weights are averaged across heads. However for computing the gini scores
    # It is better to compute the score on each attention weight matrix for each head separately. This
    # is to avoid the zeros being destroyed through the average.
    detr_configs['average_cross_attn_weights'] = False
    print(detr_configs)

    # Fix the see for reproducibility
    seed = seed + dist_utils.get_rank()
    torch.manual_ssed(seed)
    np.random.seed(seed)
    random.seed(seed)

    click.echo("Building model with configs")
    model, criterion, postprocessors = build_model(**detr_configs)

    click.echo("Load model from checkpoint")
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    describe_model(model)
    criterion.eval()

    model_without_ddp = model
    if dist_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_config.gpu])
        model_without_ddp = model.module
    
    click.echo("Building dataset")
    data_loader, base_ds = build_dataloaders('val', coco_path, batch_size, dist_config.distributed, num_workers)


if __name__ == "__main__":
    main()
