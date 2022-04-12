"""
This scripts generate predictions for all images in the evaluation dataset
"""
import os
import sys
import random
from typing import Any, Optional

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import torch
import click
import numpy as np

from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.utils import distributed as dist_utils
from sparse_detector.datasets.loaders import build_dataloaders


@click.command("generate_predictions")
@click.option('--seed', default=42, type=int)
@click.option('--decoder-act', default='softmax', type=str, help='Activation function for the decoder MH cross-attention')
@click.option('--lr', default=1e-4, type=float, help="Transformer detector learing rate")
@click.option('--lr-backbone', default=1e-5, type=float, help="Backbone learning rate")
@click.option('--batch-size', default=8, type=int, help="Batch size per GPU")
@click.option('--weight-decay', default=1e-4, type=float, help="Optimizer's weight decay")
@click.option('--epochs', default=300, type=int, help="Number of epochs")
@click.option('--lr-drop', default=200,type=int, help="Epoch after which learning rate is dropped")
@click.option('--clip-max-norm', default=0.1, type=float, help='gradient clipping max norm')
@click.option('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
@click.option('--dilation/--no-dilation', default=False, help="If true, we replace stride with dilation in the last convolutional block (DC5)")
@click.option('--position-embedding', default='sine', type=str, help="Type of positional embedding to use on top of the image features")
@click.option('--enc-layers', default=6, type=int, help="Number of encoding layers in the transformer")
@click.option('--dec-layers', default=6, type=int, help="Number of decoding layers in the transformer")
@click.option('--dim-feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
@click.option('--hidden-dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
@click.option('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
@click.option('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
@click.option('--num-queries', default=100, type=int, help="Number of query slots")
@click.option('--pre-norm/--no-pre-norm', default=True)
@click.option('--aux-loss/--no-aux-loss', default=True, help="Whether to use auxiliary decoding losses (loss at each layer)")
@click.option('--set-cost-class', default=1, type=float, help="Class coefficient in the matching cost")
@click.option('--set-cost-bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
@click.option('--set-cost-giou', default=2, type=float, help="giou box coefficient in the matching cost")
@click.option('--bbox-loss-coef', default=5, type=float)
@click.option('--giou-loss-coef', default=2, type=float)
@click.option('--eos-coef', default=0.1, type=float, help="Relative classification weight of the no-object class")
@click.option('--dataset-file', default='coco')
@click.option('--coco-path', type=str)
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--num-workers', default=12, type=int)
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
@click.pass_context
def main(
    ctx, seed, decoder_act, lr, lr_backbone, batch_size, weight_decay, lr_drop, epochs, clip_max_norm,
    backbone, dilation, position_embedding, enc_layers, dec_layers, dim_feedforward, hidden_dim,
    dropout, nheads, num_queries, pre_norm, aux_loss, set_cost_class, set_cost_bbox, set_cost_giou,
    bbox_loss_coef, giou_loss_coef, eos_coef, coco_path, dataset_file, resume_from_checkpoint,
    num_workers, dist_url
) -> None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dist_config = dist_utils.init_distributed_mode(dist_url)

    # Fix the seed for reproducibility
    seed = seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    click.echo("Building DETR model...")
    model, criterion, postprocessors = build_model(
        backbone,
        lr_backbone,
        dilation,
        True,  # return_interm_layers: bool
        position_embedding,
        hidden_dim,
        enc_layers,
        dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        eos_coef=eos_coef,
        aux_loss=aux_loss,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        nheads=nheads,
        pre_norm=pre_norm,
        dataset_file=dataset_file,
        device=device,
        decoder_act=decoder_act
    )
    model.to(device)

    model_without_ddp = model
    if dist_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_config.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    describe_model(model_without_ddp)

    click.echo("Load model from checkpoint...")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint['model'])

    click.echo("Build datasets and data loaders...")
    data_loader_train, data_loader_val, base_ds, sampler_train = build_dataloaders(
        dataset_file, coco_path, batch_size, dist_config.distributed, num_workers
    )


if __name__ == "__main__":
    main()
