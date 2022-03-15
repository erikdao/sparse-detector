"""
Entry point for training DETR Baseline
"""
import os
import sys
import time
import json
import random
import datetime
from pathlib import Path

import wandb
import click
import numpy as np

import torch

package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, package_root)

from sparse_detector.configs import load_base_configs
from sparse_detector.utils import misc as utils
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.models import build_model
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.engines.base import build_detr_optims, train_one_epoch, evaluate
from sparse_detector.utils.logging import log_to_wandb


@click.command("train_baseline")
@click.option('--config', default='', help="Path to config file")
@click.option('--exp-name', default=None, help='Experiment name. Need to be set')
@click.option('--seed', default=42, type=int)
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
@click.option('--output-dir', default='', help='path where to save, empty for no saving')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--start-epoch', default=0, type=int, help='start epoch')
@click.option('--num-workers', default=12, type=int)
@click.option('--dist_url', default='env://', help='url used to set up distributed training')
@click.pass_context
def main(
    ctx, config, exp_name, seed, backbone, lr_backbone, lr, batch_size, weight_decay, epochs,
    lr_drop, clip_max_norm, dilation, position_embedding, enc_layers, dec_layers, dim_feedforward,
    hidden_dim, dropout, nheads, num_queries, pre_norm, aux_loss, set_cost_class,
    set_cost_bbox, set_cost_giou, bbox_loss_coef, giou_loss_coef, eos_coef,
    dataset_file, coco_path, output_dir, resume_from_checkpoint, start_epoch, num_workers, dist_url
):
    # TODO: Update default configs with ctx.params
    args = locals()
    dist_config = dist_utils.init_distributed_mode(dist_url)
    default_configs = load_base_configs()
    print("Initialized distributed training")
    
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    print("Initialize WandB logging...")
    wandb_run = None
    if dist_utils.is_main_process():
        wandb_configs = default_configs.get("wandb")
        wandb_configs["name"] = exp_name
        wandb_run = wandb.init(**wandb_configs)
        # wandb.config.update(**ctx.params)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # fix the seed for reproducibility
    seed = seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Setup output directory")
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("Building DETR model...")
    model, criterion, postprocessors = build_model(
        backbone,
        lr_backbone,
        dilation,
        True,  # return_interm_layers: boo
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
        device=device
    )
    model.to(device)

    model_without_ddp = model
    if dist_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_config.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)

    print("Building datasets and data loaders...")
    data_loader_train, data_loader_val, base_ds, sampler_train = build_dataloaders(
        dataset_file, coco_path, batch_size, dist_config.distributed, num_workers
    )

    print("Building optim...")
    optimizer, lr_scheduler = build_detr_optims(
        model_without_ddp,
        lr=lr,
        lr_backbone=lr_backbone,
        lr_drop=lr_drop,
        weight_decay=weight_decay
    )

    global_step = 0  # Initialize the global step
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step'] + 1
        print(f"Resumming from checkpoint {resume_from_checkpoint}")

    print("Start training...")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        if dist_config.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, global_step = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            clip_max_norm, global_step=global_step, wandb_run=wandb_run, log_freq=default_configs["logging"].get("log_freq")
        )
        lr_scheduler.step()
        if exp_dir:
            checkpoint_paths = [exp_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(exp_dir / f'checkpoint_{epoch:04}_step-{global_step:08}.pth')
            for checkpoint_path in checkpoint_paths:
                dist_utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'hyperparams': ctx.params,
                    'global_step': global_step,
                }, checkpoint_path)
        
        # Logging epoch train stats to W&B
        if dist_utils.is_main_process():
            log_to_wandb(wandb_run, train_stats, epoch=epoch, prefix="train-epoch")

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, epoch, wandb_run=wandb_run
        )
        if dist_utils.is_main_process():
            log_to_wandb(wandb_run, test_stats, epoch=epoch, prefix="val-epoch")

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'global_step': global_step
        }

        if exp_dir and dist_utils.is_main_process():
            with (exp_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (exp_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, exp_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    wandb.finish()


if __name__ == "__main__":
    main()
