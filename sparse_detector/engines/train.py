"""
Entry point for training DETR models
"""
import os
import sys
import time
import yaml
import json
import random
import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import wandb
import click
import numpy as np

import torch

package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, package_root)

from sparse_detector.configs import build_dataset_config, build_detr_config, load_base_configs, build_trainer_config
from sparse_detector.utils import misc as utils
from sparse_detector.utils import distributed  as dist_utils
from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.datasets.loaders import build_dataloaders
from sparse_detector.engines.base import build_detr_optims, train_one_epoch, evaluate
from sparse_detector.utils.logging import log_ap_to_wandb, log_to_wandb
from sparse_detector.models.attention import VALID_ACTIVATION


@click.command("train_detr")
@click.option('--detr-config-file', default='', help="Path to config file")
@click.option('--exp-name', default=None, help='Experiment name. Need to be set')
@click.option('--seed', default=42, type=int)
@click.option('--decoder-act', default='softmax', type=str, help='Activation function for the decoder MH cross-attention')
@click.option('--coco-path', type=str)
@click.option('--output-dir', default='checkpoints', help='path where to save, empty for no saving')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--start-epoch', default=0, type=int, help='start epoch')
@click.option('--epochs', default=300, type=int, help='number of training epochs')
@click.option('--batch-size', default=6, type=int, help='batch size')
@click.option('--num-workers', default=24, type=int)
@click.option('--wandb-log/--no-wandb-log', default=True, help="Whether to enable logging to W&B")
@click.option('--wandb-group', default=None, help="The group for experiment on W&B")
@click.option('--wandb-id', default=None, help="Run ID for resume")
@click.pass_context
def main(ctx, detr_config_file, exp_name, seed, decoder_act, coco_path,
         output_dir, resume_from_checkpoint, start_epoch, epochs, 
         batch_size, num_workers, wandb_log, wandb_group, wandb_id):
    # Load the base config and initialise distributed training mode first
    # to avoid multiple hassles in printing
    base_configs = load_base_configs()
    dist_config = dist_utils.init_distributed_mode(base_configs['distributed']['dist_url'])
    print("git:\n  {}\n".format(utils.get_sha()))

    print("Base config", base_configs)
    cmd_params = ctx.params
    print("cmd_params", cmd_params)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if cmd_params['decoder_act'] not in VALID_ACTIVATION:
        raise ValueError(f"Unsupported decoder activation: {cmd_params['decoder_act']}")

    detr_config = build_detr_config(cmd_params['detr_config_file'], params=cmd_params, device=device)
    print("DETR config", detr_config)

    trainer_config = build_trainer_config(base_configs['trainer'], params=cmd_params)
    print("Trainer config", trainer_config)

    dataset_config = build_dataset_config(base_configs['dataset'], params=ctx.params)

    wandb_run = None
    wandb_configs = None
    if dist_utils.is_main_process() and cmd_params['wandb_log']:
        print("Initialize WandB logging...")
        wandb_configs = base_configs.get("wandb")
        wandb_configs["name"] = cmd_params['exp_name']
        if cmd_params['wandb_id'] is not None:
            wandb_configs["id"] = cmd_params['wandb_id']
            wandb_configs["resume"] = True

        if cmd_params['wandb_group'] is not None:
            wandb_configs["group"] = cmd_params['wandb_group']

        config_to_log = {**trainer_config, **detr_config, **dataset_config}
        wandb_run = wandb.init(**wandb_configs, config=config_to_log)

    # Fix the seed for reproducibility
    seed = seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Setup output directory")
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if dist_utils.is_main_process():
        config_to_dump = {"base": base_configs, "detr": detr_config, "wandb": wandb_configs, "cmd_params": cmd_params}
        with open(exp_dir / "configs.yml", "w") as f:
            yaml.dump(config_to_dump, f)

    print("Building DETR model...")
    model, criterion, postprocessors = build_model(**detr_config)
    model.to(device)

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients")

    model_without_ddp = model
    if dist_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_config.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    describe_model(model_without_ddp)

    print("Building datasets and data loaders...")
    data_loader_train, sampler_train = build_dataloaders(
        'train', dataset_config['coco_path'], dataset_config['batch_size'],
        dist_config.distributed, dataset_config['num_workers']
    )
    data_loader_val, base_ds = build_dataloaders(
        'val', dataset_config['coco_path'], dataset_config['batch_size'],
        dist_config.distributed, dataset_config['num_workers']
    )

    print("Building optim...")
    optimizer, lr_scheduler = build_detr_optims(
        model_without_ddp,
        lr=trainer_config['lr'],
        lr_backbone=detr_config['lr_backbone'],
        lr_drop=trainer_config['lr_drop'],
        weight_decay=trainer_config['weight_decay']
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
            trainer_config['clip_max_norm'], global_step=global_step, wandb_run=wandb_run, log_freq=base_configs['logging'].get('log_freq'),
            monitor_alpha=(decoder_act == "alpha_entmax")
        )
        lr_scheduler.step()
        if exp_dir:
            checkpoint_paths = [exp_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % trainer_config['lr_drop'] == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(exp_dir / f'checkpoint_{epoch:04}.pth')
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
            log_ap_to_wandb(wandb_run, test_stats.get("coco_eval_bbox"), epoch=epoch, prefix="val-AP")

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
                    if epoch % 10 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, exp_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    wandb.finish()


if __name__ == "__main__":
    main()
