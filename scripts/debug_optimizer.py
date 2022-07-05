"""
Inspect optimizer param groups

Run
python -m scripts.debug_optimizer --detr-config-file "configs/decoder_alpha_entmax.yml"  --decoder-act alpha_entmax
"""
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

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
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.option('--start-epoch', default=0, type=int, help='start epoch')
@click.option('--epochs', default=300, type=int, help='number of training epochs')
@click.option('--batch-size', default=6, type=int, help='batch size')
@click.option('--num-workers', default=24, type=int)
@click.pass_context
def main(ctx, detr_config_file, exp_name, seed, decoder_act, resume_from_checkpoint, start_epoch, epochs, 
         batch_size, num_workers):
    # Load the base config and initialise distributed training mode first
    # to avoid multiple hassles in printing
    base_configs = load_base_configs()
    print("git:\n  {}\n".format(utils.get_sha()))

    print("Base config", base_configs)
    cmd_params = ctx.params
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if cmd_params['decoder_act'] not in VALID_ACTIVATION:
        raise ValueError(f"Unsupported decoder activation: {cmd_params['decoder_act']}")

    detr_config = build_detr_config(cmd_params['detr_config_file'], params=cmd_params, device=device)
    print("DETR config", detr_config)

    trainer_config = build_trainer_config(base_configs['trainer'], params=cmd_params)
    print("Trainer config", trainer_config)

    # Fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Building DETR model...")
    model, criterion, postprocessors = build_model(**detr_config)
    model.to(device)
    describe_model(model)

    print("Building optim...")
    optimizer, lr_scheduler = build_detr_optims(
        model,
        lr=trainer_config['lr'],
        lr_backbone=detr_config['lr_backbone'],
        lr_drop=trainer_config['lr_drop'],
        weight_decay=trainer_config['weight_decay']
    )

    # optimizer.param_groups is a list, the first item in the param_groups
    # is likely the ones for alphas
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
