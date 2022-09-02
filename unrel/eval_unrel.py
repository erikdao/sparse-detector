"""
Evaluate detection performance on UnRel dataset

python -m unrel.eval_unrel --detr-config-file "configs/detr_baseline.yml" \
    --decoder-act softmax \
    --coco-path "data/UnRel" \
    --resume-from-checkpoint "checkpoints/v2_baseline_detr/checkpoint.pth"

python -m unrel.eval_unrel --detr-config-file "configs/decoder_sparsemax_baseline.yml" \
    --decoder-act sparsemax \
    --coco-path "data/UnRel" \
    --resume-from-checkpoint "checkpoints/v2_decoder_sparsemax/checkpoint.pth"

python -m unrel.eval_unrel --detr-config-file "configs/decoder_entmax15_baseline.yml" \
    --decoder-act entmax15 \
    --coco-path "data/UnRel" \
    --resume-from-checkpoint "checkpoints/v2_decoder_entmax15/checkpoint.pth"

python -m unrel.eval_unrel --detr-config-file "configs/decoder_alpha_entmax.yml" \
    --decoder-act alpha_entmax \
    --coco-path "data/UnRel" \
    --resume-from-checkpoint "checkpoints/v2_decoder_alpha_entmax/checkpoint.pth"
"""
import os
import sys
import random

package_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, package_root)

import click
import torch
import numpy as np

# project package
from sparse_detector.engines.base import evaluate
from sparse_detector.models import build_model
from sparse_detector.models.utils import describe_model
from sparse_detector.configs import build_dataset_config, build_detr_config, load_base_configs

from unrel.data_utils import build_dataloaders


@click.command("eval_unrel")
@click.option('--detr-config-file', default='', help="Path to config file")
@click.option('--seed', default=0, type=int)
@click.option('--coco-path', type=str)
@click.option('--decoder-act', default='softmax', type=str, help='Activation function for the decoder MH cross-attention')
@click.option('--resume-from-checkpoint', default='', help='resume from checkpoint')
@click.pass_context
def main(ctx, seed, decoder_act, detr_config_file, coco_path, resume_from_checkpoint):
    cmd_params = ctx.params

    base_configs = load_base_configs()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr_config = build_detr_config(cmd_params['detr_config_file'], params=cmd_params, device=device)
    dataset_config = build_dataset_config(base_configs['dataset'], params=ctx.params)
    print("Dataset config\n", dataset_config)

    print("Building DETR model...")
    model, criterion, postprocessors = build_model(**detr_config)
    model.to(device)

    model_without_ddp = model
    describe_model(model_without_ddp)

    print("Load model from checkpoint...")
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint['model'])
    
    data_loader_val, base_ds = build_dataloaders(
        'val', dataset_config['coco_path'], dataset_config['batch_size'],
        False, dataset_config['num_workers']
    )
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device)

    stats = test_stats['coco_eval_bbox']
    metrics = dict(
       mAP=stats[0],
       AP_50=stats[1],
       AP_75=stats[2],
       AP_S=stats[3],
       AP_M=stats[4],
       AP_L=stats[5],
    )
    import pprint
    pprint.pprint(metrics)


if __name__ == "__main__":
    main()
