#!/bin/sh
echo "Executing script for Baseline Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/train.py \
    --exp-name "decoder_alpha_entmax_local" \
    --detr-config-file "configs/decoder_alpha_entmax.yml" \
    --decoder-act "alpha_entmax" \
    --coco-path data/COCO \
    --output-dir checkpoints \
    --resume-from-checkpoint "checkpoints/v2_decoder_a-entmax_alpha-lr=1e-3/checkpoint_0199.pth" \
    --seed $SEED \
    --epochs 202 \
    --no-wandb-log
