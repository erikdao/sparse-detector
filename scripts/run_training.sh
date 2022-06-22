#!/bin/sh
echo "Executing script for Baseline Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/train.py \
    --exp-name "decoder_sparsemax_local" \
    --detr-config-file "configs/decoder_sparsemax_baseline.yml" \
    --decoder-act "alpha_entmax" \
    --coco-path data/COCO \
    --output-dir checkpoints \
    --seed $SEED \
    --epochs 2 \
    --no-wandb-log
