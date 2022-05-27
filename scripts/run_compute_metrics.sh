#!/bin/sh
echo "Computing metrics on COCO validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/compute_val_metrics.py zeros_ratio \
    --detr-config-file "configs/decoder_entmax15_baseline.yml" \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --batch-size 6 \
    --num-workers 24 \
    --decoder-act entmax15 \
    --resume-from-checkpoint checkpoints/decoder_entmax_cross-mha/checkpoint.pth