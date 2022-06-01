#!/bin/sh
echo "Computing metrics on COCO validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=8 $SCRIPT_DIR/compute_val_metrics.py zeros_ratio \
    --detr-config-file "configs/detr_baseline.yml" \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --batch-size 6 \
    --num-workers 24 \
    --decoder-act softmax \
    --metric-threshold 1e-6 # \
    # --resume-from-checkpoint checkpoints/v2_baseline_detr/checkpoint_0279.pth