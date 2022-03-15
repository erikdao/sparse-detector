#!/bin/sh
echo "Executing script for Baseline Evaluation"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/eval_baseline.py \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --batch-size 6 \
    --num-workers 12 \
    --checkpoint checkpoints/baseline_detr/checkpoint.pth
