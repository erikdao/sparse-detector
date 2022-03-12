#!/bin/sh
echo "Executing script for Baseline Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

torchrun --nproc_per_node=4 $SCRIPT_DIR/train_baseline.py \
    --exp-name detr_baseline \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 8 --num-workers 12
    --epochs 1