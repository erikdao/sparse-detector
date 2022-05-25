#!/bin/sh
echo "Executing script for Baseline Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/train.py \
    --detr-config-file "configs/detr_baseline.yml" \
    --decoder-act "sparsemax" \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 6 --num-workers 24 \
    --exp-name "config_test" \
    --epochs 4 \
    --no-wandb-log
