#!/bin/sh
echo "Computing gini scores on validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/compute_gini.py \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --batch-size 6 \
    --num-workers 24 \
    --decoder-act softmax \
    --resume-from-checkpoint checkpoints/original_detr/detr-r50.pth
