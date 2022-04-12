#!/bin/sh
echo "Generating predictions for evaluation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/generate_predictions.py \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --batch-size 6 \
    --num-workers 24 \
    --decoder-act sparsemax \
    --resume-from-checkpoint checkpoints/decoder_sparsemax_cross-mha/checkpoint.pth
