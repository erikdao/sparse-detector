#!/bin/sh
echo "Computing metrics on COCO validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=1 $SCRIPT_DIR/compute_val_metrics.py paibb \
        --detr-config-file "configs/decoder_sparsemax_baseline.yml" \
        --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
        --batch-size 6 \
        --num-workers 24 \
        --decoder-act sparsemax \
        --resume-from-checkpoint "checkpoints/v2_decoder_sparsemax/checkpoint.pth"