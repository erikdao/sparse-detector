#!/bin/sh
echo "Computing metrics on COCO validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/compute_val_metrics.py gini \
        --detr-config-file "configs/decoder_alpha_entmax.yml" \
        --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
        --batch-size 6 \
        --num-workers 24 \
        --decoder-act alpha_entmax # \
        # --resume-from-checkpoint "checkpoints/v2_decoder_a-entmax_alpha-lr=1e-3/checkpoint.pth"