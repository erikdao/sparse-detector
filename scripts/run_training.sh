#!/bin/sh
echo "Executing script for Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=3 $SCRIPT_DIR/train.py \
    --config configs/decoder_entmax.yml \
    --decoder-act "entmax15" \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 6 --num-workers 12 \
    --exp-name "entmax_alpha_test" \

