#!/bin/sh
echo "Executing script for Baseline Training"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

<<<<<<< HEAD
torchrun --nproc_per_node=4 $SCRIPT_DIR/train_baseline.py \
    --exp-name detr_baseline \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 8 --num-workers 12
    --epochs 1
=======
echo "Entering $SCRIPT_DIR"
pushd $SCRIPT_DIR

torchrun --nproc_per_node=6 train_baseline.py \
    --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 8 --num-workers 12

popd
>>>>>>> 24cc85746325a3f1d91d297257e02169b06282e8
