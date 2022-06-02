#!/bin/sh
echo "Computing metrics on COCO validation set"

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/scripts
SEED=42

# metricthreshold=(1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10)

# for m in "${metricthreshold[@]}"
# do
#     echo "Computing metric with threshold $m"
#     torchrun --nproc_per_node=8 $SCRIPT_DIR/compute_val_metrics.py zeros_ratio \
#         --detr-config-file "configs/detr_baseline.yml" \
#         --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
#         --batch-size 6 \
#         --num-workers 24 \
#         --decoder-act softmax \
#         --metric-threshold $m
# done

torchrun --nproc_per_node=8 $SCRIPT_DIR/compute_val_metrics.py zeros_ratio \
        --detr-config-file "configs/decoder_entmax15_baseline.yml" \
        --coco-path /proj/azizpour-group/users/cuongdao/data/COCO \
        --batch-size 6 \
        --num-workers 24 \
        --decoder-act entmax15