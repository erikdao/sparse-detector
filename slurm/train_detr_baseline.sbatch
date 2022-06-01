#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 24:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_baseline_detr.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_baseline_detr.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/train.py \
    --exp-name "v2_baseline_detr" \
    --detr-config-file "configs/detr_baseline.yml" \
    --decoder-act "softmax" \
    --coco-path data/COCO \
    --output-dir checkpoints \
    --seed $SEED \
    --wandb-id "19ie75vp" \
    --resume-from-checkpoint "checkpoints/v2_baseline_detr/checkpoint_0229.pth" 