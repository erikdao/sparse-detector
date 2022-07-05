#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_alpha_entmax.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_alpha_entmax.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Continue alpha-entmax lr_alpha=1e-3 without decaying lr_alpha"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/train.py \
    --exp-name "v2_decoder_alpha_entmax_no-decay-lr_alpha" \
    --detr-config-file "configs/decoder_alpha_entmax.yml" \
    --decoder-act "alpha_entmax" \
    --coco-path data/COCO \
    --output-dir checkpoints \
    --resume-from-checkpoint "checkpoints/v2_decoder_a-entmax_alpha-lr=1e-3/checkpoint_0199.pth" \
    --seed $SEED \
    --wandb-group "decoder_alpha_entmax"
