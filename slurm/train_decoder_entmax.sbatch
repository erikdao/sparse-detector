#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_entmax.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_entmax.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Decoder with entmax15 in the Multihead Cross-attention"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines/
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/train.py \
    --config "configs/decoder_entmax.yml" \
    --decoder-act "entmax15" \
    --coco-path data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 6 --num-workers 12 \
    --exp-name "decoder_entmax_cross-mha" \
    --wandb-group "decoder_entmax" \
    --wandb-id "2gho4voo" \
    --resume-from-checkpoint "checkpoints/decoder_entmax_cross-mha/checkpoint.pth"
