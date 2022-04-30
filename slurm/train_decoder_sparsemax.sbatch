#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_sparsemax.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_sparsemax.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Decoder with Sparsemax (256 queries) in the Multihead Cross-attention"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/train.py \
    --config "configs/decoder_sparsemax_256queries_lr.yml" \
    --decoder-act "sparsemax" \
    --coco-path data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 6 --num-workers 24 \
    --exp-name "decoder_sparsemax_256q_lr" \
    --wandb-group "decoder_sparsemax"
