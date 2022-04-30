#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_entmax_alpha.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_decoder_entmax_alpha.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo "Decoder with alpha-entmax in the Multihead Cross-attention"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

PROJ_DIR=/proj/azizpour-group/users/cuongdao/projects/sparse-detector
SCRIPT_DIR=$PROJ_DIR/sparse_detector/engines
SEED=42

torchrun --nproc_per_node=6 $SCRIPT_DIR/train.py \
    --config "configs/decoder_entmax_alpha.yml" \
    --decoder-act "entmax_alpha" \
    --coco-path data/COCO \
    --output-dir checkpoints --seed $SEED \
    --batch-size 6 --num-workers 24 \
    --exp-name "decoder_entmax_alpha_cross-mha"
