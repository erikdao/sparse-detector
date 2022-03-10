#!/usr/bin/env bash
#SBATCH -n 72
#SBATCH --gpus 6
#SBATCH --time 72:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user cuongdd@kth.se
#SBATCH --account berzelius-2022-50
#SBATCH --error /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_refactor.err
#SBATCH --output /proj/azizpour-group/users/cuongdao/projects/sparse-detector/logs/%J_refactor.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
module load Anaconda/2021.05-nsc1
conda activate /proj/azizpour-group/users/cuongdao/.conda/sparsedet
nvidia-smi

SEED=42
echo "DETR Baseline"
echo "Seed $SEED"
python -m torch.distributed.launch --nproc_per_node=6 --use_env \
    main.py --coco_path /proj/azizpour-group/users/cuongdao/data/COCO \
        --output_dir checkpoints --seed $SEED \
        --batch_size 8 --num_workers 12


python -m torch.distributed.launch --nproc_per_node=6 --use_env \
    main.py --coco_path /proj/azizpour-group/users/cuongdao/data/COCO \
        --output_dir checkpoints --seed 42 \
        --batch_size 8 --num_workers 12