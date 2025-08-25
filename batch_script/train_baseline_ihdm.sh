#!/bin/bash 
#SBATCH -o batch_output/train_baseline_ihdm/job_%j.output
#SBATCH -e batch_error/train_baseline_ihdm/job_%j.error
#SBATCH -p RTXA6Kq
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH -c 2

module load cuda11.1/toolkit 
module load cuda11.1/blas/11.1.1 
source activate scienceh100
python train.py --config configs/gopro64/img_size64.py --workdir runs/gopro/64

srun -p RTXA6Kq --gres=gpu:1 python train.py --config configs/gopro64/img_size64.py --workdir runs/gopro/64

#SBATCH -w node14
