#!/bin/bash 
#SBATCH -o batch_output/train_baseline_ihdm/job_%j.output
#SBATCH -e batch_error/train_baseline_ihdm/job_%j.error
#SBATCH -p PA100q
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -c 2

module load cuda11.1/toolkit 
module load cuda11.1/blas/11.1.1 
source activate scienceh100
python train.py --config configs/gopro128/default_gopro128_configs.py --workdir runs/gopro/128_100steps_sharp_lr2e5 --from_start --train_sharp

srun -p PA100q --gres=gpu:1 python train.py --config configs/gopro128/default_gopro128_configs.py --workdir runs/gopro/128_sharp --train_sharp

#SBATCH -w node14
