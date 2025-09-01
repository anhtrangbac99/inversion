#!/bin/bash 
#SBATCH -o batch_output/train_timestep/job_%j.output
#SBATCH -e batch_error/train_timestep/job_%j.error
#SBATCH -p PA100q
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH -c 2

module load cuda11.1/toolkit 
module load cuda11.1/blas/11.1.1 
source activate scienceh100
python train_timestep.py --config configs/gopro128/default_gopro128_configs.py --workdir runs/gopro/timestep_128_100steps_lr2e5 --from_start


srun -p PA100q --gres=gpu:1 python train_timestep.py --config configs/gopro128/default_gopro128_configs.py --workdir runs/gopro/timestep_128 --train_sharp

#SBATCH -w node14
