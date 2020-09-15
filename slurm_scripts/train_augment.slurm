#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

#SBATCH --output=../logs/train_augment.log

singularity exec ../pyKeras2.sif python ../python_scripts/train_augment.py -w 0
