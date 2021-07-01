#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9
#SBATCH --output=../logs/master/train_slurm_${SLURM_ARRAY_TASK_ID}.out 

singularity exec --nv ../pyTF.sif python ../python_scripts/train.py -s 10 -w ${SLURM_ARRAY_TASK_ID}
