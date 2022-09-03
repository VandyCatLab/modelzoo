#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-99
#SBATCH --output=../logs/master/train_slurm_%a.out 
#SBATCH --mail-user=jason.k.chow@vanderbilt.edu
#SBATCH --mail-type=FAIL

singularity exec --nv ../pyTF.sif python ../python_scripts/train.py -i ${SLURM_ARRAY_TASK_ID}
