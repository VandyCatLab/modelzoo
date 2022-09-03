#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=../logs/master/baseline/baseline_%a.out 
#SBATCH --mail-user=jason.k.chow@vanderbilt.edu
#SBATCH --mail-type=FAIL

# Timing 
date

# Do the stuff
singularity exec --nv ../pyTF.sif python ../python_scripts/baseline.py -a dropout -l 2,6,10 -i ${SLURM_ARRAY_TASK_ID}
singularity exec --nv ../pyTF.sif python ../python_scripts/baseline.py -a reflect -l 2,6,10 -i ${SLURM_ARRAY_TASK_ID}
singularity exec --nv ../pyTF.sif python ../python_scripts/baseline.py -a translate -l 2,6,10 -i ${SLURM_ARRAY_TASK_ID}

date