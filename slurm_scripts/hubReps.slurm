#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=../logs/master/hubReps/hubRep_%a.out 
#SBATCH --mail-user=jason.k.chow@vanderbilt.edu
#SBATCH --mail-type=FAIL

# Timing 
date

# Do the stuff
singularity exec --nv ../pyTF.sif python ../python_scripts/hubReps.py -i ${SLURM_ARRAY_TASK_ID} -d /scratch/chowjk/tensorflow_datasets -f ~/idvor/python_scripts/hubModels.json

date