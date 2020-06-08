#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:18:00
#SBATCH --array=0-2
#SBATCH --output=sanity_check/train_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Do the stuff
singularity exec ../project_baseline.simg python ../python_scripts/train.py 1
