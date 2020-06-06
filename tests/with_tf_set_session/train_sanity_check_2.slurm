#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

#SBATCH --array=0-14%5
#SBATCH --output=sanity_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Do the stuff, test 5 instances of 3 seeds, make sure it is the same
singularity exec ../../project_baseline.simg python train_sanity_check_2.py $(($SLURM_ARRAY_TASK_ID % 3))
