#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=8-00:00:00

# Note: 0-18:2%5 because we know seeds 3, 7, 9 get stuck

#SBATCH --array=0-18:2%5
#SBATCH --output=slurm_outs/Version_3/train_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Do the stuff
singularity exec project_baseline.simg python train.py ${SLURM_ARRAY_TASK_ID}
