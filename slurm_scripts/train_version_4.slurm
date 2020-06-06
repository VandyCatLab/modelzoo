#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --nnodes=1
#SBATCH --constrain=haswell 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00

#SBATCH --array=0-9
#SBATCH --output=../logs/Version_4/train_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Do the stuff
singularity exec ../project_baseline.simg python ../python_scripts/train.py ${SLURM_ARRAY_TASK_ID}
