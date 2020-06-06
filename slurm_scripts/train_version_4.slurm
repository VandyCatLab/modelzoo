#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH  -w, --nodelist=cn[1430-1477,1479-1537]
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00

#SBATCH --array=0-9%5
#SBATCH --output=slurm_outs/Version_4/train_slurm_%A_%a.out 

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Do the stuff
singularity exec project_baseline.simg python train.py ${SLURM_ARRAY_TASK_ID}
