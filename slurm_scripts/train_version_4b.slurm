#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constrain=haswell 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00

#SBATCH --output=../logs/Version_4/train_slurm_%A.out 

echo "SLURM_JOBID: " $SLURM_JOBID

# Seeds 3, 5 and 9 crapped out early. Seed 10 in new set also crapped out early. Try 1 more seed.
singularity exec ../project_baseline.simg python ../python_scripts/train.py 13 
