#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nnodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH  -w, --nodelist=cn[301-320,322-324,326-338,340,347-396,398-401,403-405]
#SBATCH --output=group_1__%A_%a.out 

singularity exec ../../project_baseline.simg python train.py 0 
