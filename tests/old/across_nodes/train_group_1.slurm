#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nnodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-1:00:00
#SBATCH  -w, --nodelist=cn[1430-1477,1479-1537]
#SBATCH --output=group_1_%A_%a.out 

singularity exec ../../project_baseline.simg python train.py 0 
