#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH  -w, --nodelist=cn1400
#SBATCH --output=group_3a__%A_%a.out 

singularity exec ../../project_baseline.simg python train.py 0 
