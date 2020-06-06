#!/bin/bash
#SBATCH --ntasks=1
#SBATCH  -w, --nodelist=cn[1403]
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00

# Do the stuff
singularity exec ../project_baseline.simg python test_shuffle.py 1 
