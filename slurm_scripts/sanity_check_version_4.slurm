#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constrain=haswell 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00
#SBATCH --output=../logs/Version_4/sanity_check.out 

# Sanity check, compare to version 4 seed 0 because all other references are skylake or sandy
singularity exec ../project_baseline.simg python ../python_scripts/train.py 0 
