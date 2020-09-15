#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constrain=haswell 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=3-00:00:00

#SBATCH --output=../outputs/baseline.out 

# Do the stuff
singularity exec ../pyKeras2.sif python ../python_scripts/accre_baseline.py
