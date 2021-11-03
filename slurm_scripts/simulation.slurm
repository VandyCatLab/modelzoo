#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH --array=0-99
#SBATCH --output=../logs/master/simulation/noiseSim_%a.out 
#SBATCH --mail-user=jason.k.chow@vanderbilt.edu
#SBATCH --mail-type=FAIL

# Timing 
date

# Do the stuff
singularity exec ../pyTF.sif python ../python_scripts/simulation.py -a noise -s ${SLURM_ARRAY_TASK_ID}

date