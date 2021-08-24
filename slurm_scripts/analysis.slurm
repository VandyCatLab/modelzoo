#!/bin/bash
#SBATCH --account=palmeri_lab
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=5:00:00
#SBATCH --array=0-1
#SBATCH --output=../logs/master/correspondence/correspondence_%a.out 
#SBATCH --mail-user=jason.k.chow@vanderbilt.edu
#SBATCH --mail-type=FAIL

# Timing 
date

# Do the stuff
singularity exec --nv ../pyTF.sif python ../python_scripts/analysis.py -a correspondence -i ${SLURM_ARRAY_TASK_ID}

date