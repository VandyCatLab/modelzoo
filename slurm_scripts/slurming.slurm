#!/bin/bash
#SBATCH --account=psychology_gpu_acc
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=pascal 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=1
#SBATCH --array=0
#SBATCH --output=../logs/master/hubReps/reps_keras/hubRep_%a.out
#SBATCH --mail-user=david.m.brust@vanderbilt.edu
#SBATCH --mail-type=FAIL


# Timing 
date

n=0;
while [[ $n -lt 705 ]];
do python3 ../python_scripts/analysis.py --analysis correspondence --reps_dir /Users/david/PycharmProjects/NeuralNet/modelnet/hubReps/all_reps --output_dir /Users/david/PycharmProjects/NeuralNet/modelnet/correspondences -i $n;
n=$((n+1));
done

date