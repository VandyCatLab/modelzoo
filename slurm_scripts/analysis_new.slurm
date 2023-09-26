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
while [[ $n -lt 50 ]];
do python3 ../python_scripts/hubReps.py --analysis reps --data_dir /Users/david/PycharmProjects/NeuralNet/modelnet/stimuli --models_file /Users/david/PycharmProjects/NeuralNet/modelnet/python_scripts/hubModels_pytorch.json --dataset novset --rep_name reps_pytorch -i $n;
n=$((n+1));
done

date