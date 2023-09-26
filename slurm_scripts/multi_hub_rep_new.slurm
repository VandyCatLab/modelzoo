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

n=0;s
while [[ $n -lt 784 ]];
do python3 ../python_scripts/multi_hub_rep_new.py -t many_odd --reps True -i $n;
n=$((n+1));
done

date

date

n=0;s
while [[ $n -lt 784 ]];
do python3 ../python_scripts/multi_hub_rep_new.py -t threeACF --reps True -i $n;
n=$((n+1));
done

date

date

n=0;s
while [[ $n -lt 784 ]];
do python3 ../python_scripts/multi_hub_rep_new.py -t learn_exemp --reps True -i $n;
n=$((n+1));
done

date