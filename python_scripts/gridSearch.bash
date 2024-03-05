set -euo pipefail

# Grid search for many_odd
for noise in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0
do 
    for encNoise in 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25 5.5 5.75 6.0
    do 
        python multi_hub_rep.py --test many_odd --noise $noise --encoding_noise $encNoise
    done
done

# Grid search for threeAFC models
for noise in 0.05 0.1 0.15 0.2 0.25 0.5 0.75 1.0 1.25 1.5
do 
    for encNoise in 0.25 0.5 0.75 1.0 1.25 1.5
    do
        for learningAdv in 0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25
        do 
            python multi_hub_rep.py --test threeAFC --noise $noise --encoding_noise $encNoise --learning_adv $learningAdv
        done
    done
done

# Grid search for LE models
for noise in 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0
do 
    for learningAdv in 0 0.01 0.015 0.02 0.025 0.05 0.075 0.1
    do
        python multi_hub_rep.py --test learn_exemp --noise $noise --learning_adv $learningAdv
    done
done
