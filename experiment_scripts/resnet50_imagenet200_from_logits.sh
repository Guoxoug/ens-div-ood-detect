#!/bin/bash


cd ..

# path the configuration file
config_path=experiment_configs/resnet50_imagenet200.json

for num in $(seq 1 1 5)
do
    python eval_logits_features.py $config_path --seed $num --gpu 0
done

python ens_results_from_logits.py $config_path 