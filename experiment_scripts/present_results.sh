#!/bin/bash


cd ..

# path the configuration file
config_path=experiment_configs/resnet50_imagenet200.json


python ens_table.py $config_path 5 --latex 0 
python plot_ens_uncs_conditional.py $config_path