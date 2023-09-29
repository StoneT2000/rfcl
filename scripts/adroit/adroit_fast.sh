#!/bin/bash

seeds=(1014 2937 5382 4785 7913)
demos=5
name_prefix="rfcl_fast"
env="adroitdoor" # can be adroitdoor, adroitpen, adroitrelocate
for seed in "${seeds[@]}"
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/adroit/sac_adroit_${env}.yml \
        logger.exp_name="adroit/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.wandb=True \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=4000000
done