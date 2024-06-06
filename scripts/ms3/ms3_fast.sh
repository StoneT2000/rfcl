#!/bin/bash

seeds=(1014 2937 5382 4785 7913)
demos=5
name_prefix="rfcl_fast"
env="peginsertion" # can be pickcube, stackcube, peginsertion, plugcharger
for seed in "${seeds[@]}"
do  
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms3/sac_ms3_${env}.yml \
        logger.exp_name="ms3/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.wandb=False \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=4000000
done