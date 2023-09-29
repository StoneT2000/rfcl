#!/bin/bash

seeds=(10)
demos=10
name_prefix="fast"
env="pickcube" # can be pickcube, stackcube, peginsertion, plugcharger
for seed in "${seeds[@]}"
do  
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms2/sac_ms2_${env}.yml \
        logger.exp_name="ms2/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.clear_out=True \
        logger.wandb=True \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=10000000
done