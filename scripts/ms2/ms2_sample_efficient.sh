#!/bin/bash

seeds=(1014 2937 5382 4785 7913)
demos=5
name_prefix="rfcl_sample_efficient"
env="pickcube" # can be pickcube, stackcube, peginsertion, plugcharger
# for plugcharger you usually need about 10M+ steps to solve the task with 30 demos, and about 12M+ to solve the task with 10 demos.
for seed in "${seeds[@]}"
do  
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms2/sac_ms2_${env}_sample_efficient.yml \
        logger.exp_name="ms2/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.wandb=True \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=2000000
done