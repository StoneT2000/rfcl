#!/bin/bash

seeds=(10)
demos=10
name_prefix="sample_efficient"
env="pickcube" # can be pickcube, stackcube, peginsertion, plugcharger
# for plugcharger you usually need about 10M+ steps to solve the task with 30 demos, and about 12M+ to solve the task with 10 demos.
for seed in "${seeds[@]}"
do  
    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/experiment_reverse_sac_full.py \
        configs/reverse/ms2/sac_ms2_${env}_sample_efficient.yml \
        logger.exp_name="ms2/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.clear_out=True logger.wandb=True logger.wandb_cfg.entity="reverse-rl" logger.wandb_cfg.group="Test" \
        train.shuffle_demos=True train.num_demos=${demos} seed=${seed} train.steps=1000000        
done