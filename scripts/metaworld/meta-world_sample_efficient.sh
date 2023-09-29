#!/bin/bash

seeds=(1014 2937 5382 4785 7913)
demos=5
name_prefix="rfcl_sample_efficient"
env="stick-pull-v2-goal-observable"
for seed in "${seeds[@]}"
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/meta-world/base_sac_metaworld_sample_efficient.yml \
        logger.exp_name="metaworld/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.wandb=True \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=1000000 \
        env.env_id="${env}" \
        train.dataset_path="demos/meta-world/${env}/trajectory.h5"
done