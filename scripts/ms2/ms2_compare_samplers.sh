#!/bin/bash
# same script as ms2_sample_efficient but adds three lines to do only stage 1 training, 
# and evaluate the policy from the initial state of the given demos only

seeds=(1014)
samplers=("uniform" "geometric" "uniform_spike")
demos=5
name_prefix="rfcl_sample_efficient_sampler_ablations"
env="peginsertion" # can be pickcube, stackcube, peginsertion, plugcharger
# for plugcharger you usually need about 10M+ steps to solve the task with 30 demos, and about 12M+ to solve the task with 10 demos.
for seed in "${seeds[@]}"
do  
  for sampler in "${samplers[@]}"
  do
    name_prefix="rfcl_sample_efficient_${sampler}_sampler_ablations"
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms2/sac_ms2_${env}_sample_efficient.yml \
        logger.exp_name="ms2/${env}/${name_prefix}_${demos}_demos_s${seed}" \
        logger.wandb=True logger.wandb_cfg.group="RFCL-ManiSkill2-SamplerAblations" \
        train.num_demos=${demos} \
        seed=${seed} \
        train.steps=2000000 \
        train.start_step_sampler="${sampler}" \
        train.use_orig_env_for_eval=False \
        train.eval_start_of_demos=True \
        stage_1_only=True
  done
done