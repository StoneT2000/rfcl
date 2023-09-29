# Reverse Forward Curriculum Learning (RFCL)

Reverse Forward Curriculum Learning (RFCL) is a novel approach to learning from demonstrations that enables extreme **demonstration and sample efficiency** in model-free RL. RFCL is capable of solving a wide range of complex tasks from just 1-10 demonstrations, far more demonstration efficient than prior model-free baselines.

[Project Page](https://reverseforward-cl.github.io/) | [Paper](https://openreview.net/pdf?id=w4rODxXsmM) | [Open Review](https://openreview.net/forum?id=w4rODxXsmM)

<!-- todo anon: add in real names and links. Remove openreview -->
## Setup

We recommend using conda, and installing from source as so
```
conda create -n "rfcl" "python==3.9"
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

You may need to use `pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` if cuda 12 is not available.

To then install dependencies for environments, run

```
pip install gymnasium-robotics==1.2.3 mani_skill2==0.5.3 # for Adroit and ManiSkill2
```

We use the older/more stable metaworld environments running on old mujoco so we recommend doing this in a separate conda env
```
mamba install -c conda-forge mesalib glew glfw patchelf
pip install "cython<3"
pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
pip install shimmy[gym-v21]
```

We further provide docker images for each environment suite benchmarked for ease of use and deployment with all dependencies installed.

## Data / Demonstrations

We benchmark on 3 environment suites, each with their own demonstrations. We have uploaded all demonstrations to TODO. We recommend you directly download these demonstrations as opposed to trying to format them to include environmnet states as the code for that is quite complicated.
<!-- todo anon: use HF  -->

If you are interested in how the demonstrations are formatted, you can take a look at `scripts/demos/<env_suite>/format_dataset.py`. We take existing demonstrations from the environment suites and format them into the flexible [ManiSkill2 demonstration format](https://haosulab.github.io/ManiSkill2/concepts/demonstrations.html#format), which is used as this format supports storing environment states out of the box which is needed by RFCL. Some environment demonstrations (e.g. Adroit human demonstrations) do not come with environment states, so we wrote some fairly complex code to extract them.

## Training

To train, run any one of the scripts for each environment suite under `scripts/<env_suite>` (ManiSkill2: ms2, Adroit: adroit, Metaworld: metaworld). 

An example for training RFCL on ManiSkill2 (with the wall-time efficient hyperparameters) would be

```bash
demos=5
env="peginsertion" # can be pickcube, stackcube, peginsertion, plugcharger
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms2/sac_ms2_${env}.yml \
    logger.exp_name="ms2/${env}/rfcl_fast_${demos}_demos_s${seed}" \
    logger.wandb=True \
    train.num_demos=${demos} \
    seed=${seed} \
    train.steps=2000000
```

And for sample-efficient hyperparameters (higher update to data ratio) you would use the _sample_efficient.yml file instead

```bash
demos=5
env="peginsertion" # can be pickcube, stackcube, peginsertion, plugcharger
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py configs/ms2/sac_ms2_${env}_sample_efficient.yml \
    logger.exp_name="ms2/${env}/rfcl_sample_efficient_${demos}_demos_s${seed}" \
    logger.wandb=True \
    train.num_demos=${demos} \
    seed=${seed} \
    train.steps=2000000
```

See `configs/<env_suite>` for all configurations if you want to understand the exact configurations used (e.g. SAC + Q-Ensemble configs, environment configs etc.)

### Tuning tips for RFCL

There are a few very important hyperparameters to tune if you want better performance / something is not working.

**Things to consider if you want to be able to get any success**

- `sac.discount`: Discount factor affects all RL algorithms including RFCL. In general a discount of 0.9 to 0.99 works fine for most tasks.
- `train.reverse_step_size`: Values from 4 to 8 work fine for all benchmarked tasks, particularly because the demonstration data is slow/suboptimal. For demonstrations that are very optimal / fast, this can be tuned lower in order to reduce the gap between adjacent reverse curriculums.
- `train.demo_horizon_to_max_steps_ratio`: 
- reward function: RFCL is designed to solve sparse reward problems (+1 on success, 0 otherwise), but can also solve tasks faster if given dense rewards. As there are Q networks trained to predict Q-values, it is recommended to normalize rewards to the range [0, 1].
- demonstrations: RFCL is very robust to the demonstrations given, and can solve tasks even if the demonstrations are overly long, suboptimal, multimodal etc. But if you are able to collect demonstration data and care about success rate (as measured as being in a success state at the end of a episode), we recommend collecting demonstration data that do not stop upon success, but stops after achieving "stable success." An example for a cube picking task would be demonstrating picking up the cube, then holding it for a little bit before stopping the recording.

**Things to consider if you care about sample efficiency vs wall-time**
- `sac.grad_updates_per_step`: This controls the number of gradient updates performed every `env.num_envs * sac.steps_per_env` steps are taken. In our sample-efficient hyperparameters, we set this value to 80 to obtain a update to data ratio of 10 as `env.num_envs = 8` and `sac.steps_per_env = 1`. A high update to data ratio can easily improve sample efficiency, but will drastically worsen training time in simulation. A much lower update to data ratio of e.g. 0.5 as used in our wall-time efficient hyperparameters will worsen sample efficiency but drastically improve training time in simulation. Tune this to your needs!
- `sac.num_qs`: This by default is set to 10, meaning we keep an ensemble of 10 Q networks. This is a known trick for improving sample efficiency, although in some initial experiments we find this is not necessary when using a update to data ratio of 0.5. Setting this to 2 will make the configuration be equivalent to the original SAC implementation.


## Testing on New Environments

To test on your own custom environments or tasks from another suite (e.g. RoboMimic), all you need to do is create an `InitialStateWrapper` TODO. We only benchmark on 22 environments in this work, but an example of how to add RoboMimic is detailed in this tutorial TODO LINK