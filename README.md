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

## Testing on New Environments

To test on your own custom environments or tasks from another suite (e.g. RoboMimic), all you need to do is create an `InitialStateWrapper` TODO. We only benchmark on 22 environments in this work, but an example of how to add RoboMimic is detailed in this tutorial TODO LINK