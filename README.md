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

We further provide docker images for each environment suite benchmarked for ease of use and deployment with all dependencies installed.

## Data / Demonstrations

We benchmark on 3 environment suites, each with their own demonstrations. We have uploaded all demonstrations to TODO
<!-- todo anon: use HF  -->

## Testing on New Environments

To test on your own custom environments or tasks from another suite (e.g. RoboMimic), all you need to do is create an `InitialStateWrapper` TODO