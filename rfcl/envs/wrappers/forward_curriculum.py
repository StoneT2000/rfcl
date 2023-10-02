from collections import deque
from dataclasses import dataclass

import jax
import numpy as np
from gymnasium.vector import VectorEnvWrapper
from gymnasium.vector.vector_env import VectorEnv


def create_filled_deque(maxlen, fill_value):
    return deque([fill_value] * maxlen, maxlen=maxlen)

@dataclass
class SeedMetadata:
    seed: int
    returns: deque
    successes: deque

import itertools


def success_once_score(seed_metadata: SeedMetadata):
    returns = seed_metadata.returns
    size = 5
    returns = np.array(list(itertools.islice(returns, returns.maxlen - size, returns.maxlen)))
    success_once_avg = (returns > 0).sum() / size
    if success_once_avg == 0:
        return 2
    if success_once_avg > 0 and success_once_avg < 0.75:
        return 3
    return 1

score_fns = {
    "success_once_score": success_once_score,
}
class SeedBasedForwardCurriculumWrapper(VectorEnvWrapper):
    """
    Forward Curriculum Wrapper. Seed based similar to PLR. Code based off of https://github.com/facebookresearch/level-replay/blob/main/level_replay/level_sampler.py
    
    Parameters:
    seeds - List of seeds to select from for training. If None, default selects seeds 0 to num_seeds - 1
    
    """

    def __init__(
        self,
        env: VectorEnv,
        seeds=None,
        score_transform = "rankmin",
        score_temperature = 1e-1,
        staleness_transform = "rankmin",
        staleness_temperature = 1e-1,
        staleness_coef=1e-1,
        score_fn: str = "success_once_score",
        rho=0,
        nu=0.5,
        num_seeds=1000,
    ):
        super().__init__(env)

        self.seeds_db = dict()


        if seeds is None:
            seeds = np.arange(0, num_seeds)
        self.eps_seed_to_idx = dict()
        self.seeds = np.array(seeds)
        for idx, seed in enumerate(seeds):
            self.eps_seed_to_idx[seed] = idx
            self.seeds_db[seed] = SeedMetadata(seed=seed, returns=create_filled_deque(20, 0), successes=create_filled_deque(20, 0))
        self.np_random = np.random.RandomState()
        self.wrapper_seed = None
        self.rng_key = None

        self.seed_scores = np.zeros(len(seeds))
        self.unseen_seed_weights = np.ones(len(seeds))
        
        self.score_transform = score_transform
        self.score_temperature = score_temperature

        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.staleness_coef = staleness_coef
        self.seed_staleness = np.zeros(len(seeds))

        self.rho = rho
        self.nu = nu

        self.score_fn = score_fns[score_fn]

        # mark all seeds as "seen", slightly different to original implementation
        all_seed_indices = np.arange(len(self.seeds))
        from tqdm import tqdm
        for i in tqdm(all_seed_indices[::self.num_envs]):
            seed_indices = all_seed_indices[i:i+self.num_envs]
            if len(seed_indices) < self.num_envs:
                seed_indices = np.concatenate([seed_indices, np.array([0] * (self.num_envs - len(seed_indices)))])
            for i, reset_seed_idx in enumerate(seed_indices):
                self.unseen_seed_weights[reset_seed_idx] = 0
                self.seed_scores[reset_seed_idx] = 2

    def step_wait(self):
        
        next_observation, reward, terminations, truncations, info = super().step_wait()
        assert (
            truncations.all() == truncations.any()
        )  # Assume all envs finish at the same time in continuous task setting. # TODO modify for episodic later?
        if terminations.any() or truncations.any():
            for i, (final_info, exists) in enumerate(zip(info["final_info"], info["_final_info"])):
                if not exists:
                    continue # should not happen
                eps_seed = final_info["seed"]
                eps_seed_idx = self.eps_seed_to_idx[eps_seed]
                self.unseen_seed_weights[eps_seed_idx] = 0 # mark as seen
                seed_metadata: SeedMetadata = self.seeds_db[eps_seed]
                seed_metadata.returns.append(final_info["eps_ret"])
                seed_metadata.successes.append(final_info["success"])
                self.seed_scores[eps_seed_idx] = self.score_fn(seed_metadata)
            next_observation, _ = self.reset() # force our own reset with our own sampled seeds
        return next_observation, reward, terminations, truncations, info

    def sample_seeds(self, count):
        num_unseen_seeds = self.unseen_seed_weights.sum()
        num_seen_seeds = len(self.seeds) - num_unseen_seeds
        proportion_seen = num_seen_seeds / len(self.seeds)
        seen_seeds_mask = self.unseen_seed_weights == 0
        if num_unseen_seeds < len(self.seeds):
            edge_ct = (self.seed_scores[seen_seeds_mask] > 2).sum()
            within_ct = (self.seed_scores[seen_seeds_mask] < 2).sum()
            print(f"percent on edge of ability: {edge_ct} - {edge_ct / num_seen_seeds}, percent solved: {within_ct} - {within_ct / num_seen_seeds}")
        if num_unseen_seeds == 0:
            seed_indices = self._sample_seen_seeds(count)
        else:
            ps = self.np_random.rand(count)

            unseen_seed_indices = self._sample_unseen_seeds(count)
            if proportion_seen <= self.rho:
                # if <= rho percent of seeds have been seen, sample unseen seeds only
                seed_indices = unseen_seed_indices
            else:
                # with probability nu sample unseen seeds, with probability (1 - nu) sample seen seeds
                seen_seed_indices = self._sample_seen_seeds(count)
                seen_seed_indices = seen_seed_indices[ps >= self.nu]
                unseen_seed_indices = unseen_seed_indices[ps < self.nu]
                seed_indices = np.concatenate([unseen_seed_indices, seen_seed_indices])
            
        # update staleness scores
        self._update_staleness(seed_indices)
        return self.seeds[seed_indices], seed_indices
    
    def _update_staleness(self, seed_indices):
        for seed in seed_indices:
            self.seed_staleness += 1
            self.seed_staleness[seed] = 0

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            # basically equivalent to no curriculum
            weights = np.ones_like(scores)
        elif transform == 'identity':
            # note: not robust to score scale
            return scores
        elif transform == 'rank':
            # this did the best in the PLR paper. My intuition here is that the distance between scores does not mean a whole lot and is too noisy. 
            # Preferences/rankings work better here then. In addition to being robust to score scale (sort of like median)
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == "rankmin":
            # same as rank but equal ranking for the same scores
            def rankmin(x):
                # rankmin returns equal ranking for same value scores
                u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
                csum = np.zeros_like(counts)
                csum[1:] = counts[:-1].cumsum()
                return csum[inv]
            ranks = len(scores) - (rankmin(scores)) + 1
            weights = (1 / ranks) ** (1/temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)
        return weights

    def _sample_seen_seeds(self, count):
        weights = self._score_transform(self.score_transform, self.score_temperature, self.seed_scores)
        weights = weights * (1 - self.unseen_seed_weights)
        z = weights.sum()
        if z > 0: weights = weights / z
        staleness_weights = 0

        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            z = staleness_weights.sum()
            if z > 0: 
                staleness_weights = staleness_weights / z
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights
        seed_indices = self.np_random.choice(range(len(self.seeds)), size=count, p=weights / weights.sum())
        return seed_indices

    def _sample_unseen_seeds(self, count):
        num_unseen_seeds = self.unseen_seed_weights.sum()
        assert num_unseen_seeds > 0
        seed_indices = self.np_random.choice(range(len(self.seeds)), size=count, p=self.unseen_seed_weights / num_unseen_seeds)
        return seed_indices

    def reset(self, seed = None):
        # seed=None means to sample from env's groundtruth initial state distribution. Otherwise we can pick one from our database
        if seed is not None:
            assert self.wrapper_seed is None, "should only seed curriculum wrapper once"
            # first call to reset the wrapper
            self.wrapper_seed = seed
            self.np_random = np.random.RandomState(seed=seed)
            self.rng_key = jax.random.PRNGKey(seed=self.np_random.randint(0, 2**31 - 1))
        seeds, seed_indices = self.sample_seeds(self.num_envs)
        seeds = seeds.tolist()
        seed_indices = seed_indices.tolist()
        return super().reset(seed=seeds)