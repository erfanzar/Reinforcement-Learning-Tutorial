from jax import random


class GenerateRNG:
    def __init__(self, seed: int = 42):
        self.key = random.PRNGKey(seed)

    def __next__(self):
        self.key, rng = random.split(self.key)
        return rng
