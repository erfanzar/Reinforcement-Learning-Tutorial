from jax import lax, numpy as jnp, random as jr
from utils import GenerateRNG
from tensorboardX import SummaryWriter
import multiprocessing as mp
import os
import tqdm

SEED = 42
NUM_TRIALS = 10000
EPS = 0.1


class BanditArm:
    def __init__(self, m: float):
        """
The __init__ function initializes the class with a mass, m.
It also sets up an estimate of the mass, m_estimate, and a counter for how many times we've called it.
The rng_gen is used to generate random numbers.

:param self: Represent the instance of the class
:param m: float: Set the mean of the distribution
        """
        self.m = m
        self.m_estimate = 0.
        self.N = 0.
        self.rng_gen = GenerateRNG(SEED)

    def pull(self):
        """
The pull function is a method of the class Bandit. It takes no arguments, and returns True or False based on whether
the bandit's arm was pulled. The probability that it will be pulled is determined by the bandit's mean value.

:param self: Access the attributes of the class
:return: A boolean value
        """
        return jr.normal(next(self.rng_gen)) < self.m

    def update(self, x):
        """
The update function is used to update the mean estimate of a running average.

:param self: Ensure that the update function can access the variables in the class
:param x: Update the mean estimate
:return: The new mean
        """
        self.N += 1
        self.m_estimate = (1. - 1. / self.N) * self.m_estimate + 1. / self.N * x


def run_tensorboard(name):
    """
The run_tensorboard function is a simple wrapper for the tensorboard command.
It takes in a name argument, which is the name of the directory where your
tensorboard logs are stored. It then runs tensorboard with that logdir as an
argument.

:param name: Specify the name of the directory where
    """
    os.system(
        f"tensorboard --logdir {os.getcwd()}/{name}"
    )


def main(): ...


if __name__ == "__main__":
    main()
