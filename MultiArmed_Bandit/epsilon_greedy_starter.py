from jax import lax, numpy as jnp, random as jr
from utils import GenerateRNG
from tensorboardX import SummaryWriter
import multiprocessing as mp
import os
import tqdm

SEED = 42
NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p: float):
        """
The __init__ function is called when the class is instantiated.
It sets up the initial state of the object, and defines any variables that will be used by all methods in this class.
In this case, we are setting up a Bernoulli distribution with parameter p (the probability of success),
and defining two variables to keep track of our estimate for p and how many trials have been run so far.

:param self: Represent the instance of the class
:param p: float: Set the probability of success
:return: Nothing, it just initializes the class
        """
        self.p = p
        self.p_estimate = 0.
        self.N = 0.
        self.rng_gen = GenerateRNG(SEED)

    def pull(self):
        """
The pull function is a method of the class Bandit. It takes no arguments, and returns True or False based on whether
the bandit was pulled (True) or not (False). The probability that it will be pulled is determined by the p attribute
of the instance.

:param self: Refer to the object itself
:return: A boolean value
        """
        return jr.normal(next(self.rng_gen)) < self.p

    def update(self, x):
        """
The update function takes in a new data point and updates the current estimate of p.

:param self: Refer to the object itself
:param x: Update the p_estimate
:return: nothing it will just update The probability estimate
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def run_tensorboard(name):
    """
    The run_tensorboard function is a simple wrapper for the tensorboard command.
    It takes in a name argument, which is the name of the directory where your
    tensorboard logs are stored. It then runs tensorboard with that logdir as an
    argument.

    :param name: Specify the name of the folder where the tensorboard logs are stored

    """
    os.system(
        f"tensorboard --logdir {os.getcwd()}/{name}"
    )


def experiment():
    """
The experiment function is the main function of this file.
It runs a multi-armed bandit experiment with epsilon greedy strategy.
The experiment is run for NUM_TRIALS times and the rewards are recorded in an array.
The mean estimate of each bandit is also recorded at every trial and plotted using tensorboardX library.

    """
    summery_writer = SummaryWriter(
        "epsilon_greedy_starter_tensorboard",
        comment="MultiArmed_bandit_epsilon_greedy_starter"
    )
    process = mp.Process(
        target=run_tensorboard,
        kwargs={"name": "epsilon_greedy_starter_tensorboard"}
    )
    process.start()
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rng_gen = GenerateRNG(SEED)
    rewards = jnp.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = jnp.argmax(jnp.array([b.p for b in bandits]))
    for i in tqdm.trange(NUM_TRIALS):
        if jr.normal(next(rng_gen), (1,))[0] < EPS:
            num_times_explored += 1
            j = jr.randint(key=next(rng_gen), shape=(1,), maxval=len(bandits), minval=0)[0]
        else:
            num_times_exploited += 1
            j = jnp.argmax(jnp.array([b.p_estimate for b in bandits]))
        if j == optimal_j:
            num_optimal += 1
        x = bandits[j].pull()
        rewards = rewards.at[i].set(x)
        bandits[j].update(x)
        for ea, b in enumerate(bandits):
            summery_writer.add_scalar(
                f"Bandits/Bandit {ea} Mean Estimate",
                b.p_estimate,
                i
            )

        summery_writer.add_scalar(
            "Scalars/Total Reward earned",
            rewards.sum(),
            i
        )

        summery_writer.add_scalar(
            "Scalars/Overall win rate",
            rewards.sum() / i + 1,
            i
        )

        summery_writer.add_scalar(
            "Scalars/Num Times Explored",
            num_times_explored,
            i
        )
        summery_writer.add_scalar(
            "Scalars/Num Times Exploited",
            num_times_exploited,
            i
        )
    for b in bandits:
        print("Mean Estimate : ", b.p_estimate)

    print("Total Reward earned : ", rewards.sum())
    print("Overall win rate    : ", rewards.sum() / NUM_TRIALS)
    print("Num Times Explored  : ", num_times_explored)
    print("Num Times Exploited : ", num_times_exploited)
    print("Num Times Selected Optimal Bandit : ", num_optimal)
    process.join()


def main():
    experiment()


if __name__ == "__main__":
    main()
