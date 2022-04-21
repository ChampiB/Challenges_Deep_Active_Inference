import numpy as np
import math
import random


#
# Class that performs a random action selection.
#
class EpsilonGreedyActionSelection:

    def __init__(self, epsilon_start, epsilon_end, epsilon_decay, **_):
        """
        Construct a epsilon-greedy action selection strategy.
        :param epsilon_start: the initial value of epsilon.
        :param epsilon_end: the final value of epsilon.
        :param epsilon_decay: how slowly should epsilon decay? The bigger, the slower.
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def __iter__(self):
        """
        Make the class iterable.
        :return: the next key and value.
        """
        for key, value in {
            "module": str(self.__module__),
            "class": str(self.__class__.__name__),
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }.items():
            yield key, value

    def select(self, quality, steps_done):
        """
        Select an action by according to an epsilon greedy scheme.
        :param quality: a vector containing the quality of all actions.
        :param steps_done: the number of steps performed in the environment to date.
        :return: the selected action.
        """
        # Compute the current epsilon value.
        epsilon_threshold = \
            self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * steps_done / self.epsilon_decay)

        # Sample a number between 0 and 1, and either execute a random action or
        # the reward maximizing action according to the sampled value.
        if random.random() > epsilon_threshold:
            return quality.max(1)[1].item()
        return np.random.choice(quality.shape[1])
