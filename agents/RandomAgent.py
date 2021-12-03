import numpy as np


#
# Implement a agent acting randomly.
#
class RandomAgent:

    def __init__(self, n_actions):
        """
        Constructor.
        :param n_actions: the number of available actions.
        """
        self.n_actions = n_actions

    def step(self, obs):
        """
        Select a random action.
        :param obs: unused.
        :return: the random action.
        """
        return np.random.choice(self.n_actions)
