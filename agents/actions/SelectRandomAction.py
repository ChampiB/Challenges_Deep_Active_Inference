import numpy as np


#
# Class that performs a random action selection.
#
class SelectRandomAction:

    def __init__(self, **_):
        pass

    def __iter__(self):
        """
        Make the class iterable.
        :return: the next key and value.
        """
        for key, value in {
            "module": str(self.__module__),
            "class": str(self.__class__.__name__)
        }.items():
            yield key, value

    def select(self, quality, steps_done):
        """
        Select a random action.
        :param quality: a vector containing the quality of all actions (unused).
        :param steps_done: the number of steps performed in the environment to date.
        :return: the selected action.
        """
        return np.random.choice(quality.shape[1])
