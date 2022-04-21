from torch import softmax
from torch.distributions.categorical import Categorical


#
# Class that performs a random action selection.
#
class SoftmaxActionSelection:

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
        Select an action by sampling a softmax function of the quality.
        :param quality: a vector containing the quality of all actions.
        :param steps_done: the number of steps performed in the environment to date.
        :return: the selected action.
        """
        return Categorical(softmax(quality, dim=1)).sample()
