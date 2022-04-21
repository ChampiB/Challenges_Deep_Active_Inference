#
# Class that performs a greedy action selection.
#
class SelectBestAction:

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
        Select an action based on the quality of all actions.
        :param quality: a vector containing the quality of all actions.
        :param steps_done: the number of steps performed in the environment to date.
        :return: the selected action.
        """
        return quality.max(1)[1].item()
