import gym
import numpy as np


#
# Class normalizing the pixels value to force them between zero and one.
#
class NormalizePixelsWrapper(gym.ObservationWrapper):

    def observation(self, observation):
        """
        Scale the pixels value to force them between zero and one.
        :param observation: the input observation.
        :return: the scaled observation.
        """
        return np.array(observation).astype(np.float32) / 255.0
