import gym
import numpy as np
from torch import from_numpy


#
# Class turning the images into the format required by pytorch.
#
class ToPytorchFormatWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Constructor.
        :param env: the environment to be wrapped.
        """
        super().__init__(env)

        # Update observation space
        os = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation(os.low), self.observation(os.high), dtype=os.dtype)

    def observation(self, obs):
        """
        Swap the axis to match Pytorch default shape.
        :param obs: the observation to pre-process.
        :return: the pre-processed observation
        """
        return np.moveaxis(obs, 2, 0)
