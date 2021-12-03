import gym
import numpy as np


#
# Class concatenating several images to enable the agent to understand the motion of objects.
#
import torch


class BufferWrapper(gym.ObservationWrapper):

    def __init__(self, env, n_steps):
        """
        Constructor.
        :param env: the environment to wrap.
        :param n_steps: the number of steps to include in the buffer.
        """
        super().__init__(env)

        os = env.observation_space
        self.observation_space = \
            gym.spaces.Box(low=os.low.repeat(n_steps, axis=0), high=os.high.repeat(n_steps, axis=0), dtype=os.dtype)
        self.__buffer = np.zeros_like(self.observation_space.low)

    def reset(self):
        """
        Reset the environment.
        :return: the initial observation.
        """
        self.__buffer = np.zeros_like(self.observation_space.low)
        return self.observation(self.env.reset())

    def observation(self, obs):
        """
        Forget the first image that came in the internal buffer and add the new observation to this buffer.
        :param obs: the last observation.
        :return: the last n observations received.
        """
        self.__buffer[:-1] = self.__buffer[1:]
        self.__buffer[-1] = obs
        return self.__buffer
