import gym
import numpy as np
from PIL import Image


#
# Scale the image down to a specify size and turn RBG to grayscale.
#
class ProcessFrameWrapper(gym.ObservationWrapper):

    def __init__(self, env, image_shape=(3, 84, 84)):
        """
        Constructor.
        :param env: the environment to wrap.
        :param image_shape: the shape of the image to generate.
        """
        super().__init__(env)

        self.__image_shape = image_shape
        self.observation_space = \
            gym.spaces.Box(low=0, high=255, shape=(1, image_shape[1], image_shape[2]), dtype=np.uint8)

    def observation(self, obs):
        """
        Pre-process the observation.
        :param obs: the observation to pre-process.
        :return: the pre-processed observation.
        """
        img = obs.astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((self.__image_shape[1], self.__image_shape[2]), Image.ANTIALIAS)
        img = np.reshape(np.array(img), [1, self.__image_shape[1], self.__image_shape[2]])
        return img.astype(np.uint8)
