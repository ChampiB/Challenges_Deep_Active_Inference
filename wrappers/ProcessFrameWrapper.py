import gym
from skimage.transform import resize
import numpy as np


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
            gym.spaces.Box(low=0, high=255, shape=(image_shape[1], image_shape[2], 1), dtype=np.uint8)

    def observation(self, obs):
        """
        Pre-process the observation.
        :param obs: the observation to pre-process.
        :return: the pre-processed observation.
        """
        if obs.size == 210 * 160 * 3:
            img = np.reshape(obs, [210, 160, 3]).astype(np.float32)
        elif obs.size == 250 * 160 * 3:
            img = np.reshape(obs, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] + 0.114
        resized_img = resize(img, (self.__image_shape[1], self.__image_shape[2]))
        resized_img = np.reshape(resized_img, [self.__image_shape[1], self.__image_shape[2], 1])
        return resized_img.astype(np.uint8)
