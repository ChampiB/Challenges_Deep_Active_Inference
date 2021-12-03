import numpy as np
import gym
from gym import spaces
from environments.viewers.DefaultViewer import DefaultViewer
import torchvision.datasets as datasets


#
# This file contains the code of the MNIST environment.
#
class MnistEnv(gym.Env):

    def __init__(self, config):
        """
        Constructor (compatible with OpenAI gym environment)
        :param config: the hydra config.
        """

        # Gym compatibility
        super(MnistEnv, self).__init__()
        self.np_precision = np.float64
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(28, 28, 1), dtype=self.np_precision)

        # Initialize fields
        dataset = datasets.MNIST(root=config["env"]["images_dir"], train=True, download=True, transform=None)
        self.images = dataset.data
        self.images = self.images.reshape(-1, 28, 28, 1)
        self.targets = dataset.targets
        self.n_images = len(dataset)
        self.index = 0
        self.last_r = 0
        self.reset()

        # Graphical interface
        self.viewer = None

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.index = np.random.randint(self.n_images)
        return self.current_frame()

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: the action to perform, i.e. either even (0) or odd (1) prediction.
        :return: next observation, reward, is the trial done?, information
        """
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action > 1:
            exit('Invalid action.')
        self.last_r = int(self.targets[self.index] % 2 == action)
        return self.reset(), self.last_r, False, {}

    def render(self, mode='human', close=False):
        """
        Display the current state of the environment as an image.
        :param mode: unused.
        :param close: unused.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer("MNIST", self.last_r, self.current_frame())
        else:
            self.viewer.update(self.last_r, self.current_frame())

    def current_frame(self):
        """
        Return the current frame (i.e. the current observation).
        :return: the current observation.
        """
        image = self.images[self.index].numpy().astype(self.np_precision)
        return np.repeat(image, 3, 2)
