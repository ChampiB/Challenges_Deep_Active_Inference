import numpy as np
import gym
from gym import spaces
from environments.viewers.DefaultViewer import DefaultViewer


#
# This file contains the code of the random dSprites environment. The main difference
# with the standard dSprites is that actions does not move the shape around. Instead,
# a random image is returned at each time step.
#
class RandomdSpritesEnv(gym.Env):

    def __init__(self, config):
        """
        Constructor (compatible with OpenAI gym environment)
        :param config: the hydra configuration.
        """

        # Gym compatibility
        super(RandomdSpritesEnv, self).__init__()
        self.np_precision = np.float64
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=self.np_precision)

        # Initialize fields
        dataset = np.load(config["env"]["images_archive"], allow_pickle=True, encoding='latin1')
        self.images = dataset['imgs'].reshape(-1, 64, 64, 1)
        metadata = dataset['metadata'][()]
        self.s_sizes = metadata['latents_sizes']  # [1 3 6 40 32 32]
        self.s_dim = self.s_sizes.size
        self.s_bases = np.concatenate((metadata['latents_sizes'][::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.s_bases = np.squeeze(self.s_bases)  # self.s_bases = [737280 245760  40960 1024 32]
        self.s_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        self.reset()
        self.last_r = 0.0

        # Graphical interface
        self.viewer = None

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.reset_hidden_state()
        return self.current_frame()

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: the action to perform.
        :return: next observation, reward, is the trial done?, information
        """
        return self.reset(), self.last_r, True, {}

    def render(self, mode='human', close=False):
        """
        Display the current state of the environment as an image.
        :param mode: unused.
        :param close: unused.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer('Random_dSprites', self.last_r, self.current_frame())
        else:
            self.viewer.update(self.last_r, self.current_frame())

    def s_to_index(self, s):
        """
        Compute the index of the image corresponding to the state sent as parameter.
        :param s: the state whose index must be computed.
        :return: the index.
        """
        return np.dot(s, self.s_bases).astype(int)

    def current_frame(self):
        """
        Return the current frame (i.e. the current observation).
        :return: the current observation.
        """
        image = self.images[self.s_to_index(self.state)].astype(self.np_precision)
        return np.repeat(image, 3, 2) * 255.0

    def reset_hidden_state(self):
        """
        Reset the latent state, i.e, sample the a latent state randomly.
        The latent state contains:
         - a color, i.e. white
         - a shape, i.e. square, ellipse, or heart
         - a scale, i.e. 6 values linearly spaced in [0.5, 1]
         - an orientation, i.e. 40 values in [0, 2 pi]
         - a position in X, i.e. 32 values in [0, 1]
         - a position in Y, i.e. 32 values in [0, 1]
        :return: the state sampled.
        """
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        for s_i, s_size in enumerate(self.s_sizes):
            self.state[s_i] = np.random.randint(s_size)

    #
    # Getter and setter.
    #

    @property
    def y_pos(self):
        """
        Getter.
        :return: the current position of the object on the y axis.
        """
        return self.state[5]

    @y_pos.setter
    def y_pos(self, new_value):
        """
        Setter.
        :param new_value: the new position of the object on the y axis.
        :return: nothing.
        """
        self.state[5] = new_value

    @property
    def x_pos(self):
        """
        Getter.
        :return: the current position of the object on the x axis.
        """
        return self.state[4]

    @x_pos.setter
    def x_pos(self, new_value):
        """
        Setter.
        :param new_value: the new position of the object on the x axis.
        :return: nothing.
        """
        self.state[4] = new_value
