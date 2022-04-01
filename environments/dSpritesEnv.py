import numpy as np
import gym
from gym import spaces
from environments.viewers.DefaultViewer import DefaultViewer
from singletons.dSpritesDataset import DataSet


#
# This file contains the code of the dSprites environment adapted from:
# https://github.com/zfountas/deep-active-inference-mc/blob/master/src/game_environment.py
#
class dSpritesEnv(gym.Env):

    def __init__(self, config):
        """
        Constructor (compatible with OpenAI gym environment)
        :param config: the hydra configuration.
        """

        # Gym compatibility
        super(dSpritesEnv, self).__init__()
        self.np_precision = np.float64
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=self.np_precision)

        # Initialize fields
        self.repeats = config["env"]["repeats"]

        self.images, self.s_sizes, self.s_dim, self.s_bases = \
            DataSet.get(config["env"]["images_archive"])

        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        self.last_r = 0.0
        self.frame_id = 0
        self.max_episode_length = config["env"]["max_episode_length"]
        self.reset()

        # Graphical interface
        self.viewer = None

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.state = np.zeros(self.s_dim, dtype=self.np_precision)
        self.last_r = 0.0
        self.frame_id = 0
        self.reset_hidden_state()
        return self.current_frame()

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: the action to perform.
        :return: next observation, reward, is the trial done?, information
        """
        # Increase the frame index, that count the number of frames since
        # the beginning of the episode.
        self.frame_id += 1

        # Simulate the action requested by the user.
        actions_fn = [self.down, self.up, self.left, self.right]
        if not isinstance(action, int):
            action = action.item()
        for i in range(self.repeats):
            if action < 0 or action > 3:
                exit('Invalid action.')
            if actions_fn[action]():
                return self.current_frame(), self.last_r, True, {}

        # Make sure the environment is reset if the maximum number of steps in
        # the episode has been reached.
        if self.frame_id >= self.max_episode_length:
            return self.current_frame(), -1.0, True, {}
        else:
            return self.current_frame(), self.last_r, False, {}

    def render(self, mode='human', close=False):
        """
        Display the current state of the environment as an image.
        :param mode: unused.
        :param close: unused.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer('dSprites', self.last_r, self.current_frame())
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
    # Actions
    #

    def down(self):
        """
        Execute the action "down" in the environment.
        :return: true if the object crossed the bottom line.
        """

        # Increase y coordinate
        self.y_pos += 1.0

        # If the object did not cross the bottom line, return false
        if self.y_pos < 32:
            return False

        # If the object crossed the bottom line, then:
        # compute the reward, generate a new image and return true.
        if self.state[1] < 0.5:
            self.last_r = self.compute_square_reward()
        else:
            self.last_r = self.compute_non_square_reward()
        self.y_pos -= 1.0
        return True

    def up(self):
        """
        Execute the action "up" in the environment.
        :return: false (the object never cross the bottom line when moving up).
        """
        if self.y_pos > 0:
            self.y_pos -= 1.0
        return False

    def right(self):
        """
        Execute the action "right" in the environment.
        :return: false (the object never cross the bottom line when moving left).
        """
        if self.x_pos < 31:
            self.x_pos += 1.0
        return False

    def left(self):
        """
        Execute the action "left" in the environment.
        :return: false (the object never cross the bottom line when moving right).
        """
        if self.x_pos > 0:
            self.x_pos -= 1.0
        return False

    #
    # Reward computation
    #

    def compute_square_reward(self):
        """
        Compute the obtained by the agent when a square crosses the bottom wall.
        :return: the reward.
        """
        if self.x_pos > 15:
            return float(15.0 - self.x_pos) / 16.0
        else:
            return float(16.0 - self.x_pos) / 16.0

    def compute_non_square_reward(self):
        """
        Compute the obtained by the agent when a ellipse or heart crosses the bottom wall.
        :return: the reward.
        """
        if self.x_pos > 15:
            return float(self.x_pos - 15.0) / 16.0
        else:
            return float(self.x_pos - 16.0) / 16.0

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
