import numpy as np
import gym
from gym import spaces
import torch
from PIL import Image
from environments.viewers.DefaultViewer import DefaultViewer


#
# This file contains the code of the Maze environment.
#
class MazeEnv(gym.Env):

    def __init__(self, config):
        """
        Constructor (compatible with OpenAI gym environment)
        :param config: the hydra config.
        """

        # Gym compatibility
        super(MazeEnv, self).__init__()
        self.np_precision = np.float64
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(28, 28, 1), dtype=self.np_precision)

        # Initialize fields
        self.exit_pos = [-1, -1]
        self.agent_pos = [-1, -1]
        self.last_r = 0  # Last reward
        self.scale = 10  # How big should each cell be represented (in pixels) ?

        # Load maze from file
        file = open(config["env"]["maze_file"], "r")
        maze = file.readlines()

        h, w = maze[0].split(" ")
        h = int(h)
        w = int(w)
        maze = maze[1:h+1]
        maze = [line.rstrip('\n') for line in maze]
        self.maze = torch.ones((h, w))

        for i in range(0, h):
            for j in range(0, w):
                if maze[i][j] == 'W':
                    self.maze[i][j] = 1
                elif maze[i][j] == '.':
                    self.maze[i][j] = 0
                elif maze[i][j] == 'E':
                    self.maze[i][j] = 0
                    self.exit_pos[0] = i
                    self.exit_pos[1] = j
                elif maze[i][j] == 'S':
                    self.maze[i][j] = 0
                    self.agent_pos[0] = i
                    self.agent_pos[1] = j
                else:
                    raise Exception("Invalid file format: '" + config["env"]["maze_file"] + "'")

        self.agent_initial_pos = [self.agent_pos[0], self.agent_pos[1]]
        self.reset()

        # Graphical interface
        self.viewer = None

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.agent_pos[0] = self.agent_initial_pos[0]
        self.agent_pos[1] = self.agent_initial_pos[1]
        return self.current_frame()

    def step(self, action):
        """
        Execute one action within the environment.
        :param action: the action to perform, i.e. UP, DOWN, LEFT, or RIGHT.
        :return: next observation, reward, is the trial done?, information
        """
        actions_fn = [self.up, self.down, self.right, self.left]
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action > 3:
            exit('Invalid action.')
        done = actions_fn[action]()
        self.last_r = int(self.is_solved())
        return self.current_frame(), self.last_r, done, {}

    def render(self, mode='human', close=False):
        """
        Display the current state of the environment as an image.
        :param mode: unused.
        :param close: unused.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer("Maze", self.last_r, self.current_frame(), resize_type=Image.NEAREST)
        else:
            self.viewer.update(self.last_r, self.current_frame())

    def current_frame(self):
        """
        Return the current frame (i.e. the current observation).
        :return: the current observation.
        """
        image = 1 - self.maze
        image[self.agent_pos[0]][self.agent_pos[1]] = 0.5
        image = image.numpy().astype(self.np_precision)
        return np.repeat(np.expand_dims(image, axis=2), 3, 2) * 255

    def is_solved(self):
        """
        Answer the question: Has the agent reached the exit?
        :return: true if the agent reached the exit, false otherwise.
        """
        return self.agent_pos[0] == self.exit_pos[0] and self.agent_pos[1] == self.exit_pos[1]

    #
    # Actions
    #

    def up(self):
        """
        Perform the action "going up" in the environment.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        if self.agent_pos[0] - 1 >= 0 and self.maze[self.agent_pos[0] - 1][self.agent_pos[1]] == 0:
            self.agent_pos[0] -= 1
        return self.is_solved()

    def down(self):
        """
        Perform the action "going down" in the environment.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        if self.agent_pos[0] + 1 < self.maze.shape[0] and self.maze[self.agent_pos[0] + 1][self.agent_pos[1]] == 0:
            self.agent_pos[0] += 1
        return self.is_solved()

    def left(self):
        """
        Perform the action "going left" in the environment.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        if self.agent_pos[1] - 1 >= 0 and self.maze[self.agent_pos[0]][self.agent_pos[1] - 1] == 0:
            self.agent_pos[1] -= 1
        return self.is_solved()

    def right(self):
        """
        Perform the action "going right" in the environment.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        if self.agent_pos[1] + 1 < self.maze.shape[1] and self.maze[self.agent_pos[0]][self.agent_pos[1] + 1] == 0:
            self.agent_pos[1] += 1
        return self.is_solved()
