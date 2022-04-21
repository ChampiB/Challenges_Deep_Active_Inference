from torch import nn, cat
import torch
from torch.nn.functional import one_hot
from agents.layers.DiagonalGaussian import DiagonalGaussian as Gaussian


#
# Class implementing a network modeling the temporal transition between hidden state.
#
class LinearRelu3x100(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the transition network.
        self.__net = nn.Sequential(
            nn.Linear(n_states + n_actions, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            Gaussian(100, n_states)
        )

        # Remember the number of actions.
        self.n_actions = n_actions

    def forward(self, states, actions):
        """
        Forward pass through the transition network.
        :param states: the input states.
        :param actions: the input actions.
        :return: the mean and log of the variance of the Gaussian over hidden state.
        """
        actions = one_hot(actions.to(torch.int64), self.n_actions)
        x = cat((states, actions), dim=1)
        return self.__net(x)


#
# Class implementing a network modeling the temporal transition between hidden state.
#
class LinearReluDropout4x512(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the transition network.
        self.__net = nn.Sequential(
            nn.Linear(n_states + n_actions, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            Gaussian(512, n_states)
        )

        # Remember the number of actions.
        self.n_actions = n_actions

    def forward(self, states, actions):
        """
        Forward pass through the transition network.
        :param states: the input states.
        :param actions: the input actions.
        :return: the mean and log of the variance of the Gaussian over hidden state.
        """
        actions = one_hot(actions.to(torch.int64), self.n_actions)
        x = cat((states, actions), dim=1)
        return self.__net(x)
