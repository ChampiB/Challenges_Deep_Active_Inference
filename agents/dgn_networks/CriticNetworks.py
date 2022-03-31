from torch import nn


#
# Class implementing a network modeling the cost of each action given a state.
#
class LinearRelu(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the critic network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the critic network.
        :param states: the input states.
        :return: the cost of performing each action in that state.
        """
        return self.__net(states)
