from torch import nn


#
# Class implementing a network modeling a state-action policy.
#
class PolicyNetwork(nn.Module):

    def __init__(self, n_states, n_actions):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param n_actions: the number of allowable actions.
        """

        super().__init__()

        # Create the policy network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
        )

    def forward(self, states):
        """
        Forward pass through the policy network.
        :param states: the input states.
        :return: the log probability of each action.
        """
        return self.__net(states)
