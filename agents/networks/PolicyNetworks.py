from torch import nn, zeros
from math import prod


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


#
# Class implementing a network modeling the posterior over actions given a state.
#
class LinearRelu4x100(nn.Module):

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


#
# Implement the policy network that compute the q-values.
#
class ConvPolicy64(nn.Module):

    def __init__(self, images_shape, n_actions):
        """
        Constructor.
        :param images_shape: the shape of the input images.
        :param n_actions: the number of actions.
        """

        super().__init__()

        # Create convolutional part of the policy network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(images_shape[0], 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(images_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear part of the policy network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=1),
        )

        # Create the full policy network.
        self.__policy_net = nn.Sequential(
            self.__conv_net,
            self.__linear_net
        )

    def __conv_output_shape(self, images_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param images_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        images_shape = list(images_shape)
        images_shape.insert(0, 1)
        input_image = zeros(images_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Compute the q-values of each possible actions.
        :param x: an observations from the environment.
        :return: the q-values.
        """
        return self.__policy_net(x)
