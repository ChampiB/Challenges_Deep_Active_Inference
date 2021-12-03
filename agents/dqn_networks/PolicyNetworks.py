from torch import nn, zeros
from math import prod


#
# Implement the policy network that compute the q-values.
#
class ConvPolicy(nn.Module):

    def __init__(self, image_shape, n_actions):
        """
        Constructor.
        :param image_shape: the shape of the input images.
        :param n_actions: the number of actions.
        """

        super().__init__()

        # Create convolutional part of the policy network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
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

    def __conv_output_shape(self, image_shape):
        """
        Compute the shape of the features output by the convolutional encoder.
        :param image_shape: the shape of the input image.
        :return: the shape of the features output by the convolutional encoder.
        """
        image_shape = list(image_shape)
        image_shape.insert(0, 1)
        input_image = zeros(image_shape)
        return self.__conv_net(input_image).shape

    def forward(self, x):
        """
        Compute the q-values of each possible actions.
        :param x: a observations from the environment.
        :return: the q-values.
        """
        return self.__policy_net(x)
