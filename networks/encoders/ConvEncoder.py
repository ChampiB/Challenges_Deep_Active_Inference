from math import prod
from networks.DiagonalGaussian import DiagonalGaussian as Gaussian
from torch import nn, zeros


#
# Class implementing a convolutional encoder.
#
class ConvEncoder(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28)):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the convolutional encoder network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 8, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), stride=(2, 2), padding=0),
            nn.ReLU()
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 128),
            Gaussian(128, n_states)
        )

        # Create the full encoder network.
        self.__net = nn.Sequential(
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
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variable.
        """
        return self.__net(x)

    def conv_output_shape(self):
        """
        Getter.
        :return:
        """
        return self.__conv_output_shape
