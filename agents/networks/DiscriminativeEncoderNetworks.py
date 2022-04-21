from math import prod
from agents.layers.DiagonalGaussian import DiscriminativeDiagonalGaussian as DiscriminativeGaussian
from torch import nn, zeros


#
# Class implementing a convolutional encoder for 84 by 84 images.
#
class ConvEncoder84(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28)):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the convolutional encoder network.
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

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            DiscriminativeGaussian(256, n_states)
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
        :return: the mean and logarithm of the variance of the Gaussian over latent variables.
        """
        return self.__net(x)

    def conv_output_shape(self):
        """
        Getter.
        :return:
        """
        return self.__conv_output_shape


#
# Class implementing a convolutional encoder for 64 by 64 images.
#
class ConvEncoder64(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28)):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the convolutional encoder network.
        self.__conv_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, (4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (2, 2), stride=(2, 2), padding=1),
            nn.ReLU(),
        )
        self.__conv_output_shape = self.__conv_output_shape(image_shape)
        self.__conv_output_shape = self.__conv_output_shape[1:]
        conv_output_size = prod(self.__conv_output_shape)

        # Create the linear encoder network.
        self.__linear_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            DiscriminativeGaussian(256, n_states)
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
        :return: the mean and logarithm of the variance of the Gaussian over latent variables.
        """
        return self.__net(x)

    def conv_output_shape(self):
        """
        Getter.
        :return:
        """
        return self.__conv_output_shape


#
# Class implementing a linear encoder used for the CartPole environment.
#
class CartPoleEncoder(nn.Module):

    def __init__(self, n_states, input_size):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        :param input_size: the number of input features.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            DiscriminativeGaussian(256, n_states)
        )

    def forward(self, x):
        """
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variables.
        """
        return self.__net(x)
