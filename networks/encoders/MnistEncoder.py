from networks.DiagonalGaussian import DiagonalGaussian as Gaussian
from torch import nn


#
# Class implementing a convolutional encoder.
#
class MnistEncoder(nn.Module):

    def __init__(self, n_states):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            Gaussian(128, n_states)
        )

    def forward(self, x):
        """
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variable.
        """
        return self.__net(x)
