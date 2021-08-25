from networks.DiagonalGaussian import DiagonalGaussian as Gaussian
from torch import nn


#
# Class implementing a linear encoder.
#
class LinearEncoder(nn.Module):

    def __init__(self, n_states):
        """
        Constructor.
        :param n_states: the number of components of the Gaussian over latent variables.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            Gaussian(300, n_states)
        )

    def forward(self, x):
        """
        Forward pass through this encoder.
        :param x: the input.
        :return: the mean and logarithm of the variance of the Gaussian over latent variable.
        """
        return self.__net(x)
