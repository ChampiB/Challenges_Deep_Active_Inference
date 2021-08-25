from torch import nn


#
# Class implementing a linear decoder.
#
class LinearDecoder(nn.Module):

    def __init__(self, n_states):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 300),
            nn.ReLU(),
            nn.Linear(300, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(32, 28, 28)),
        )

    def forward(self, x):
        """
        Compute the mean and logarithm of the variance of the Gaussian distribution over latent variables.
        :param x: the input.
        :return: the mean and logarithm of the variance of Gaussian over hidden variables.
        """
        return self.__net(x)
