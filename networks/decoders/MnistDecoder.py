from torch import nn


#
# Class implementing a deconvolution decoder.
#
class MnistDecoder(nn.Module):

    def __init__(self, n_states):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 3 * 3 * 64),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Compute the mean and logarithm of the variance of the Gaussian distribution over latent variables.
        :param x: the input.
        :return: the mean and logarithm of the variance of Gaussian over hidden variables.
        """
        return self.__net(x)
