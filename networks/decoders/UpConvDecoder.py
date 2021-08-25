from math import prod
from torch import nn


#
# Class implementing a deconvolution decoder.
#
class UpConvDecoder(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28), conv_output_shape=(32, 3, 3)):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        :param conv_output_shape: the shape of the last convolutional layer in the encoder.
        """

        super().__init__()

        # Create the encoder network.
        self.__net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.Linear(128, prod(conv_output_shape)),
            nn.Unflatten(dim=1, unflattened_size=conv_output_shape),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_shape[0], (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Compute the mean and logarithm of the variance of the Gaussian distribution over latent variables.
        :param x: the input.
        :return: the mean and logarithm of the variance of Gaussian over hidden variables.
        """
        return self.__net(x)
