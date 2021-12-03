from math import prod
from torch import nn


#
# Class implementing a deconvolution decoder.
#
class ConvDecoder(nn.Module):

    def __init__(self, n_states, image_shape=(1, 28, 28)):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        :param conv_output_shape: the shape of the last convolutional layer in the encoder.
        """

        super().__init__()

        # Create the encoder network.
        self.__lin_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.__compat_net = None
        self.__up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_shape[0], (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0)),
            nn.Sigmoid()
        )

    def build(self, conv_output_shape):
        """
        Build the network bridging the gap between the .
        :param conv_output_shape:
        :return: nothing.
        """
        self.__compat_net = nn.Sequential(
            nn.Linear(256, prod(conv_output_shape)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Unflatten(dim=1, unflattened_size=conv_output_shape)
        )

    def forward(self, x):
        """
        Compute the shape parameters of a product of beta distribution.
        :param x: a hidden state.
        :return: the shape parameters of a product of beta distribution.
        """
        if self.__compat_net is None:
            raise Exception("Error: the decoder was not build().")
        x = self.__lin_net(x)
        x = self.__compat_net(x)
        return self.__up_conv_net(x)
