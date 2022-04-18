from math import prod
from torch import nn
from agents.layers.ConvTranspose2d import ConvTranspose2d
import torch


#
# Class implementing a deconvolution decoder for 84 by 84 images.
#
class ConvDecoder84(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the deconvolutional network.
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


#
# Class implementing a deconvolution decoder for 64 by 64 images.
#
class ConvDecoder64(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the deconvolutional network.
        self.__lin_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 1600),
            nn.ReLU(),
        )
        self.__up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(0, 0), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2), padding=(0, 0), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_shape[0], (4, 4), stride=(1, 1), padding=(0, 0), output_padding=(0, 0)),
        )

    def forward(self, x):
        """
        Compute the shape parameters of a product of beta distribution.
        :param x: a hidden state.
        :return: the shape parameters of a product of beta distribution.
        """
        x = self.__lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 5, 5))
        return self.__up_conv_net(x)


#
# Class implementing a deconvolution decoder for 64 by 64 images.
#
class ConvDecoderDAIMC(nn.Module):

    def __init__(self, n_states, image_shape):
        """
        Constructor.
        :param n_states: the number of hidden variables, i.e., number of dimension in the Gaussian.
        :param image_shape: the shape of the input images.
        """

        super().__init__()

        # Create the deconvolutional network.
        self.__lin_net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 16 * 16 * 64),
            nn.ReLU(),
            nn.Dropout()
        )
        self.__up_conv_net = nn.Sequential(
            ConvTranspose2d(64, 64, (3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(),
            ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding='same'),
            nn.ReLU(),
            ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding='same'),
            nn.ReLU(),
            ConvTranspose2d(32, image_shape[0], (3, 3), stride=(1, 1), padding='same'),
        )

    def forward(self, x):
        """
        Compute the shape parameters of a product of beta distribution.
        :param x: a hidden state.
        :return: the shape parameters of a product of beta distribution.
        """
        x = self.__lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 16, 16))
        return self.__up_conv_net(x)
