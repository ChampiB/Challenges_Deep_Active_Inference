from torch import nn


#
# Class implementing a network that maps a vector of size n into two vectors representing the mean
# and variance of a Gaussian with diagonal covariance matrix.
#
class DiagonalGaussian(nn.Module):

    def __init__(self, input_size, nb_components):
        """
        Constructor.
        :param input_size: size of the vector send as input of the layer.
        :param nb_components: the number of components of the diagonal Gaussian.
        """
        super().__init__()
        self.__mean = nn.Sequential(
            nn.Linear(input_size, nb_components)
        )
        self.__log_var = nn.Sequential(
            nn.Linear(input_size, nb_components),
        )

    def forward(self, x):
        """
        Compute the mean and the variance of the diagonal Gaussian (DG).
        :param x: the input vector
        :return: the mean and the log of the variance of the DG.
        """
        return self.__mean(x), self.__log_var(x)


#
# Class implementing a network that maps a vector of size n into two vectors representing the mean
# and variance of a Gaussian with diagonal covariance matrix, and a scalar between zero and one
# that discriminates between true and fake inputs.
#
class DiscriminativeDiagonalGaussian(nn.Module):

    def __init__(self, input_size, nb_components):
        """
        Constructor.
        :param input_size: size of the vector send as input of the layer.
        :param nb_components: the number of components of the diagonal Gaussian.
        """
        super().__init__()
        self.__mean = nn.Sequential(
            nn.Linear(input_size, nb_components)
        )
        self.__log_var = nn.Sequential(
            nn.Linear(input_size, nb_components),
        )
        self.__discriminator = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Compute the mean and the variance of the diagonal Gaussian (DG).
        :param x: the input vector
        :return: the mean and the log of the variance of the DG.
        """
        return self.__mean(x), self.__log_var(x), self.__discriminator(x)
