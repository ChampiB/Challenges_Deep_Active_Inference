from torch import nn


#
# Class implementing a network that maps a vector of size "m" into two sets of
# two vectors representing the mean and variance of two Gaussian with diagonal
# covariance matrix.
#
class DiagonalGaussian2LS(nn.Module):

    def __init__(self, input_size, nb_components):
        """
        Constructor.
        :param input_size: size of the vector send as input of the layer.
        :param nb_components: a list whose i-th element indicates the
        number of components of the i-th diagonal Gaussian.
        """
        super().__init__()
        self.n_latent_spaces = len(nb_components)
        self.mean_1 = nn.Linear(input_size, nb_components[0])
        self.mean_2 = nn.Linear(input_size, nb_components[1])
        self.log_var_1 = nn.Linear(input_size, nb_components[0])
        self.log_var_2 = nn.Linear(input_size, nb_components[1])

    def forward(self, x):
        """
        Compute the mean and the variance of the diagonal Gaussian (DG).
        :param x: the input vector
        :return: the mean and the log of the variance of the DG.
        """
        return self.mean_1(x), self.log_var_1(x), self.mean_2(x), self.log_var_2(x)
