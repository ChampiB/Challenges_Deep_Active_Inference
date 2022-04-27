from torch import nn
import itertools


#
# Class implementing a network that maps a vector of size "m" into "n" sets of
# two vectors representing the mean and variance of "n" Gaussian with diagonal
# covariance matrix.
#
class DiagonalGaussianNLS(nn.Module):

    def __init__(self, input_size, nb_components):
        """
        Constructor.
        :param input_size: size of the vector send as input of the layer.
        :param nb_components: a list whose i-th element indicates the
        number of components of the i-th diagonal Gaussian.
        """
        super().__init__()
        self.n_latent_spaces = len(nb_components)
        self.means = []
        self.log_vars = []
        for i in range(self.n_latent_spaces):
            self.means.append(nn.Sequential(
                nn.Linear(input_size, nb_components[i])
            ))
            self.log_vars.append(nn.Sequential(
                nn.Linear(input_size, nb_components[i])
            ))

    def forward(self, x):
        """
        Compute the mean and the variance of the diagonal Gaussian (DG).
        :param x: the input vector
        :return: the mean and the log of the variance of the DG.
        """
        res = [(self.means[i](x), self.log_vars[i](x)) for i in range(self.n_latent_spaces)]
        return list(itertools.chain.from_iterable(res))
