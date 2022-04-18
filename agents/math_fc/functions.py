from torch.distributions.multivariate_normal import MultivariateNormal
from torch import zeros, eye
from singletons.Device import Device
import torch


def entropy_gaussian(log_var, sum_dims=None):
    """
    Compute the entropy of a Gaussian distribution
    :param log_var: the logarithm of the variance parameter
    :param sum_dims: the dimensions along which to sum over before to return, by default only dimension one
    :return: the entropy of a Gaussian distribution
    """
    ln2pie = 1.23247435026
    sum_dims = [1] if sum_dims is None else sum_dims
    return log_var.size()[1] * 0.5 * ln2pie + 0.5 * log_var.sum(sum_dims)


def kl_div_gaussian(mean, log_var, mean_hat, log_var_hat, sum_dims=None, displacement=0.00001):
    """
    Compute the KL-divergence between two Gaussian distributions
    :param mean: the mean of the first Gaussian distribution
    :param log_var: the logarithm of variance of the first Gaussian distribution
    :param mean_hat: the mean of the second Gaussian distribution
    :param log_var_hat: the logarithm of variance of the second Gaussian distribution
    :param sum_dims: the dimensions along which to sum over before to return, by default all of them
    :param displacement: small value to avoid dividing by zero.
    :return: the KL-divergence between the two Gaussian distributions
    """
    var = log_var.exp()
    var_hat = log_var_hat.exp()
    print("log_var_hat:")
    print(log_var_hat)
    print("var_hat:")
    print(var_hat)
    print("var:")
    print(var)
    kl_div = log_var - log_var_hat + (mean_hat - mean) ** 2 / var
    kl_div += var_hat / var

    if sum_dims is None:
        return 0.5 * kl_div.sum(dim=1).mean()
    else:
        return 0.5 * kl_div.sum(dim=sum_dims)


def log_bernoulli_with_logits(obs, alpha):
    """
    Compute the log probability of the observation (obs), given the logits (alpha), assuming
    a bernoulli distribution, c.f.
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    :param obs: the observation
    :param alpha: the logits
    :return: the log probability of the observation
    """
    out = torch.exp(alpha)
    one = torch.ones_like(out)
    out = alpha * obs - torch.log(one + out)
    return out.sum(dim=(1, 2, 3)).mean()


def reparameterize(mean, log_var):
    """
    Perform the reparameterization trick
    :param mean: the mean of the Gaussian
    :param log_var: the log of the variance of the Gaussian
    :return: a sample from the Gaussian on which back-propagation can be performed
    """
    nb_states = mean.shape[1]
    epsilon = MultivariateNormal(zeros(nb_states), eye(nb_states)).sample([mean.shape[0]]).to(Device.get())
    return epsilon * torch.exp(0.5 * log_var) + mean
