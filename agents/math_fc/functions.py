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


def kl_div_categorical(pi_hat, pi):
    """
    Compute the KL-divergence between two categorical distribution.
    :param pi_hat: the parameters of the first categorical distribution.
    :param pi: the parameters of the second categorical distribution.
    :return: the KL-divergence.
    """
    shift = 0.00001
    kl = pi_hat * ((pi_hat + shift).log() - (pi + shift).log())
    return kl.sum()


def kl_div_gaussian(mean_hat, log_var_hat, mean, log_var, sum_dims=None):
    """
    Compute the KL-divergence between two Gaussian distributions
    :param mean: the mean of the first Gaussian distribution
    :param log_var: the logarithm of variance of the first Gaussian distribution
    :param mean_hat: the mean of the second Gaussian distribution
    :param log_var_hat: the logarithm of variance of the second Gaussian distribution
    :param sum_dims: the dimensions along which to sum over before to return, by default all of them
    :return: the KL-divergence between the two Gaussian distributions
    """
    var = torch.clamp(log_var, max=10).exp()  # Clamp to avoid overflow of exponential
    var_hat = torch.clamp(log_var_hat, max=10).exp()  # Clamp to avoid overflow of exponential
    kl_div = log_var - log_var_hat + var_hat / var + (mean_hat - mean) ** 2 / var

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


def compute_info_gain(g_value, mean_hat, log_var_hat, mean, log_var, shift=-20):
    """
    Compute the efe.
    :param g_value: the definition of the efe to use, i.e., reward, efe_0, efe_1,
        efe_2, efe_3, befe_0, befe_1, befe_2, and befe_3.
    :param mean_hat: the mean from the encoder.
    :param log_var_hat: the log variance from the encoder.
    :param mean: the mean from the transition.
    :param log_var: the log variance from the transition.
    :param shift: the shift to apply if efe must be bounded.
    :return: the efe.
    """
    efe = torch.zeros([1]).to(Device.get())
    if g_value[-5:] == "efe_0":  # G^1
        efe = entropy_gaussian(log_var_hat) - entropy_gaussian(log_var)
    elif g_value[-5:] == "efe_1":  # G^3
        efe = kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
    elif g_value[-5:] == "efe_2":  # G^2
        efe = entropy_gaussian(log_var) - entropy_gaussian(log_var_hat)
    elif g_value[-5:] == "efe_3":  # G
        efe = kl_div_gaussian(mean, log_var, mean_hat, log_var_hat)
    elif g_value[0:1] == "b":
        efe = torch.sigmoid(efe + shift)
    return efe
